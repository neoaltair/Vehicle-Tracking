import numpy as np

from incident_search.degrade.transforms import (
    apply_crop,
    apply_downscale,
    apply_gaussian_blur,
    apply_occlusion,
)
from incident_search.eval.protocol import (
    assert_identity_disjoint,
    build_upstream_protocol,
    make_identity_splits,
    pruning_reduction,
    true_match_survival,
)
from incident_search.graph.camera_graph import build_camera_graph_from_training_tracklets
from incident_search.io.schema import Tracklet


def _make_trk(trk_id: str, cam: str, start: float, end: float, vid: str) -> Tracklet:
    return Tracklet(
        tracklet_id=trk_id,
        camera_id=cam,
        start_time_s=start,
        end_time_s=end,
        frames=[1, 2],
        boxes=[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
        scores=[1.0, 1.0],
        vehicle_class="car",
        gt_vehicle_id=vid,
    )


def test_train_val_test_isolation():
    """Verify train, validation, and test splits are strictly disjoint."""
    tracklets = [_make_trk(f"t{i}", f"c{i%3}", float(i*10), float(i*10+5), f"veh_{i}") for i in range(30)]
    splits = make_identity_splits(tracklets, train_fraction=0.6, val_fraction=0.2, seed=123)
    assert_identity_disjoint(splits)

    # Intersection checks
    assert len(splits.train & splits.val) == 0
    assert len(splits.train & splits.test) == 0
    assert len(splits.val & splits.test) == 0
    assert len(splits.train) + len(splits.val) + len(splits.test) == 30


def test_temporal_filtering_and_survival():
    """Verify temporal window candidate filtering and survival metric."""
    q = _make_trk("q", "c002", 100.0, 110.0, "v1")
    # c1 is 20s prior (inside 60s window) -> survives
    c1 = _make_trk("c1", "c001", 70.0, 80.0, "v1")
    # c2 is 200s prior (outside 60s window) -> pruned
    c2 = _make_trk("c2", "c003", 0.0, 10.0, "v1")

    gallery = [c1, c2]
    base_mask, pos_mask = build_upstream_protocol([q], gallery, time_window_s=None)
    win_mask, _ = build_upstream_protocol([q], gallery, time_window_s=60.0)

    assert base_mask.tolist() == [[True, True]]
    assert win_mask.tolist() == [[True, False]]

    reduction = pruning_reduction(base_mask, win_mask)
    survival = true_match_survival(pos_mask, win_mask)
    assert reduction == 0.5  # 1 out of 2 candidates pruned
    assert survival == 1.0   # c1 survived


def test_validation_hyperparameter_selection():
    """Simulate hyperparameter grid search on validation data."""
    # 2 validation queries
    val_trks = [
        _make_trk("vq1", "c002", 50.0, 60.0, "val_v1"),
        _make_trk("vg1", "c001", 30.0, 40.0, "val_v1"),  # dt = 10s
        _make_trk("vq2", "c002", 90.0, 100.0, "val_v2"),
        _make_trk("vg2", "c001", 20.0, 30.0, "val_v2"),  # dt = 60s
    ]
    # Synthetic embedding map
    emb_map = {
        "vq1": np.array([1.0, 0.0], dtype=np.float32),
        "vg1": np.array([0.9, 0.1], dtype=np.float32),
        "vq2": np.array([0.0, 1.0], dtype=np.float32),
        "vg2": np.array([0.1, 0.9], dtype=np.float32),
    }

    # Graph on train dummy
    train_trks = [
        _make_trk("t1", "c001", 10.0, 20.0, "tr_v1"),
        _make_trk("t2", "c002", 30.0, 40.0, "tr_v1"),  # dt = 10s
    ] * 5
    graph = build_camera_graph_from_training_tracklets(train_trks, min_edge_samples=5)

    from incident_search.eval.protocol import SplitIds
    from scripts.tune_m4_hyperparameters import run_val_grid_search

    splits = SplitIds(train={"tr_v1"}, val={"val_v1", "val_v2"}, test=set())
    best_cfg, results = run_val_grid_search(
        val_trks, emb_map, graph, splits,
        time_windows=[60.0, 120.0],
        alphas=[0.05, 0.10],
    )
    assert best_cfg is not None
    assert "val_mAP" in best_cfg
    assert len(results) == 4


def test_degradation_transforms():
    """Test deterministic degradation transforms."""
    img = np.full((64, 64, 3), 200, dtype=np.uint8)

    # Downscale
    down = apply_downscale(img, factor=0.5)
    assert down.shape == (64, 64, 3)

    # Blur
    blur = apply_gaussian_blur(img, sigma=2.0)
    assert blur.shape == (64, 64, 3)

    # Crop
    crop = apply_crop(img, keep_ratio=0.5)
    assert crop.shape == (64, 64, 3)

    # Occlusion
    occ = apply_occlusion(img, occlusion_ratio=0.25, seed=42)
    assert occ.shape == (64, 64, 3)
    assert np.any(occ == 128)  # Patch exists


def test_degraded_embedding_differs_from_clean():
    """Sanity check: verify FastReID embedding of degraded image differs from clean."""
    from incident_search.degrade.transforms import degrade_crop
    from incident_search.reid.extractor import FastReIDExtractor

    extractor = FastReIDExtractor(device="cpu")
    # Generate non-trivial image with color patterns
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[20:100, 20:100, 0] = 220
    img[40:80, 40:80, 1] = 180
    img[60:120, 60:120, 2] = 240

    clean_emb = extractor.embed_crops([img])[0]

    # Test each degradation family with significant severity
    for family, sev in [
        ("downscale", 0.125),
        ("blur", 4.0),
        ("crop", 0.25),
        ("occlusion", 0.50),
    ]:
        deg_img = degrade_crop(img, family, sev, seed=42)
        deg_emb = extractor.embed_crops([deg_img])[0]
        cosine_sim = float(np.dot(clean_emb, deg_emb))
        # Degraded embedding must differ from clean (cosine similarity < 0.999)
        assert cosine_sim < 0.999, f"{family} produced identical embedding to clean"

