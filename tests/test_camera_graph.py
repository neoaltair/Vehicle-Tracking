import numpy as np

from incident_search.eval.protocol import assert_identity_disjoint, make_identity_splits
from incident_search.graph.camera_graph import (
    CameraTransitionGraph,
    EdgeTravelTimeModel,
    build_camera_graph_from_training_tracklets,
    extract_transitions_from_tracklets,
)
from incident_search.io.schema import Tracklet
from incident_search.search.spatiotemporal import (
    build_graph_prior_matrix,
    compute_fused_scores,
)


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


def test_edge_travel_time_model_pdf():
    model = EdgeTravelTimeModel(
        src_camera="c001",
        dst_camera="c002",
        sample_count=10,
        mean_dt=30.0,
        std_dt=5.0,
        min_dt=20.0,
        max_dt=40.0,
        fitted_shape=0.15,
        fitted_scale=30.0,
    )
    # Peak should be around 30.0
    p30 = model.pdf(30.0)
    p100 = model.pdf(100.0)
    p_neg = model.pdf(-5.0)

    assert p30 > 0.0
    assert p30 > p100
    assert p_neg == 0.0


def test_extract_transitions_and_build_graph():
    # Vehicle v1: c001 [10-20] -> c002 [30-40] (dt = 30 - 20 = 10)
    # Vehicle v1: c002 [30-40] -> c003 [60-70] (dt = 60 - 40 = 20)
    # Vehicle v2: c001 [50-60] -> c002 [72-82] (dt = 72 - 60 = 12)
    tracklets = [
        _make_trk("t1", "c001", 10.0, 20.0, "v1"),
        _make_trk("t2", "c002", 30.0, 40.0, "v1"),
        _make_trk("t3", "c003", 60.0, 70.0, "v1"),
        _make_trk("t4", "c001", 50.0, 60.0, "v2"),
        _make_trk("t5", "c002", 72.0, 82.0, "v2"),
    ]
    transitions = extract_transitions_from_tracklets(tracklets)
    assert len(transitions) == 3

    # Build graph with min_edge_samples=2
    # Edge c001 -> c002 has 2 samples (10 and 12).
    # Edge c002 -> c003 has 1 sample (20).
    graph = build_camera_graph_from_training_tracklets(tracklets, min_edge_samples=2)

    assert graph.has_edge("c001", "c002")
    assert not graph.has_edge("c002", "c003")  # Only 1 sample, below min_edge_samples

    # Model for c001 -> c002
    prior_good = graph.get_prior("c001", "c002", 11.0)
    prior_bad = graph.get_prior("c001", "c002", 100.0)
    assert prior_good > prior_bad


def test_leakage_prevention_split_isolation():
    """Verify that test identities never influence the transition graph."""
    # 10 vehicles
    all_trks = []
    for i in range(10):
        all_trks.extend([
            _make_trk(f"t_c1_{i}", "c001", float(i * 10), float(i * 10 + 5), f"veh_{i}"),
            _make_trk(f"t_c2_{i}", "c002", float(i * 10 + 20), float(i * 10 + 25), f"veh_{i}"),
        ])
    splits = make_identity_splits(all_trks, train_fraction=0.6, val_fraction=0.2, seed=42)
    assert_identity_disjoint(splits)

    # Filter training tracklets only
    train_trks = [t for t in all_trks if t.gt_vehicle_id in splits.train]

    # Build graph exclusively on train
    train_graph = build_camera_graph_from_training_tracklets(train_trks, min_edge_samples=1)

    # Check edge sample count matches train count only
    edge_model = train_graph.edge_models[("c001", "c002")]
    assert edge_model.sample_count == len(splits.train)
    assert edge_model.sample_count < len(all_trks) // 2


def test_spatiotemporal_prior_matrix_and_fused_scores():
    # Candidate at c001 (end time 20.0), Query at c002 (start time 35.0) -> dt = 15.0
    cand = _make_trk("cand", "c001", 10.0, 20.0, "v1")
    query = _make_trk("q", "c002", 35.0, 45.0, "v1")

    graph = CameraTransitionGraph(min_edge_samples=1)
    graph.add_transition("c001", "c002", 15.0)
    graph.fit_models()

    priors, connected, topology = build_graph_prior_matrix([query], [cand], graph)
    assert priors.shape == (1, 1)
    assert priors[0, 0] > 0.0
    assert connected[0, 0]
    assert topology[0, 0]

    # Fused score
    app_scores = np.array([[0.8]], dtype=np.float32)
    fused = compute_fused_scores(app_scores, priors, alpha=0.1)
    assert fused[0, 0] > 0.8  # Prior boosts score
