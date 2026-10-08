#!/usr/bin/env python3
"""Run controlled query degradation evaluation on the frozen test split.

Compares:
1. Appearance-only retrieval
2. Spatio-temporal retrieval (fused prior + window)

Under query degradations:
- Downscaling: factors {1.0, 0.5, 0.25, 0.125}
- Gaussian blur: sigmas {0.0, 1.0, 2.0, 4.0}
- Crop (central area): ratios {1.0, 0.75, 0.50, 0.25}
- Occlusion: ratios {0.0, 0.25, 0.50} with seeds {0, 1, 2}

Bug fix (post-M4-freeze correction):
  The original script relied on tracklet.crop_paths which is always empty for
  GT tracklets loaded from disk (crops are not saved during GT loading).
  The correct pipeline is:
    1. Read video file for the query camera.
    2. Extract raw crop from GT box + frame ID.
    3. Apply degradation to the raw crop.
    4. Run FastReID on the degraded crop → new embedding.
  Gallery embeddings remain clean (loaded from cache).

Outputs:
  outputs/m4_degradation/degradation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path even if package isn't installed in editable mode in Colab
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from incident_search.degrade.transforms import degrade_crop  # noqa: E402
from incident_search.eval.metrics import evaluate_retrieval  # noqa: E402
from incident_search.eval.protocol import (  # noqa: E402
    assert_identity_disjoint,
    build_upstream_protocol,
    filter_tracklets_by_identities,
    make_identity_splits,
)
from incident_search.graph.camera_graph import (  # noqa: E402
    build_camera_graph_from_training_tracklets,
)
from incident_search.io.cityflow import (  # noqa: E402
    discover_cityflow_cameras,
    load_cityflow_gt_tracklets,
)
from incident_search.reid.extractor import FastReIDExtractor, aggregate_embeddings  # noqa: E402
from incident_search.search.spatiotemporal import (  # noqa: E402
    build_graph_prior_matrix,
    compute_fused_scores,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEGRADATION_SPECS = [
    ("downscale", [1.0, 0.5, 0.25, 0.125]),
    ("blur", [0.0, 1.0, 2.0, 4.0]),
    ("crop", [1.0, 0.75, 0.50, 0.25]),
    ("occlusion", [0.0, 0.25, 0.50]),
]

# Frozen M4 hyperparameters — must match best_config.json
FROZEN_TIME_WINDOW_S = 60.0
FROZEN_ALPHA = 0.05
FROZEN_MIN_EDGE_SAMPLES = 5


def _build_video_map(root: str | Path) -> dict[str, Path]:
    """Return {scoped_camera_id: video_path} for all cameras under root."""
    cameras = discover_cityflow_cameras(root)
    result: dict[str, Path] = {}
    for cam in cameras:
        if cam.video_path is not None:
            result[cam.camera_id] = cam.video_path
    return result


def _extract_crops_from_video(
    tracklet,
    video_path: Path,
    k: int = 8,
) -> list[np.ndarray]:
    """Extract up to k quality-sampled raw crops (BGR) from the tracklet GT boxes."""
    n = len(tracklet.frames)
    boxes = tracklet.boxes
    scores = tracklet.scores

    qualities = []
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        q = scores[i] * float(np.log1p(area))
        qualities.append((q, i))

    qualities.sort(key=lambda x: x[0], reverse=True)
    selected = sorted([idx for _, idx in qualities[:k]])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crops: list[np.ndarray] = []

    try:
        for idx in selected:
            frame_id = tracklet.frames[idx]
            box = boxes[idx]
            zero_based = max(0, int(frame_id) - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
            ret, frame = cap.read()
            if not ret:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, min(width - 1, int(round(x1))))
            y1 = max(0, min(height - 1, int(round(y1))))
            x2 = max(x1 + 1, min(width, int(round(x2))))
            y2 = max(y1 + 1, min(height, int(round(y2))))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    finally:
        cap.release()

    return crops


def extract_degraded_query_embeddings(
    queries: list,
    extractor: FastReIDExtractor,
    video_map: dict[str, Path],
    family: str,
    severity: float,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Extract degraded embeddings for queries only. Gallery remains clean.

    Pipeline (correct):
      raw crop from video → apply degradation → FastReID → L2 normalize → aggregate.

    If a video file cannot be found, the tracklet is skipped (no fallback to
    clean embeddings — this is intentional so the degradation metric is honest).
    """
    query_embs: dict[str, np.ndarray] = {}
    skipped = 0
    for q in queries:
        video_path = video_map.get(q.camera_id)
        if video_path is None:
            skipped += 1
            continue
        raw_crops = _extract_crops_from_video(q, video_path)
        if not raw_crops:
            skipped += 1
            continue
        degraded_imgs = [degrade_crop(img, family, severity, seed=seed) for img in raw_crops]
        frame_embs = extractor.embed_crops(degraded_imgs)
        trk_emb = aggregate_embeddings(frame_embs, method="mean")
        query_embs[q.tracklet_id] = trk_emb

    if skipped:
        logger.warning(
            "Skipped %d / %d queries (no video file or no extractable crops).",
            skipped,
            len(queries),
        )
    return query_embs


def sanity_check_degradation_differs(
    queries: list,
    extractor: FastReIDExtractor,
    video_map: dict[str, Path],
    clean_embedding_map: dict[str, np.ndarray],
) -> None:
    """Assert that at least one degraded query embedding differs from its clean embedding.

    Uses heavy downscaling (factor=0.125) on a single query as a cheap probe.
    Raises RuntimeError if all degraded embeddings are identical to clean ones.
    """
    logger.info("Running degradation sanity check (downscale factor=0.125 on first 5 queries)...")
    differences: list[float] = []
    for q in queries[:5]:
        if q.tracklet_id not in clean_embedding_map:
            continue
        video_path = video_map.get(q.camera_id)
        if video_path is None:
            continue
        raw_crops = _extract_crops_from_video(q, video_path, k=2)
        if not raw_crops:
            continue
        degraded_imgs = [degrade_crop(img, "downscale", 0.125) for img in raw_crops]
        frame_embs = extractor.embed_crops(degraded_imgs)
        deg_emb = aggregate_embeddings(frame_embs, method="mean")
        clean_emb = clean_embedding_map[q.tracklet_id]
        cosine_sim = float(np.dot(deg_emb, clean_emb))
        differences.append(1.0 - cosine_sim)
        logger.info(
            "  Sanity [%s]: cosine(clean, degraded) = %.4f  (distance = %.4f)",
            q.tracklet_id,
            cosine_sim,
            1.0 - cosine_sim,
        )

    if not differences:
        raise RuntimeError(
            "Sanity check failed: could not extract any degraded crops. "
            "Check that video files exist under --root and camera IDs match."
        )
    max_diff = max(differences)
    if max_diff < 1e-4:
        raise RuntimeError(
            f"Sanity check FAILED: max cosine distance between clean and degraded "
            f"embeddings is {max_diff:.6f} — degradation has no effect on embeddings. "
            "This indicates the embedding cache is still being reused instead of "
            "re-extracting from degraded crops."
        )
    logger.info(
        "Sanity check PASSED: max embedding distance = %.4f (degradation is effective).",
        max_diff,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 controlled degraded query evaluation.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--weights", default="outputs/weights/veri_sbs_R50-ibn.pth")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--min-frames", type=int, default=2)
    # Frozen M4 hyperparameters — do not change without updating the frozen config
    parser.add_argument("--time-window-s", type=float, default=FROZEN_TIME_WINDOW_S,
                        help="Frozen M4 temporal window in seconds (default: 60.0)")
    parser.add_argument("--alpha", type=float, default=FROZEN_ALPHA,
                        help="Frozen M4 prior blend weight (default: 0.05)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/m4_degradation")
    parser.add_argument("--skip-sanity-check", action="store_true",
                        help="Skip the degradation sanity check (not recommended).")
    args = parser.parse_args()

    logger.info("Loading tracklets from %s ...", args.root)
    tracklets = load_cityflow_gt_tracklets(args.root, min_frames=args.min_frames)
    splits = make_identity_splits(tracklets, train_fraction=0.6, val_fraction=0.2, seed=args.seed)
    assert_identity_disjoint(splits)

    # Graph on train only
    train_tracklets = filter_tracklets_by_identities(tracklets, splits.train)
    camera_graph = build_camera_graph_from_training_tracklets(
        train_tracklets, min_edge_samples=FROZEN_MIN_EDGE_SAMPLES
    )

    # Evaluate on test tracklets
    test_tracklets = filter_tracklets_by_identities(tracklets, splits.test)
    logger.info("Degradation study on %d test tracklets.", len(test_tracklets))

    # Build video map: scoped_camera_id → video file path
    video_map = _build_video_map(args.root)
    logger.info("Video map has %d entries.", len(video_map))

    # Load clean gallery embeddings from cache
    from incident_search.eval.baseline import load_tracklet_embedding_map
    clean_embedding_map = load_tracklet_embedding_map(
        test_tracklets, cache_dir=args.embedding_cache, model_name=args.model_name
    )
    logger.info(
        "Loaded clean embeddings for %d / %d test tracklets.",
        len(clean_embedding_map),
        len(test_tracklets),
    )
    if not clean_embedding_map:
        raise FileNotFoundError(
            f"No clean embeddings found in '{args.embedding_cache}/{args.model_name}'. "
            "The gallery requires the existing clean embedding cache. "
            "If running in a fresh Colab instance, please restore or copy the 'outputs/cache/embeddings' "
            "folder from your previous session or Google Drive."
        )

    extractor = FastReIDExtractor(args.weights)

    # Sanity check: verify degradation actually changes embeddings
    if not args.skip_sanity_check:
        sanity_check_degradation_differs(
            test_tracklets, extractor, video_map, clean_embedding_map
        )

    # Precompute eligibility and prior matrix (same for all conditions)
    eligible_baseline, positives = build_upstream_protocol(
        test_tracklets, test_tracklets, time_window_s=None
    )
    eligible_window, _ = build_upstream_protocol(
        test_tracklets, test_tracklets, time_window_s=args.time_window_s
    )
    prior_matrix, _, _ = build_graph_prior_matrix(test_tracklets, test_tracklets, camera_graph)

    # Gallery matrix is always clean
    g_mat = np.stack([clean_embedding_map[t.tracklet_id] for t in test_tracklets])

    results = []

    for family, severities in DEGRADATION_SPECS:
        for sev in severities:
            seeds = [0, 1, 2] if family == "occlusion" else [args.seed]
            maps_app: list[float] = []
            maps_st: list[float] = []
            r1s_app: list[float] = []
            r1s_st: list[float] = []

            for s in seeds:
                # Freshly extract degraded query embeddings from video (no cache fallback)
                deg_query_embs = extract_degraded_query_embeddings(
                    test_tracklets, extractor, video_map, family, sev, seed=s
                )

                if not deg_query_embs:
                    logger.error(
                        "No degraded embeddings extracted for %s sev=%.3f seed=%d. "
                        "Check that video files are accessible.",
                        family, sev, s,
                    )
                    continue

                # Build query matrix: use degraded embedding where available,
                # skip tracklets with no video (they are excluded from the query set).
                # Only include test tracklets that have a degraded embedding.
                valid_indices = [
                    i for i, t in enumerate(test_tracklets)
                    if t.tracklet_id in deg_query_embs
                ]
                if not valid_indices:
                    continue

                q_mat = np.stack([deg_query_embs[test_tracklets[i].tracklet_id] for i in valid_indices])
                scores_app_full = (q_mat @ g_mat.T).astype(np.float32)

                # Subset the eligibility and positives masks to the valid query indices
                eligible_sub_base = eligible_baseline[np.array(valid_indices), :]
                eligible_sub_win = eligible_window[np.array(valid_indices), :]
                positives_sub = positives[np.array(valid_indices), :]

                # Subset prior matrix rows
                prior_sub = prior_matrix[np.array(valid_indices), :]
                scores_st_full = compute_fused_scores(scores_app_full, prior_sub, alpha=args.alpha)

                met_app, _ = evaluate_retrieval(
                    scores_app_full, positives_sub, eligible_mask=eligible_sub_base
                )
                met_st, _ = evaluate_retrieval(
                    scores_st_full, positives_sub, eligible_mask=eligible_sub_win
                )

                maps_app.append(met_app.mAP)
                maps_st.append(met_st.mAP)
                r1s_app.append(met_app.rank1)
                r1s_st.append(met_st.rank1)

            if not maps_app:
                continue

            entry = {
                "family": family,
                "severity": sev,
                "app_mAP_mean": round(float(np.mean(maps_app)), 6),
                "app_mAP_std": round(float(np.std(maps_app)), 6),
                "st_mAP_mean": round(float(np.mean(maps_st)), 6),
                "st_mAP_std": round(float(np.std(maps_st)), 6),
                "app_rank1_mean": round(float(np.mean(r1s_app)), 6),
                "st_rank1_mean": round(float(np.mean(r1s_st)), 6),
                "mAP_gain": round(float(np.mean(maps_st) - np.mean(maps_app)), 6),
                "num_queries": len(valid_indices),
            }
            results.append(entry)
            logger.info(
                "Degradation %s (sev=%.3f): App mAP=%.4f -> ST mAP=%.4f (gain=%+.4f)",
                family, sev, entry["app_mAP_mean"], entry["st_mAP_mean"], entry["mAP_gain"],
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "degradation_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== Degradation Study Summary ===")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
