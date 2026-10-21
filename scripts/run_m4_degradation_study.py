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

Optimized version:
  Groups frame extraction video-by-video so OpenCV opens each video file exactly ONCE
  and extracts crops in a single sequential sweep, avoiding FUSE network seek deadlocks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure src/ is on sys.path
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
from incident_search.reid.extractor import (  # noqa: E402
    FastReIDExtractor,
    aggregate_embeddings,
    save_embeddings,
)
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

FROZEN_TIME_WINDOW_S = 60.0
FROZEN_ALPHA = 0.05
FROZEN_MIN_EDGE_SAMPLES = 5


def _build_video_map(root: str | Path) -> dict[str, Path]:
    cameras = discover_cityflow_cameras(root)
    result: dict[str, Path] = {}
    for cam in cameras:
        if cam.video_path is not None:
            result[cam.camera_id] = cam.video_path
    return result


def extract_crops_batch(
    tracklets: list,
    video_map: dict[str, Path],
    k: int = 4,
) -> dict[str, list[np.ndarray]]:
    """Extract raw vehicle crops by grouping requests per video file.

    Opens each video file exactly ONCE and reads frames sequentially or with
    minimal forward seeks, preventing OpenCV network FUSE hang.
    """
    by_video: dict[Path, list[tuple[int, list[float], str]]] = defaultdict(list)
    tracklet_crops: dict[str, list[np.ndarray]] = defaultdict(list)

    for trk in tracklets:
        vpath = video_map.get(trk.camera_id)
        if not vpath or not vpath.exists():
            continue

        n = len(trk.frames)
        boxes = trk.boxes
        scores = trk.scores

        qualities = []
        for i in range(n):
            x1, y1, x2, y2 = boxes[i]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            q = scores[i] * float(np.log1p(area))
            qualities.append((q, i))

        qualities.sort(key=lambda x: x[0], reverse=True)
        selected = sorted([idx for _, idx in qualities[:k]])

        for idx in selected:
            frame_id = trk.frames[idx]
            box = boxes[idx]
            zero_based = max(0, int(frame_id) - 1)
            by_video[vpath].append((zero_based, box, trk.tracklet_id))

    logger.info("Extracting crops across %d unique video files...", len(by_video))

    for vidx, (vpath, reqs) in enumerate(by_video.items()):
        # Sort requests by frame index for forward-only decoding
        reqs.sort(key=lambda x: x[0])
        reqs_by_frame = defaultdict(list)
        for f_idx, box, tid in reqs:
            reqs_by_frame[f_idx].append((box, tid))

        target_frames = sorted(reqs_by_frame.keys())
        if not target_frames:
            continue

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        curr_frame = 0

        try:
            for t_frame in target_frames:
                # Seek forward only if needed
                if t_frame > curr_frame + 5:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, t_frame)
                    curr_frame = t_frame

                while curr_frame < t_frame:
                    ret = cap.grab()
                    if not ret:
                        break
                    curr_frame += 1

                ret, frame = cap.read()
                curr_frame += 1
                if not ret or frame is None:
                    continue

                for box, tid in reqs_by_frame[t_frame]:
                    x1, y1, x2, y2 = box
                    x1 = max(0, min(width - 1, int(round(x1))))
                    y1 = max(0, min(height - 1, int(round(y1))))
                    x2 = max(x1 + 1, min(width, int(round(x2))))
                    y2 = max(y1 + 1, min(height, int(round(y2))))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        tracklet_crops[tid].append(crop)
        finally:
            cap.release()

        if (vidx + 1) % 5 == 0 or (vidx + 1) == len(by_video):
            logger.info("Processed %d / %d video streams.", vidx + 1, len(by_video))

    return tracklet_crops


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 controlled degraded query evaluation.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--weights", default="outputs/weights/veri_sbs_R50-ibn.pth")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--time-window-s", type=float, default=FROZEN_TIME_WINDOW_S)
    parser.add_argument("--alpha", type=float, default=FROZEN_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/m4_degradation")
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

    test_tracklets = filter_tracklets_by_identities(tracklets, splits.test)
    logger.info("Degradation study on %d test tracklets.", len(test_tracklets))

    video_map = _build_video_map(args.root)
    extractor = FastReIDExtractor(args.weights)

    # Fast batch extraction: opens each video once
    logger.info("Extracting raw test crops in batched video sweep...")
    raw_crops_cache = extract_crops_batch(test_tracklets, video_map, k=4)
    logger.info("Successfully extracted raw crops for %d test vehicles.", len(raw_crops_cache))

    if not raw_crops_cache:
        raise RuntimeError("No crops could be extracted from video files. Check dataset root path.")

    # Compute clean embeddings directly in RAM
    clean_embedding_map: dict[str, np.ndarray] = {}
    for tid, crops in raw_crops_cache.items():
        f_embs = extractor.embed_crops(crops)
        clean_embedding_map[tid] = aggregate_embeddings(f_embs, method="mean")
        save_embeddings(
            cache_dir=args.embedding_cache,
            tracklet_id=tid,
            frame_embeddings=f_embs,
            tracklet_embedding=clean_embedding_map[tid],
            model_name=args.model_name,
        )

    # Restrict to test tracklets that have valid crops
    valid_test_tracklets = [t for t in test_tracklets if t.tracklet_id in clean_embedding_map]
    logger.info("Proceeding with %d validated test tracklets.", len(valid_test_tracklets))

    # Precompute eligibility and prior matrix
    eligible_baseline, positives = build_upstream_protocol(
        valid_test_tracklets, valid_test_tracklets, time_window_s=None
    )
    eligible_window, _ = build_upstream_protocol(
        valid_test_tracklets, valid_test_tracklets, time_window_s=args.time_window_s
    )
    prior_matrix, _, _ = build_graph_prior_matrix(
        valid_test_tracklets, valid_test_tracklets, camera_graph
    )

    g_mat = np.stack([clean_embedding_map[t.tracklet_id] for t in valid_test_tracklets])
    results = []

    # Fast evaluation loop over in-memory crops
    for family, severities in DEGRADATION_SPECS:
        for sev in severities:
            seeds = [0, 1, 2] if family == "occlusion" else [args.seed]
            maps_app: list[float] = []
            maps_st: list[float] = []
            r1s_app: list[float] = []
            r1s_st: list[float] = []

            for s in seeds:
                deg_query_embs = {}
                for tid, crops in raw_crops_cache.items():
                    deg_imgs = [degrade_crop(img, family, sev, seed=s) for img in crops]
                    f_embs = extractor.embed_crops(deg_imgs)
                    deg_query_embs[tid] = aggregate_embeddings(f_embs, method="mean")

                q_mat = np.stack([deg_query_embs[t.tracklet_id] for t in valid_test_tracklets])
                scores_app = (q_mat @ g_mat.T).astype(np.float32)
                scores_st = compute_fused_scores(scores_app, prior_matrix, alpha=args.alpha)

                met_app, _ = evaluate_retrieval(
                    scores_app, positives, eligible_mask=eligible_baseline
                )
                met_st, _ = evaluate_retrieval(
                    scores_st, positives, eligible_mask=eligible_window
                )

                maps_app.append(met_app.mAP)
                maps_st.append(met_st.mAP)
                r1s_app.append(met_app.rank1)
                r1s_st.append(met_st.rank1)

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
                "num_queries": len(valid_test_tracklets),
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
