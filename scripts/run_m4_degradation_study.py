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

Outputs:
  outputs/m4_degradation/degradation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from incident_search.degrade.transforms import degrade_crop
from incident_search.eval.metrics import evaluate_retrieval
from incident_search.eval.protocol import (
    assert_identity_disjoint,
    build_upstream_protocol,
    filter_tracklets_by_identities,
    make_identity_splits,
)
from incident_search.graph.camera_graph import build_camera_graph_from_training_tracklets
from incident_search.io.cityflow import load_cityflow_gt_tracklets
from incident_search.reid.extractor import FastReIDExtractor, aggregate_embeddings
from incident_search.search.spatiotemporal import (
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


def extract_degraded_query_embeddings(
    queries: list,
    extractor: FastReIDExtractor,
    family: str,
    severity: float,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Extract degraded embeddings for queries only. Gallery remains clean."""
    query_embs: dict[str, np.ndarray] = {}
    for q in queries:
        valid_crops = [p for p in q.crop_paths if Path(p).exists()]
        if not valid_crops:
            continue
        degraded_imgs = []
        for p in valid_crops:
            img = cv2.imread(p)
            if img is not None:
                deg = degrade_crop(img, family, severity, seed=seed)
                degraded_imgs.append(deg)

        if degraded_imgs:
            frame_embs = extractor.embed_crops(degraded_imgs)
            trk_emb = aggregate_embeddings(frame_embs, method="mean")
            query_embs[q.tracklet_id] = trk_emb

    return query_embs


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 controlled degraded query evaluation.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--weights", default="outputs/weights/veri_sbs_R50-ibn.pth")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--time-window-s", type=float, default=300.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/m4_degradation")
    args = parser.parse_args()

    tracklets = load_cityflow_gt_tracklets(args.root, min_frames=args.min_frames)
    splits = make_identity_splits(tracklets, train_fraction=0.6, val_fraction=0.2, seed=args.seed)
    assert_identity_disjoint(splits)

    # Graph on train only
    train_tracklets = filter_tracklets_by_identities(tracklets, splits.train)
    camera_graph = build_camera_graph_from_training_tracklets(train_tracklets)

    # Evaluate on test tracklets
    test_tracklets = filter_tracklets_by_identities(tracklets, splits.test)
    logger.info("Running degradation study on %d test tracklets...", len(test_tracklets))

    # Clean gallery embeddings from cache
    from incident_search.eval.baseline import load_tracklet_embedding_map
    clean_embedding_map = load_tracklet_embedding_map(test_tracklets, cache_dir=args.embedding_cache, model_name=args.model_name)

    extractor = FastReIDExtractor(args.weights)

    eligible_baseline, positives = build_upstream_protocol(test_tracklets, test_tracklets, time_window_s=None)
    eligible_window, _ = build_upstream_protocol(test_tracklets, test_tracklets, time_window_s=args.time_window_s)
    prior_matrix, _, _ = build_graph_prior_matrix(test_tracklets, test_tracklets, camera_graph)

    results = []

    for family, severities in DEGRADATION_SPECS:
        for sev in severities:
            seeds = [0, 1, 2] if family == "occlusion" else [0]
            maps_app = []
            maps_st = []
            r1s_app = []
            r1s_st = []

            for s in seeds:
                deg_query_embs = extract_degraded_query_embeddings(test_tracklets, extractor, family, sev, seed=s)
                # Query matrix with degraded embeddings, gallery with clean embeddings
                q_mat = np.stack([deg_query_embs.get(t.tracklet_id, clean_embedding_map[t.tracklet_id]) for t in test_tracklets])
                g_mat = np.stack([clean_embedding_map[t.tracklet_id] for t in test_tracklets])
                scores_app = (q_mat @ g_mat.T).astype(np.float32)
                scores_st = compute_fused_scores(scores_app, prior_matrix, alpha=args.alpha)

                met_app, _ = evaluate_retrieval(scores_app, positives, eligible_mask=eligible_baseline)
                met_st, _ = evaluate_retrieval(scores_st, positives, eligible_mask=eligible_window)

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
            }
            results.append(entry)
            logger.info("Degradation %s (sev=%.3f): App mAP=%.4f -> ST mAP=%.4f (gain=%+.4f)", family, sev, entry["app_mAP_mean"], entry["st_mAP_mean"], entry["mAP_gain"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "degradation_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\n=== Degradation Study Summary ===")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
