#!/usr/bin/env python3
"""Run validation grid search to select optimal (time_window_s, alpha).

Uses VALIDATION IDENTITIES ONLY to prevent any test leakage.
Grid:
  time_window_s: [60.0, 120.0, 300.0, 600.0]
  alpha: [0.05, 0.10, 0.20]

Outputs:
  outputs/m4_validation/val_grid_results.json
  outputs/m4_validation/best_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from incident_search.eval.baseline import (
    cosine_score_matrix,
    load_tracklet_embedding_map,
)
from incident_search.eval.metrics import evaluate_retrieval
from incident_search.eval.protocol import (
    assert_identity_disjoint,
    build_upstream_protocol,
    filter_tracklets_by_identities,
    make_identity_splits,
    pruning_reduction,
    true_match_survival,
)
from incident_search.graph.camera_graph import build_camera_graph_from_training_tracklets
from incident_search.io.cityflow import load_cityflow_gt_tracklets
from incident_search.search.spatiotemporal import (
    build_graph_prior_matrix,
    compute_fused_scores,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TIME_WINDOWS = [60.0, 120.0, 300.0, 600.0]
ALPHAS = [0.05, 0.10, 0.20]


def run_val_grid_search(
    tracklets: list,
    embedding_map: dict,
    camera_graph,
    splits,
    time_windows: list[float] | None = None,
    alphas: list[float] | None = None,
) -> tuple[dict, list[dict]]:
    time_windows = time_windows or TIME_WINDOWS
    alphas = alphas or ALPHAS

    # Filter to validation tracklets ONLY
    val_tracklets = filter_tracklets_by_identities(tracklets, splits.val)
    if len(val_tracklets) < 2:
        raise ValueError("Need at least 2 validation tracklets for validation selection")

    # Baseline pool for validation queries
    val_baseline_eligible, val_positives = build_upstream_protocol(val_tracklets, val_tracklets, time_window_s=None)
    val_app_scores = cosine_score_matrix(val_tracklets, val_tracklets, embedding_map)

    # Precompute prior matrix on validation set
    prior_matrix, connected_mask, topology_mask = build_graph_prior_matrix(val_tracklets, val_tracklets, camera_graph)

    results: list[dict] = []
    best_config: dict | None = None
    best_map = -1.0

    for tw in time_windows:
        val_tw_eligible, _ = build_upstream_protocol(val_tracklets, val_tracklets, time_window_s=tw)
        reduction = pruning_reduction(val_baseline_eligible, val_tw_eligible)
        survival = true_match_survival(val_positives, val_tw_eligible)

        for a in alphas:
            fused = compute_fused_scores(val_app_scores, prior_matrix, alpha=a)
            metrics, _ = evaluate_retrieval(fused, val_positives, eligible_mask=val_tw_eligible)

            entry = {
                "time_window_s": tw,
                "alpha": a,
                "val_mAP": round(metrics.mAP, 6),
                "val_rank1": round(metrics.rank1, 6),
                "val_rank5": round(metrics.rank5, 6),
                "val_rank10": round(metrics.rank10, 6),
                "val_reduction": round(reduction, 6),
                "val_survival": round(survival, 6),
                "num_queries": metrics.num_queries,
            }
            results.append(entry)

            if metrics.mAP > best_map:
                best_map = metrics.mAP
                best_config = entry

    return best_config, results


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 validation hyperparameter selection.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--min-edge-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/m4_validation")
    args = parser.parse_args()

    tracklets = load_cityflow_gt_tracklets(args.root, min_frames=args.min_frames)
    splits = make_identity_splits(tracklets, train_fraction=0.6, val_fraction=0.2, seed=args.seed)
    assert_identity_disjoint(splits)

    # Graph strictly on train
    train_tracklets = filter_tracklets_by_identities(tracklets, splits.train)
    camera_graph = build_camera_graph_from_training_tracklets(train_tracklets, min_edge_samples=args.min_edge_samples)

    embedding_map = load_tracklet_embedding_map(tracklets, cache_dir=args.embedding_cache, model_name=args.model_name)

    best_config, all_results = run_val_grid_search(tracklets, embedding_map, camera_graph, splits)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_grid_results.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "best_config.json").write_text(json.dumps(best_config, indent=2) + "\n", encoding="utf-8")

    logger.info("Validation search complete. Best config on validation:")
    print(json.dumps(best_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
