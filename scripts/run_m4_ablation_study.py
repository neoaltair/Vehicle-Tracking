#!/usr/bin/env python3
"""Run full M4 ablations and final test evaluation with frozen validation-selected hyperparameters.

Methods evaluated:
1. Appearance only (Method A)
2. Appearance + fixed time window (Method B)
3. Appearance + camera topology (Method B + topology filtering)
4. Appearance + topology + travel-time prior (Method C, full spatio-temporal)

Also optionally runs the controlled degraded query evaluation.

Outputs:
  outputs/m4_ablation/ablation_summary.json
  outputs/m4_ablation/test_final_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from incident_search.eval.baseline import (
    cosine_score_matrix,
    load_tracklet_embedding_map,
    write_per_query_csv,
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


def evaluate_method(
    name: str,
    scores,
    positives,
    eligible_baseline,
    eligible_active,
) -> dict:
    """Evaluate a single retrieval method and return metric dictionary."""
    metrics, per_query = evaluate_retrieval(scores, positives, eligible_mask=eligible_active)
    reduction = pruning_reduction(eligible_baseline, eligible_active)
    survival = true_match_survival(positives, eligible_active)

    return {
        "method": name,
        "mAP": round(metrics.mAP, 6),
        "rank1": round(metrics.rank1, 6),
        "rank5": round(metrics.rank5, 6),
        "rank10": round(metrics.rank10, 6),
        "candidate_reduction": round(reduction, 6),
        "true_match_survival": round(survival, 6),
        "num_queries": metrics.num_queries,
    }, per_query


def run_ablations_on_split(
    tracklets: list,
    embedding_map: dict,
    camera_graph,
    time_window_s: float,
    alpha: float,
) -> dict[str, dict]:
    """Run all 4 ablations on the supplied tracklet set."""
    eligible_baseline, positives = build_upstream_protocol(tracklets, tracklets, time_window_s=None)
    eligible_window, _ = build_upstream_protocol(tracklets, tracklets, time_window_s=time_window_s)

    prior_matrix, connected_mask, topology_mask = build_graph_prior_matrix(tracklets, tracklets, camera_graph)
    appearance_scores = cosine_score_matrix(tracklets, tracklets, embedding_map)
    fused_scores = compute_fused_scores(appearance_scores, prior_matrix, alpha=alpha)

    # 1. Appearance only (no window, no topology, no prior)
    res_app, _ = evaluate_method("appearance_only", appearance_scores, positives, eligible_baseline, eligible_baseline)

    # 2. Appearance + fixed time window
    res_win, _ = evaluate_method("appearance_plus_window", appearance_scores, positives, eligible_baseline, eligible_window)

    # 3. Appearance + camera topology (window + topology connected)
    eligible_topology = eligible_window & topology_mask
    res_top, _ = evaluate_method("appearance_plus_topology", appearance_scores, positives, eligible_baseline, eligible_topology)

    # 4. Appearance + topology + travel-time prior (fused scores on window)
    res_prior, per_query_final = evaluate_method("appearance_topology_travel_time_prior", fused_scores, positives, eligible_baseline, eligible_window)

    return {
        "appearance_only": res_app,
        "appearance_plus_window": res_win,
        "appearance_plus_topology": res_top,
        "appearance_topology_travel_time_prior": res_prior,
    }, per_query_final


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 ablation study and final test evaluation.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--min-edge-samples", type=int, default=5)
    parser.add_argument("--best-config", default="outputs/m4_validation/best_config.json")
    parser.add_argument("--default-time-window-s", type=float, default=300.0)
    parser.add_argument("--default-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/m4_ablation")
    args = parser.parse_args()

    tracklets = load_cityflow_gt_tracklets(args.root, min_frames=args.min_frames)
    splits = make_identity_splits(tracklets, train_fraction=0.6, val_fraction=0.2, seed=args.seed)
    assert_identity_disjoint(splits)

    # Build graph strictly on train
    train_tracklets = filter_tracklets_by_identities(tracklets, splits.train)
    camera_graph = build_camera_graph_from_training_tracklets(train_tracklets, min_edge_samples=args.min_edge_samples)

    embedding_map = load_tracklet_embedding_map(tracklets, cache_dir=args.embedding_cache, model_name=args.model_name)

    # Load frozen parameters selected on validation
    time_window_s = args.default_time_window_s
    alpha = args.default_alpha
    best_cfg_path = Path(args.best_config)
    if best_cfg_path.exists():
        cfg = json.loads(best_cfg_path.read_text(encoding="utf-8"))
        time_window_s = cfg.get("time_window_s", time_window_s)
        alpha = cfg.get("alpha", alpha)
        logger.info("Loaded validation-selected parameters: window=%.1fs, alpha=%.2f", time_window_s, alpha)
    else:
        logger.info("Using default parameters: window=%.1fs, alpha=%.2f", time_window_s, alpha)

    # Run on TEST SPLIT ONLY for final evaluation
    test_tracklets = filter_tracklets_by_identities(tracklets, splits.test)
    logger.info("Evaluating on TEST SPLIT ONLY (%d test tracklets)...", len(test_tracklets))

    ablations_test, per_query_final = run_ablations_on_split(
        test_tracklets,
        embedding_map,
        camera_graph,
        time_window_s=time_window_s,
        alpha=alpha,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_summary.json").write_text(json.dumps(ablations_test, indent=2) + "\n", encoding="utf-8")
    (out_dir / "test_final_results.json").write_text(json.dumps(ablations_test["appearance_topology_travel_time_prior"], indent=2) + "\n", encoding="utf-8")
    write_per_query_csv(per_query_final, out_dir / "final_test_per_query.csv")

    print("\n=== M4 Final Test Ablation Results ===")
    print(json.dumps(ablations_test, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
