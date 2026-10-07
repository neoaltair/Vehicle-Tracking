#!/usr/bin/env python3
"""Evaluate M4 Camera Graph and Travel-Time Spatio-Temporal Prior retrieval.

Builds camera graph using training identities only, estimates travel-time
distributions, and evaluates fused retrieval and candidate filtering
against the frozen M3 appearance baseline.

Outputs:
  outputs/m4_spatiotemporal/summary.json
  outputs/m4_spatiotemporal/comparison.json
  outputs/m4_spatiotemporal/per_query_results.csv
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


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 Spatio-temporal evaluation runner.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--default-fps", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--time-window-s", type=float, default=300.0)
    parser.add_argument("--alpha", type=float, default=0.1, help="Weight for prior in fused score")
    parser.add_argument("--min-edge-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prune-unconnected", action="store_true", help="Filter out candidates with 0 prior")
    parser.add_argument("--m3-summary", default="outputs/m3_appearance/summary.json", help="Path to frozen M3 summary")
    parser.add_argument("--out-dir", default="outputs/m4_spatiotemporal")
    args = parser.parse_args()

    # 1. Load GT tracklets
    logger.info("Loading GT tracklets from %s", args.root)
    tracklets = load_cityflow_gt_tracklets(
        args.root,
        min_frames=args.min_frames,
        default_fps=args.default_fps,
    )
    if args.max_tracklets:
        tracklets = tracklets[: args.max_tracklets]
    if len(tracklets) < 2:
        raise ValueError("Need at least two GT tracklets for evaluation")

    # 2. Strict identity-disjoint splits to PREVENT LEAKAGE
    splits = make_identity_splits(tracklets, train_fraction=0.6, val_fraction=0.2, seed=args.seed)
    assert_identity_disjoint(splits)
    logger.info(
        "Splits created: %d train IDs, %d val IDs, %d test IDs",
        len(splits.train), len(splits.val), len(splits.test),
    )

    # 3. Build Camera Transition Graph using TRAINING IDENTITIES ONLY
    train_tracklets = filter_tracklets_by_identities(tracklets, splits.train)
    logger.info("Building camera transition graph from %d training tracklets...", len(train_tracklets))
    camera_graph = build_camera_graph_from_training_tracklets(
        train_tracklets,
        min_edge_samples=args.min_edge_samples,
    )
    logger.info("Fitted %d directed transition edge models", len(camera_graph.edge_models))

    # 4. Load cached embeddings
    embedding_map = load_tracklet_embedding_map(
        tracklets,
        cache_dir=args.embedding_cache,
        model_name=args.model_name,
    )
    if len(embedding_map) < len(tracklets):
        missing = len(tracklets) - len(embedding_map)
        raise ValueError(f"Missing cached embeddings for {missing} / {len(tracklets)} tracklets.")

    # 5. Upstream cross-camera protocol (Method A baseline pool vs Method B window pool)
    eligible_baseline, positives = build_upstream_protocol(tracklets, tracklets, time_window_s=None)
    eligible_window, _ = build_upstream_protocol(tracklets, tracklets, time_window_s=args.time_window_s)

    # 6. Spatio-temporal travel-time priors
    logger.info("Computing travel-time priors...")
    prior_matrix, connected_mask = build_graph_prior_matrix(tracklets, tracklets, camera_graph)

    # Candidate mask for M4
    if args.prune_unconnected:
        # Pruning: must be within window AND have learned graph edge
        eligible_m4 = eligible_window & connected_mask
    else:
        # Window-based eligibility with soft prior re-ranking
        eligible_m4 = eligible_window

    # Compute search-space reduction & true-match survival
    reduction = pruning_reduction(eligible_baseline, eligible_m4)
    survival = true_match_survival(positives, eligible_m4)
    logger.info("Search-space reduction: %.2f%%", reduction * 100)
    logger.info("True-match survival: %.2f%%", survival * 100)

    # 7. Appearance cosine scores + Spatio-temporal fused scores
    appearance_scores = cosine_score_matrix(tracklets, tracklets, embedding_map)
    fused_scores = compute_fused_scores(appearance_scores, prior_matrix, alpha=args.alpha)

    # 8. Evaluate M4
    metrics, per_query = evaluate_retrieval(fused_scores, positives, eligible_mask=eligible_m4)

    # 9. Load frozen M3 baseline for comparison
    m3_data = {}
    m3_path = Path(args.m3_summary)
    if m3_path.exists():
        try:
            m3_data = json.loads(m3_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read M3 summary: %s", e)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_tracklets": len(tracklets),
        "num_queries": metrics.num_queries,
        "mAP": round(metrics.mAP, 6),
        "rank1": round(metrics.rank1, 6),
        "rank5": round(metrics.rank5, 6),
        "rank10": round(metrics.rank10, 6),
        "search_space_reduction": round(reduction, 6),
        "true_match_survival": round(survival, 6),
        "alpha": args.alpha,
        "time_window_s": args.time_window_s,
        "num_graph_edges": len(camera_graph.edge_models),
        "train_identities_used": len(splits.train),
        "model_name": args.model_name,
    }

    comparison = {
        "m3_appearance_baseline": {
            "mAP": m3_data.get("mAP", 0.266684),
            "rank1": m3_data.get("rank1", 0.402660),
            "rank5": m3_data.get("rank5", 0.549468),
            "rank10": m3_data.get("rank10", 0.623404),
        },
        "m4_spatiotemporal": {
            "mAP": summary["mAP"],
            "rank1": summary["rank1"],
            "rank5": summary["rank5"],
            "rank10": summary["rank10"],
            "search_space_reduction": summary["search_space_reduction"],
            "true_match_survival": summary["true_match_survival"],
        },
        "delta": {
            "mAP": round(summary["mAP"] - m3_data.get("mAP", 0.266684), 6),
            "rank1": round(summary["rank1"] - m3_data.get("rank1", 0.402660), 6),
            "rank5": round(summary["rank5"] - m3_data.get("rank5", 0.549468), 6),
            "rank10": round(summary["rank10"] - m3_data.get("rank10", 0.623404), 6),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    write_per_query_csv(per_query, out_dir / "per_query_results.csv")

    print("\n=== M4 Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Comparison with M3 Baseline ===")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
