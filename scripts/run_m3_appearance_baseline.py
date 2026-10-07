#!/usr/bin/env python3
"""Evaluate M3 appearance-only retrieval from cached CityFlow embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from incident_search.eval.baseline import (
    cosine_score_matrix,
    load_tracklet_embedding_map,
    write_per_query_csv,
)
from incident_search.eval.metrics import evaluate_retrieval
from incident_search.eval.protocol import build_upstream_protocol
from incident_search.io.cityflow import load_cityflow_gt_tracklets


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 appearance-only baseline evaluator.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--embedding-cache", default="outputs/cache/embeddings")
    parser.add_argument("--model-name", default="sbs_R50-ibn")
    parser.add_argument("--default-fps", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/m3_appearance")
    args = parser.parse_args()

    tracklets = load_cityflow_gt_tracklets(
        args.root,
        min_frames=args.min_frames,
        default_fps=args.default_fps,
    )
    if args.max_tracklets:
        tracklets = tracklets[: args.max_tracklets]
    if len(tracklets) < 2:
        raise ValueError("Need at least two GT tracklets for appearance baseline")

    embedding_map = load_tracklet_embedding_map(
        tracklets,
        cache_dir=args.embedding_cache,
        model_name=args.model_name,
    )
    if len(embedding_map) < len(tracklets):
        missing = len(tracklets) - len(embedding_map)
        raise ValueError(
            f"Missing cached embeddings for {missing} / {len(tracklets)} tracklets. "
            "Run embedding extraction before the appearance baseline."
        )

    eligible, positives = build_upstream_protocol(tracklets, tracklets, time_window_s=None)
    scores = cosine_score_matrix(tracklets, tracklets, embedding_map)
    metrics, per_query = evaluate_retrieval(scores, positives, eligible_mask=eligible)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_tracklets": len(tracklets),
        "num_queries": metrics.num_queries,
        "mAP": metrics.mAP,
        "rank1": metrics.rank1,
        "rank5": metrics.rank5,
        "rank10": metrics.rank10,
        "embedding_cache": str(args.embedding_cache),
        "model_name": args.model_name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_per_query_csv(per_query, out_dir / "per_query_results.csv")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
