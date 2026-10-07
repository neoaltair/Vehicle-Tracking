#!/usr/bin/env python3
"""Run a lightweight M3 protocol smoke test on CityFlow GT tracklets.

This does not extract embeddings. It verifies that a supplied CityFlow root can
produce GT-derived tracklets and upstream query/gallery masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from incident_search.eval.protocol import (
    build_upstream_protocol,
    make_identity_splits,
)
from incident_search.io.cityflow import load_cityflow_gt_tracklets


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 CityFlow protocol smoke test.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--default-fps", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-tracklets", type=int, default=500)
    parser.add_argument("--time-window-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/m3_smoke/summary.json")
    args = parser.parse_args()

    tracklets = load_cityflow_gt_tracklets(
        args.root,
        min_frames=args.min_frames,
        default_fps=args.default_fps,
    )
    if args.max_tracklets:
        tracklets = tracklets[: args.max_tracklets]

    if len(tracklets) < 2:
        raise ValueError("Need at least two GT tracklets for protocol smoke test")

    splits = make_identity_splits(tracklets, seed=args.seed)
    eligible_all, positives_all = build_upstream_protocol(tracklets, tracklets, time_window_s=None)
    eligible_window, positives_window = build_upstream_protocol(
        tracklets,
        tracklets,
        time_window_s=args.time_window_s,
    )

    summary = {
        "num_tracklets": len(tracklets),
        "num_identities": len({tracklet.gt_vehicle_id for tracklet in tracklets}),
        "split_num_train_ids": len(splits.train),
        "split_num_val_ids": len(splits.val),
        "split_num_test_ids": len(splits.test),
        "appearance_only_eligible_pairs": int(np.sum(eligible_all)),
        "appearance_only_positive_pairs": int(np.sum(positives_all)),
        "window_eligible_pairs": int(np.sum(eligible_window)),
        "window_positive_pairs": int(np.sum(positives_window)),
        "time_window_s": args.time_window_s,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
