#!/usr/bin/env python3
"""Inspect a supplied CityFlowV2 root without running heavy ML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from incident_search.io.cityflow import (
    discover_cityflow_cameras,
    load_cityflow_gt_tracklets,
    parse_cityflow_gt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect CityFlowV2 dataset structure.")
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument("--default-fps", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument("--out", default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    root = Path(args.root)
    cameras = discover_cityflow_cameras(root, default_fps=args.default_fps)
    selected = cameras[: args.max_cameras] if args.max_cameras else cameras

    camera_rows = []
    total_detections = 0
    for camera in selected:
        num_detections = 0
        num_ids = 0
        if camera.gt_path is not None:
            detections = parse_cityflow_gt(
                camera.gt_path,
                scenario_id=camera.scenario_id,
                camera_id=camera.camera_id,
                fps=camera.fps,
                time_offset_s=camera.time_offset_s,
            )
            num_detections = len(detections)
            num_ids = len({det.gt_vehicle_id for det in detections})
            total_detections += num_detections

        camera_rows.append(
            {
                "scenario_id": camera.scenario_id,
                "camera_id": camera.camera_id,
                "root": str(camera.root),
                "video_path": str(camera.video_path) if camera.video_path else None,
                "gt_path": str(camera.gt_path) if camera.gt_path else None,
                "fps": camera.fps,
                "num_detections": num_detections,
                "num_gt_ids": num_ids,
            }
        )

    tracklets = load_cityflow_gt_tracklets(
        root,
        min_frames=args.min_frames,
        default_fps=args.default_fps,
    )

    summary = {
        "root": str(root),
        "num_discovered_cameras": len(cameras),
        "num_inspected_cameras": len(selected),
        "num_gt_detections_in_inspected_cameras": total_detections,
        "num_gt_tracklets_all_cameras": len(tracklets),
        "num_gt_identities_all_cameras": len(
            {tracklet.gt_vehicle_id for tracklet in tracklets if tracklet.gt_vehicle_id}
        ),
        "cameras": camera_rows,
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
