#!/usr/bin/env python3
"""CLI entry point for vehicle detection and single-camera tracking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from incident_search.detect_track import run_detect_track
from incident_search.io.schema import save_tracklets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_detect_track")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run vehicle detection and tracking using YOLO11m + ByteTrack"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--camera-id", type=str, default="c001", help="Camera identifier (default: c001)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/detect_track",
        help="Output directory to save tracklets and outputs",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (optional)",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Camera time offset in seconds relative to global clock",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable saving annotated video",
    )

    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting detection and tracking on %s (camera %s)", args.video, args.camera_id)
    tracklets = run_detect_track(
        video_path=args.video,
        camera_id=args.camera_id,
        config=config,
        output_dir=out_dir,
        max_frames=args.max_frames,
        save_annotated_video=not args.no_video,
        time_offset_s=args.time_offset,
    )

    tracklets_file = out_dir / f"{args.camera_id}_tracklets.json"
    save_tracklets(tracklets, tracklets_file)
    logger.info("Successfully saved %d tracklets to %s", len(tracklets), tracklets_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
