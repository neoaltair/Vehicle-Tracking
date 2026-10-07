#!/usr/bin/env python3
"""Extract and cache FastReID embeddings for CityFlow GT-derived tracklets.

Run this BEFORE run_m3_appearance_baseline.py.  It discovers all GT tracklets
under the CityFlow root, extracts crops from the corresponding videos (caching
them under `--crops-dir`), then computes and caches per-tracklet embeddings
under `--embedding-cache`.

Usage (from repo root):
    python scripts/extract_m3_embeddings.py \\
        --root data/raw/cityflowv2 \\
        --crops-dir data/processed/cityflowv2/crops \\
        --weights outputs/weights/veri_sbs_R50-ibn.pth \\
        --embedding-cache outputs/cache/embeddings \\
        [--min-frames 2] [--max-tracklets N]

On Kaggle, replace the paths with the attached dataset mount points.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve package root so the script works from any cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from incident_search.io.cityflow import (  # noqa: E402
    discover_cityflow_cameras,
    gt_detections_to_tracklets,
    parse_cityflow_gt,
    save_tracklet_crops,
)
from incident_search.io.schema import Tracklet  # noqa: E402
from incident_search.reid.extractor import (  # noqa: E402
    FastReIDExtractor,
    aggregate_embeddings,
    load_embeddings,
    save_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _embed_one(
    tracklet: Tracklet,
    extractor: FastReIDExtractor,
    cache_dir: Path,
) -> bool:
    """Embed a single tracklet (if not already cached). Returns True if cached/done."""
    cached = load_embeddings(str(cache_dir), tracklet.tracklet_id)
    if cached is not None:
        return True  # already have this one

    valid_paths = [p for p in tracklet.crop_paths if Path(p).exists()]
    if not valid_paths:
        log.warning("Tracklet %s has no valid crop paths — skipping.", tracklet.tracklet_id)
        return False

    frame_embs = extractor.embed_crops(valid_paths)           # [N, 2048]
    trk_emb = aggregate_embeddings(frame_embs, method="mean")  # [2048]
    save_embeddings(
        cache_dir=str(cache_dir),
        tracklet_id=tracklet.tracklet_id,
        frame_embeddings=frame_embs,
        tracklet_embedding=trk_emb,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract FastReID embeddings for CityFlow GT tracklets."
    )
    parser.add_argument("--root", required=True, help="CityFlowV2 root directory.")
    parser.add_argument(
        "--crops-dir",
        default="data/processed/cityflowv2/crops",
        help="Directory for saved crop JPEG images.",
    )
    parser.add_argument(
        "--weights",
        default=str(_REPO_ROOT / "outputs" / "weights" / "veri_sbs_R50-ibn.pth"),
        help="Path to FastReID weights (.pth).",
    )
    parser.add_argument(
        "--embedding-cache",
        default="outputs/cache/embeddings",
        help="Directory for .npz embedding cache.",
    )
    parser.add_argument("--default-fps", type=float, default=10.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-crops-k", type=int, default=8, help="Crops per tracklet.")
    parser.add_argument(
        "--max-tracklets",
        type=int,
        default=None,
        help="Limit to first N tracklets (for smoke tests).",
    )
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    cache_dir = Path(args.embedding_cache)

    # 1. Discover cameras and parse GT files
    log.info("Discovering cameras under %s", args.root)
    cameras = discover_cityflow_cameras(args.root, default_fps=args.default_fps)
    log.info("Found %d camera folders with GT files.", sum(1 for c in cameras if c.gt_path))

    # 2. Build GT tracklets camera-by-camera so we have access to video_path
    all_tracklets: list[Tracklet] = []
    camera_video_map: dict[str, Path | None] = {}
    for camera in cameras:
        if camera.gt_path is None:
            continue
        detections = parse_cityflow_gt(
            camera.gt_path,
            scenario_id=camera.scenario_id,
            camera_id=camera.camera_id,
            fps=camera.fps,
            time_offset_s=camera.time_offset_s,
        )
        trks = gt_detections_to_tracklets(detections, min_frames=args.min_frames)
        for tr in trks:
            camera_video_map[tr.tracklet_id] = camera.video_path
        all_tracklets.extend(trks)

    if args.max_tracklets:
        all_tracklets = all_tracklets[: args.max_tracklets]
        log.info("Limiting to %d tracklets.", len(all_tracklets))

    log.info("Total GT tracklets to process: %d", len(all_tracklets))

    if len(all_tracklets) < 2:
        raise ValueError("Need at least 2 GT tracklets — check dataset root and GT files.")

    # 3. Load FastReID model
    log.info("Loading FastReID model from %s", args.weights)
    extractor = FastReIDExtractor(args.weights, device="cuda")

    # 4. Extract crops + embed
    n_cropped = 0
    n_embedded = 0
    n_skipped = 0

    for i, tracklet in enumerate(all_tracklets):
        # --- Crop extraction ---
        video_path = camera_video_map.get(tracklet.tracklet_id)
        if not tracklet.crop_paths or not any(Path(p).exists() for p in tracklet.crop_paths):
            if video_path is not None and video_path.exists():
                cam_crops_dir = crops_dir / tracklet.camera_id
                tracklet = save_tracklet_crops(tracklet, video_path, cam_crops_dir, k=args.max_crops_k)
                n_cropped += 1
            else:
                log.warning(
                    "[%d/%d] %s — no video available; cannot extract crops.",
                    i + 1, len(all_tracklets), tracklet.tracklet_id,
                )
                n_skipped += 1
                continue

        # --- Embedding ---
        ok = _embed_one(tracklet, extractor, cache_dir)
        if ok:
            n_embedded += 1
        else:
            n_skipped += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(all_tracklets):
            log.info(
                "Progress: %d / %d  (cropped %d, embedded %d, skipped %d)",
                i + 1, len(all_tracklets), n_cropped, n_embedded, n_skipped,
            )

    log.info("Done. Cropped: %d  Embedded: %d  Skipped: %d", n_cropped, n_embedded, n_skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
