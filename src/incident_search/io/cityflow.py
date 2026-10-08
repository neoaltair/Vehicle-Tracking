"""CityFlowV2 dataset inspection and GT-tracklet conversion utilities.

The loader is intentionally conservative: it discovers the dataset structure
from a supplied root and parses conventional AI City / CityFlow MTMC ground
truth files without assuming the dataset is available during local development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from incident_search.io.schema import Tracklet


@dataclass(frozen=True)
class CityFlowDetection:
    """One ground-truth vehicle box from a CityFlow-style annotation file."""

    scenario_id: str
    camera_id: str
    frame_id: int
    gt_vehicle_id: str
    bbox_xyxy: tuple[float, float, float, float]
    time_s: float
    world_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class CityFlowCamera:
    """Discovered camera folder and available annotation/video files."""

    scenario_id: str
    camera_id: str
    root: Path
    video_path: Path | None
    gt_path: Path | None
    fps: float
    time_offset_s: float = 0.0


def _split_gt_line(line: str) -> list[str]:
    """Split a CityFlow/MOT-style annotation line on comma or whitespace."""
    stripped = line.strip()
    if "," in stripped:
        return [part.strip() for part in stripped.split(",")]
    return stripped.split()


def probe_video_fps(video_path: Path | None, default_fps: float = 10.0) -> float:
    """Return video FPS using OpenCV, falling back to `default_fps`."""
    if video_path is None or not video_path.exists():
        return default_fps

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return default_fps
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else default_fps
    finally:
        cap.release()


def discover_cityflow_cameras(
    root: str | Path,
    default_fps: float = 10.0,
) -> list[CityFlowCamera]:
    """Discover CityFlowV2 scenario/camera folders under `root`.

    A camera folder is any directory containing either a known video filename
    such as `vdo.avi` or a conventional `gt/gt.txt` annotation file. This keeps
    the scanner useful for partial Kaggle smoke-test inputs.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"CityFlow root does not exist: {root}")

    cameras: list[CityFlowCamera] = []
    seen: set[Path] = set()
    candidate_dirs: set[Path] = set()

    for gt_path in root.rglob("gt.txt"):
        if gt_path.parent.name.lower() == "gt":
            candidate_dirs.add(gt_path.parent.parent)

    for video_name in ("vdo.avi", "vdo.mp4", "vdo.mov", "video.avi", "video.mp4"):
        for video_path in root.rglob(video_name):
            candidate_dirs.add(video_path.parent)

    for camera_dir in sorted(candidate_dirs):
        if camera_dir in seen:
            continue
        seen.add(camera_dir)

        rel_parts = camera_dir.relative_to(root).parts
        scenario_id = rel_parts[-2] if len(rel_parts) >= 2 else "unknown"
        # Scope camera_id with scenario to prevent cross-scenario basename collisions
        # (e.g. S03/c010 and S05/c010 both have dirname "c010").
        camera_id = f"{scenario_id}_{camera_dir.name}"

        video_path = None
        for video_name in ("vdo.avi", "vdo.mp4", "vdo.mov", "video.avi", "video.mp4"):
            candidate = camera_dir / video_name
            if candidate.exists():
                video_path = candidate
                break

        gt_path = camera_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            gt_path = None

        cameras.append(
            CityFlowCamera(
                scenario_id=scenario_id,
                camera_id=camera_id,
                root=camera_dir,
                video_path=video_path,
                gt_path=gt_path,
                fps=probe_video_fps(video_path, default_fps=default_fps),
            )
        )

    return cameras


def parse_cityflow_gt(
    gt_path: str | Path,
    scenario_id: str,
    camera_id: str,
    fps: float = 10.0,
    time_offset_s: float = 0.0,
) -> list[CityFlowDetection]:
    """Parse a CityFlow/MOT-style `gt.txt` file.

    Expected columns are at least:
    `frame_id, object_id, x, y, width, height`.

    CityFlow challenge documentation uses 1-based frame IDs, so timestamps use
    `(frame_id - 1) / fps + time_offset_s`.
    """
    gt_path = Path(gt_path)
    detections: list[CityFlowDetection] = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            parts = _split_gt_line(line)
            if len(parts) < 6:
                raise ValueError(f"{gt_path}:{line_no}: expected at least 6 columns")

            frame_id = int(float(parts[0]))
            object_id = str(int(float(parts[1])))
            x = float(parts[2])
            y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            world_xy = None
            if len(parts) >= 8:
                try:
                    world_xy = (float(parts[6]), float(parts[7]))
                except ValueError:
                    world_xy = None

            detections.append(
                CityFlowDetection(
                    scenario_id=scenario_id,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    gt_vehicle_id=object_id,
                    bbox_xyxy=(x, y, x + width, y + height),
                    time_s=((frame_id - 1) / fps) + time_offset_s,
                    world_xy=world_xy,
                )
            )

    return detections


def gt_detections_to_tracklets(
    detections: list[CityFlowDetection],
    min_frames: int = 1,
) -> list[Tracklet]:
    """Convert GT detections into one tracklet per scenario/camera/identity."""
    grouped: dict[tuple[str, str, str], list[CityFlowDetection]] = {}
    for det in detections:
        key = (det.scenario_id, det.camera_id, det.gt_vehicle_id)
        grouped.setdefault(key, []).append(det)

    tracklets: list[Tracklet] = []
    for (scenario_id, camera_id, vehicle_id), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda det: det.frame_id)
        if len(rows) < min_frames:
            continue

        # camera_id is scoped (e.g. "S01_c001"); strip the scenario prefix for
        # the tracklet_id so the format stays "S01_c001_gtN" (not "S01_S01_c001_gtN").
        # We extract the original dirname by splitting off the scenario prefix.
        cam_basename = camera_id[len(scenario_id) + 1:] if camera_id.startswith(scenario_id + "_") else camera_id
        tracklet_id = f"{scenario_id}_{cam_basename}_gt{vehicle_id}"
        tracklets.append(
            Tracklet(
                tracklet_id=tracklet_id,
                camera_id=camera_id,
                start_time_s=round(rows[0].time_s, 3),
                end_time_s=round(rows[-1].time_s, 3),
                frames=[det.frame_id for det in rows],
                boxes=[
                    [round(coord, 2) for coord in det.bbox_xyxy]
                    for det in rows
                ],
                scores=[1.0 for _ in rows],
                vehicle_class="car",
                gt_vehicle_id=vehicle_id,
            )
        )

    return tracklets


def load_cityflow_gt_tracklets(
    root: str | Path,
    min_frames: int = 1,
    default_fps: float = 10.0,
) -> list[Tracklet]:
    """Discover all GT files under `root` and return GT-derived tracklets."""
    tracklets: list[Tracklet] = []
    for camera in discover_cityflow_cameras(root, default_fps=default_fps):
        if camera.gt_path is None:
            continue
        detections = parse_cityflow_gt(
            camera.gt_path,
            scenario_id=camera.scenario_id,
            camera_id=camera.camera_id,
            fps=camera.fps,
            time_offset_s=camera.time_offset_s,
        )
        tracklets.extend(gt_detections_to_tracklets(detections, min_frames=min_frames))
    return tracklets



def save_tracklet_crops(
    tracklet: Tracklet,
    video_path: str | Path,
    crops_dir: str | Path,
    k: int = 8,
) -> Tracklet:
    """Extract up to k quality-sampled GT crops using direct frame seeking."""
    import logging

    import cv2
    import numpy as np

    logger = logging.getLogger(__name__)

    video_path = Path(video_path)
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    n = len(tracklet.frames)
    boxes = tracklet.boxes
    scores = tracklet.scores

    # Select the k largest/highest-quality detections.
    qualities = []

    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        q = scores[i] * float(np.log1p(area))
        qualities.append((q, i))

    qualities.sort(key=lambda x: x[0], reverse=True)
    selected_indices = sorted([idx for _, idx in qualities[:k]])

    selected_frames = [tracklet.frames[i] for i in selected_indices]
    selected_boxes = [boxes[i] for i in selected_indices]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.warning("Could not open video %s", video_path)
        return tracklet

    saved_paths = []

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for frame_id, box in zip(selected_frames, selected_boxes):

            # CityFlow GT frame IDs are 1-based.
            # OpenCV CAP_PROP_POS_FRAMES is 0-based.
            zero_based = max(0, int(frame_id) - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)

            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Could not read frame %s from %s",
                    frame_id,
                    video_path,
                )
                continue

            x1, y1, x2, y2 = box

            x1 = max(0, min(width - 1, int(round(x1))))
            y1 = max(0, min(height - 1, int(round(y1))))
            x2 = max(x1 + 1, min(width, int(round(x2))))
            y2 = max(y1 + 1, min(height, int(round(y2))))

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            fname = f"{tracklet.tracklet_id}_f{int(frame_id):05d}.jpg"
            fpath = crops_dir / fname

            if not fpath.exists():
                cv2.imwrite(str(fpath), crop)

            saved_paths.append(str(fpath.as_posix()))

    finally:
        cap.release()

    return tracklet.model_copy(update={"crop_paths": saved_paths})
