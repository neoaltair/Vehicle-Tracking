"""Streaming vehicle detection and tracking yielding annotated frames in real-time."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from incident_search.detect_track.tracker import (
    CLASS_NAMES,
    VEHICLE_CLASS_IDS,
    filter_tracklets_by_length,
    sample_best_crops,
)
from incident_search.io.schema import Tracklet

logger = logging.getLogger(__name__)


def stream_detect_track(
    video_path: str | Path,
    camera_id: str = "c001",
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "outputs/detect_track",
    max_frames: int | None = None,
    time_offset_s: float = 0.0,
    min_len: int = 5,
    max_crops_k: int = 8,
) -> Generator[tuple[np.ndarray, int, int], None, list[Tracklet]]:
    """Yield annotated RGB frames live while accumulating Tracklet data.

    Yields:
        (annotated_frame_rgb, frame_idx, active_vehicle_count)
    Returns:
        List of finalized Tracklet objects.
    """
    config = config or {}
    model_name = config.get("model", {}).get("name", "yolo11m.pt")
    conf_thresh = config.get("detector", {}).get("conf_threshold", 0.35)
    classes = config.get("detector", {}).get("classes", VEHICLE_CLASS_IDS)
    tracker_type = config.get("tracker", {}).get("type", "bytetrack.yaml")
    device = config.get("model", {}).get("device", None)

    video_path = Path(video_path)
    out_path = Path(output_dir) if output_dir else None
    crops_dir = out_path / "crops" / camera_id if out_path else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(model_name)
    raw_tracklets: dict[str, dict[str, Any]] = {}
    frame_idx = 0

    results = model.track(
        source=str(video_path),
        tracker=tracker_type,
        classes=classes,
        conf=conf_thresh,
        device=device,
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results:
        if max_frames and frame_idx >= max_frames:
            break

        orig_frame = result.orig_img.copy()
        annotated_frame = orig_frame.copy()

        boxes_obj = result.boxes
        active_count = 0

        if boxes_obj is not None and len(boxes_obj) > 0 and boxes_obj.id is not None:
            track_ids = boxes_obj.id.int().cpu().tolist()
            coords = boxes_obj.xyxy.cpu().tolist()
            confs = boxes_obj.conf.cpu().tolist()
            cls_ids = boxes_obj.cls.int().cpu().tolist()
            active_count = len(track_ids)

            for trk_id_int, (x1, y1, x2, y2), conf, cls_id in zip(
                track_ids, coords, confs, cls_ids
            ):
                trk_id_str = f"{camera_id}_trk{trk_id_int:04d}"
                v_class = CLASS_NAMES.get(cls_id, "car")

                x1_c = max(0, min(width - 1, int(round(x1))))
                y1_c = max(0, min(height - 1, int(round(y1))))
                x2_c = max(x1_c + 1, min(width, int(round(x2))))
                y2_c = max(y1_c + 1, min(height, int(round(y2))))

                crop_img = orig_frame[y1_c:y2_c, x1_c:x2_c].copy()

                if trk_id_str not in raw_tracklets:
                    raw_tracklets[trk_id_str] = {
                        "frames": [],
                        "boxes": [],
                        "scores": [],
                        "crop_images": [],
                        "vehicle_class": v_class,
                    }

                raw_tracklets[trk_id_str]["frames"].append(frame_idx)
                raw_tracklets[trk_id_str]["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
                raw_tracklets[trk_id_str]["scores"].append(float(conf))
                raw_tracklets[trk_id_str]["crop_images"].append(crop_img)

                # Draw bounding box and identification badge
                cv2.rectangle(
                    annotated_frame,
                    (x1_c, y1_c),
                    (x2_c, y2_c),
                    (0, 255, 64),
                    2,
                )
                label = f"Target ID #{trk_id_int} [{v_class}] {conf:.2f}"
                cv2.rectangle(
                    annotated_frame,
                    (x1_c, max(0, y1_c - 22)),
                    (x1_c + len(label) * 9, y1_c),
                    (0, 255, 64),
                    cv2.FILLED,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1_c + 2, max(15, y1_c - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Convert to RGB for direct Streamlit display
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        yield (rgb_frame, frame_idx, active_count)
        frame_idx += 1

    # Finalize tracklets
    filtered = filter_tracklets_by_length(raw_tracklets, min_len=min_len)
    final_tracklets: list[Tracklet] = []

    for trk_id_str, data in filtered.items():
        frames = data["frames"]
        boxes = data["boxes"]
        scores = data["scores"]
        crop_imgs = data["crop_images"]

        start_time_s = (frames[0] / fps) + time_offset_s
        end_time_s = (frames[-1] / fps) + time_offset_s

        saved_crop_paths: list[str] = []
        best_crops = sample_best_crops(frames, boxes, scores, crop_imgs, k=max_crops_k)

        if crops_dir:
            for s_frame, _, s_img in best_crops:
                crop_filename = f"{trk_id_str}_f{s_frame:05d}.jpg"
                crop_full_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_full_path), s_img)
                saved_crop_paths.append(str(crop_full_path.as_posix()))

        tracklet = Tracklet(
            tracklet_id=trk_id_str,
            camera_id=camera_id,
            start_time_s=round(start_time_s, 3),
            end_time_s=round(end_time_s, 3),
            frames=frames,
            boxes=[[round(coord, 2) for coord in b] for b in boxes],
            scores=[round(s, 4) for s in scores],
            vehicle_class=data["vehicle_class"],
            crop_paths=saved_crop_paths,
        )
        final_tracklets.append(tracklet)

    return final_tracklets
