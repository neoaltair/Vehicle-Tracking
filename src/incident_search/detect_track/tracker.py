"""Vehicle detection and single-camera multi-object tracking.

Uses Ultralytics YOLO11 medium (yolo11m.pt) with ByteTrack to generate
standardized Tracklet objects with annotated video output and sampled vehicle crops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from incident_search.io.schema import Tracklet

logger = logging.getLogger(__name__)

# COCO vehicle class mappings
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def filter_tracklets_by_length(
    tracklet_candidates: dict[str, dict[str, Any]],
    min_len: int = 10,
) -> dict[str, dict[str, Any]]:
    """Filter out tracklets with fewer than min_len observed frames.

    Args:
        tracklet_candidates: Dictionary of raw tracklet observations keyed by track ID.
        min_len: Minimum number of frames required to retain tracklet.

    Returns:
        Filtered dictionary containing only valid tracklets.
    """
    return {
        track_id: data
        for track_id, data in tracklet_candidates.items()
        if len(data["frames"]) >= min_len
    }


def sample_best_crops(
    frames: list[int],
    boxes: list[list[float]],
    scores: list[float],
    crop_images: list[np.ndarray],
    k: int = 8,
) -> list[tuple[int, list[float], np.ndarray]]:
    """Sample up to k best crops for a tracklet.

    Prioritizes a mix of high confidence and large bounding box area.

    Args:
        frames: Frame index list.
        boxes: Bounding boxes [[x1, y1, x2, y2]].
        scores: Confidence scores.
        crop_images: Cropped image arrays.
        k: Maximum number of crops to sample.

    Returns:
        List of selected (frame_idx, box, crop_image) tuples.
    """
    n = len(frames)
    if n <= k:
        return list(zip(frames, boxes, crop_images))

    # Score each observation by confidence * log(area + 1)
    qualities = []
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        quality = scores[i] * float(np.log1p(area))
        qualities.append((quality, i))

    # Sort descending by quality score
    qualities.sort(key=lambda x: x[0], reverse=True)
    selected_indices = sorted([idx for _, idx in qualities[:k]])

    return [(frames[i], boxes[i], crop_images[i]) for i in selected_indices]


def run_detect_track(
    video_path: str | Path,
    camera_id: str,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    max_frames: int | None = None,
    save_annotated_video: bool = True,
    time_offset_s: float = 0.0,
) -> list[Tracklet]:
    """Run detection and single-camera tracking on video footage.

    Args:
        video_path: Path to input video file.
        camera_id: Identifier for camera.
        config: Configuration dictionary for detector and tracker parameters.
        output_dir: Base directory to save crops, json, and rendered video.
        max_frames: Optional frame limit for fast debugging/runs.
        save_annotated_video: Whether to write an annotated MP4 video to output_dir.
        time_offset_s: Time offset to common global clock in seconds.

    Returns:
        List of filtered Tracklet objects.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    config = config or {}
    det_cfg = config.get("detector", {})
    trk_cfg = config.get("tracker", {})

    model_name = det_cfg.get("model_name", "yolo11m.pt")
    conf_thresh = det_cfg.get("conf_threshold", 0.25)
    classes = det_cfg.get("classes", VEHICLE_CLASS_IDS)
    device = det_cfg.get("device", "cpu")

    tracker_type = trk_cfg.get("tracker_yaml", "bytetrack.yaml")
    min_len = trk_cfg.get("min_len", 10)
    max_crops_k = trk_cfg.get("max_crops_per_tracklet", 8)

    if output_dir:
        out_path = Path(output_dir)
        crops_dir = out_path / "crops" / camera_id
        crops_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None
        crops_dir = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file via OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames and max_frames > 0:
        total_to_process = min(total_frames, max_frames)
    else:
        total_to_process = total_frames

    logger.info(
        "Processing video %s: %dx%d @ %.1f FPS, %d frames (processing %d)",
        video_path.name,
        width,
        height,
        fps,
        total_frames,
        total_to_process,
    )

    # Initialize video writer for annotated video
    video_writer = None
    if save_annotated_video and out_path:
        annotated_video_path = out_path / f"{camera_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(annotated_video_path), fourcc, fps, (width, height)
        )

    # Load YOLO detector
    model = YOLO(model_name)

    # Dictionary storing raw observations per track ID
    raw_tracklets: dict[str, dict[str, Any]] = {}

    frame_idx = 0
    try:
        # Run tracking stream
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
            if boxes_obj is not None and len(boxes_obj) > 0 and boxes_obj.id is not None:
                track_ids = boxes_obj.id.int().cpu().tolist()
                coords = boxes_obj.xyxy.cpu().tolist()
                confs = boxes_obj.conf.cpu().tolist()
                cls_ids = boxes_obj.cls.int().cpu().tolist()

                for trk_id_int, (x1, y1, x2, y2), conf, cls_id in zip(
                    track_ids, coords, confs, cls_ids
                ):
                    trk_id_str = f"{camera_id}_trk{trk_id_int:04d}"
                    v_class = CLASS_NAMES.get(cls_id, "car")

                    # Clamp box coordinates
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

                    # Draw box and track ID on annotated frame
                    cv2.rectangle(
                        annotated_frame,
                        (x1_c, y1_c),
                        (x2_c, y2_c),
                        (0, 255, 0),
                        2,
                    )
                    label = f"ID:{trk_id_int} {v_class} {conf:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1_c, max(15, y1_c - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            if video_writer:
                video_writer.write(annotated_frame)

            frame_idx += 1

    finally:
        cap.release()
        if video_writer:
            video_writer.release()

    # Filter short tracklets
    filtered = filter_tracklets_by_length(raw_tracklets, min_len=min_len)
    logger.info(
        "Detected %d raw tracklets, %d passed min_len=%d filter",
        len(raw_tracklets),
        len(filtered),
        min_len,
    )

    # Build final Tracklet objects and save crops
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
                # Store path relative to workspace or absolute string
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
