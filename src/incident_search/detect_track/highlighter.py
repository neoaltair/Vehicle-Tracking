from pathlib import Path

import cv2
import imageio

from incident_search.detect_track.tracker import format_surveillance_clock
from incident_search.io.schema import Tracklet


def render_highlighted_video(
    video_path: str | Path,
    tracklets: list[Tracklet],
    output_video_path: str | Path,
    suspect_tracklet_ids: dict[str, dict],
    fps: float | None = None,
    time_offset_s: float = 0.0,
    max_frames: int | None = None,
) -> Path:
    """Render annotated surveillance video where suspect vehicle(s) are highlighted in bold red."""
    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    v_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames and max_frames > 0:
        total_to_process = min(total_frames, max_frames)
    else:
        total_to_process = total_frames

    # Build per-frame lookup: frame_idx -> list of observations
    frame_obs = {}
    for trk in tracklets:
        is_suspect = trk.tracklet_id in suspect_tracklet_ids
        sus_meta = suspect_tracklet_ids.get(trk.tracklet_id, {})
        for f_idx, box, score in zip(trk.frames, trk.boxes, trk.scores):
            frame_obs.setdefault(f_idx, []).append({
                "trk_id": trk.tracklet_id,
                "box": box,
                "score": score,
                "v_class": trk.vehicle_class,
                "is_suspect": is_suspect,
                "label": sus_meta.get("label", "SUSPECT MATCH"),
                "color": sus_meta.get("color", (0, 0, 255)),
            })

    track_trails = {}
    writer = imageio.get_writer(
        str(output_video_path),
        fps=v_fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    cam_id = tracklets[0].camera_id if tracklets else "SURV_CAM"

    try:
        for f_idx in range(total_to_process):
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()
            elapsed_time_s = time_offset_s + (f_idx / v_fps)
            clock_str = format_surveillance_clock(elapsed_time_s)

            obs_list = frame_obs.get(f_idx, [])
            suspect_present = any(obs["is_suspect"] for obs in obs_list)

            # Draw regular vehicles first, then draw suspect vehicles on top
            regular_obs = [o for o in obs_list if not o["is_suspect"]]
            suspect_obs = [o for o in obs_list if o["is_suspect"]]

            # 1. Regular vehicles (subtle / muted)
            for o in regular_obs:
                x1, y1, x2, y2 = [int(round(c)) for c in o["box"]]
                x1_c = max(0, min(width - 1, x1))
                y1_c = max(0, min(height - 1, y1))
                x2_c = max(x1_c + 1, min(width, x2))
                y2_c = max(y1_c + 1, min(height, y2))

                # Thin subtle box
                cv2.rectangle(annotated, (x1_c, y1_c), (x2_c, y2_c), (180, 180, 180), 1)
                lbl = f"{o['trk_id'].split('_')[-1]} {o['v_class']}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (x1_c, max(0, y1_c - th - 4)), (x1_c + tw + 6, y1_c), (30, 30, 30), -1)
                cv2.putText(annotated, lbl, (x1_c + 3, y1_c - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

            # 2. Suspect vehicles (BOLD NEON RED + PROMINENT BADGE)
            for o in suspect_obs:
                x1, y1, x2, y2 = [int(round(c)) for c in o["box"]]
                x1_c = max(0, min(width - 1, x1))
                y1_c = max(0, min(height - 1, y1))
                x2_c = max(x1_c + 1, min(width, x2))
                y2_c = max(y1_c + 1, min(height, y2))
                color = o["color"]

                center_pt = ((x1_c + x2_c) // 2, (y1_c + y2_c) // 2)
                track_trails.setdefault(o["trk_id"], []).append(center_pt)
                if len(track_trails[o["trk_id"]]) > 24:
                    track_trails[o["trk_id"]].pop(0)

                # Draw bold trajectory trail
                pts = track_trails[o["trk_id"]]
                for p_i in range(1, len(pts)):
                    cv2.line(annotated, pts[p_i - 1], pts[p_i], color, 3)

                # Thick outer box + inner highlight box (double border)
                cv2.rectangle(annotated, (x1_c - 2, y1_c - 2), (x2_c + 2, y2_c + 2), (0, 0, 0), 1)
                cv2.rectangle(annotated, (x1_c, y1_c), (x2_c, y2_c), color, 3)

                # Prominent Red Alert Badge
                badge_text = f"TARGET: {o['label']} [{o['trk_id']}]"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                badge_y1 = max(0, y1_c - th - 10)
                badge_y2 = y1_c
                cv2.rectangle(annotated, (x1_c, badge_y1), (x1_c + tw + 12, badge_y2), (0, 0, 180), -1)
                cv2.rectangle(annotated, (x1_c, badge_y1), (x1_c + tw + 12, badge_y2), (255, 255, 255), 1)
                cv2.putText(annotated, badge_text, (x1_c + 6, badge_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # CCTV Header Overlay
            banner_h = 36
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, banner_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            line_color = (0, 0, 255) if suspect_present else (59, 130, 246)
            cv2.line(annotated, (0, banner_h), (width, banner_h), line_color, 2)

            # Alert indicator
            ind_color = (0, 0, 255) if suspect_present else (0, 200, 0)
            cv2.circle(annotated, (20, 18), 6, ind_color, -1)

            status_lbl = "🚨 TARGET SUSPECT DETECTED" if suspect_present else "SURVEILLANCE NORMAL"
            osd_text = f"CAM: {cam_id.upper()}  |  {status_lbl}  |  TIME: {clock_str}  |  TRACKS: {len(obs_list):02d}"
            cv2.putText(annotated, osd_text, (35, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
        writer.close()

    return output_video_path
