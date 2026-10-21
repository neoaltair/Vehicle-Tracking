"""Rule-based kinematic incident detector for vehicle surveillance videos."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from incident_search.io.schema import Incident, Tracklet

logger = logging.getLogger(__name__)


def _box_center(box: list[float]) -> tuple[float, float]:
    """Compute (cx, cy) center of bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def create_manual_incident(
    camera_id: str,
    tracklet: Tracklet,
    time_s: float | None = None,
    description: str = "Operator flagged vehicle incident",
) -> Incident:
    """Create an incident event manually triggered by an operator selecting a vehicle."""
    inc_time = time_s if time_s is not None else tracklet.end_time_s
    return Incident(
        incident_id=f"inc_{uuid.uuid4().hex[:8]}",
        camera_id=camera_id,
        time_s=round(inc_time, 2),
        involved_tracklet_ids=[tracklet.tracklet_id],
        confidence=1.0,
        description=description,
    )


class KinematicIncidentDetector:
    """Detects kinematic anomalies from vehicle tracklets (sudden stops, stalled vehicles)."""

    def __init__(
        self,
        stop_velocity_thresh: float = 2.0,  # pixels per frame considered stopped
        min_stopped_frames: int = 15,       # frames stopped to count as stalled / abrupt stop
        deceleration_thresh: float = 8.0,    # abrupt speed drop (px / frame^2)
    ) -> None:
        self.stop_velocity_thresh = stop_velocity_thresh
        self.min_stopped_frames = min_stopped_frames
        self.deceleration_thresh = deceleration_thresh

    def analyze_tracklet(self, tracklet: Tracklet, fps: float = 30.0) -> list[Incident]:
        """Analyze a single tracklet for sudden stop or stalled vehicle anomalies."""
        boxes = tracklet.boxes
        frames = tracklet.frames
        n = len(boxes)

        if n < self.min_stopped_frames:
            return []

        centers = [_box_center(b) for b in boxes]
        displacements = [
            float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
            for i in range(1, n)
        ]

        incidents: list[Incident] = []

        # Check for sudden deceleration: high speed initially followed by near-zero speed
        if len(displacements) >= 10:
            window_size = 5
            speeds = [
                float(np.mean(displacements[i : i + window_size]))
                for i in range(len(displacements) - window_size)
            ]

            for i in range(len(speeds) - 1):
                speed_drop = speeds[i] - speeds[i + 1]
                if speed_drop >= self.deceleration_thresh and speeds[i + 1] <= self.stop_velocity_thresh:
                    inc_frame = frames[min(i + window_size, n - 1)]
                    inc_time = (inc_frame / fps) + tracklet.start_time_s
                    inc = Incident(
                        incident_id=f"inc_decel_{uuid.uuid4().hex[:6]}",
                        camera_id=tracklet.camera_id,
                        time_s=round(inc_time, 2),
                        involved_tracklet_ids=[tracklet.tracklet_id],
                        confidence=min(0.95, round(speed_drop / 15.0, 2)),
                        description=f"Abrupt deceleration detected on {tracklet.vehicle_class} ({tracklet.tracklet_id})",
                    )
                    incidents.append(inc)
                    break

        # Check for prolonged stationary / stalled vehicle
        stationary_count = 0
        for disp in displacements:
            if disp <= self.stop_velocity_thresh:
                stationary_count += 1
            else:
                stationary_count = 0

            if stationary_count >= self.min_stopped_frames:
                inc_time = tracklet.end_time_s
                inc = Incident(
                    incident_id=f"inc_stall_{uuid.uuid4().hex[:6]}",
                    camera_id=tracklet.camera_id,
                    time_s=round(inc_time, 2),
                    involved_tracklet_ids=[tracklet.tracklet_id],
                    confidence=0.85,
                    description=f"Stationary / stalled {tracklet.vehicle_class} ({tracklet.tracklet_id}) in roadway",
                )
                incidents.append(inc)
                break

        return incidents


def detect_incidents_from_tracklets(
    tracklets: list[Tracklet],
    fps: float = 30.0,
    config: dict[str, Any] | None = None,
) -> list[Incident]:
    """Run kinematic incident detection on a pool of tracklets."""
    detector = KinematicIncidentDetector()
    all_incidents: list[Incident] = []
    for trk in tracklets:
        incs = detector.analyze_tracklet(trk, fps=fps)
        all_incidents.extend(incs)
    return all_incidents
