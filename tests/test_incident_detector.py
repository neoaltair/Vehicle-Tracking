"""Unit tests for kinematic incident detection and triggering (M5)."""

from __future__ import annotations

from incident_search.incident.detector import (
    KinematicIncidentDetector,
    create_manual_incident,
    detect_incidents_from_tracklets,
)
from incident_search.io.schema import Tracklet


def _make_tracklet(tracklet_id: str, camera_id: str, centers: list[tuple[float, float]]) -> Tracklet:
    boxes = [[cx - 20, cy - 15, cx + 20, cy + 15] for cx, cy in centers]
    frames = list(range(len(centers)))
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        start_time_s=0.0,
        end_time_s=len(centers) / 30.0,
        frames=frames,
        boxes=boxes,
        scores=[0.9] * len(centers),
        vehicle_class="car",
    )


def test_manual_incident_creation():
    trk = _make_tracklet("trk_01", "Cam_Main", [(100, 100), (120, 100), (140, 100)])
    inc = create_manual_incident("Cam_Main", trk, time_s=5.0, description="Collision at intersection")
    assert inc.camera_id == "Cam_Main"
    assert inc.time_s == 5.0
    assert inc.involved_tracklet_ids == ["trk_01"]
    assert inc.confidence == 1.0


def test_kinematic_detector_sudden_deceleration():
    # Vehicle moving fast (25 px/frame) then suddenly stops (0 px/frame)
    centers = [(float(i * 25), 100.0) for i in range(12)] + [(float(11 * 25), 100.0)] * 10
    trk = _make_tracklet("trk_decel", "Cam_Main", centers)

    detector = KinematicIncidentDetector(stop_velocity_thresh=2.0, deceleration_thresh=5.0)
    incidents = detector.analyze_tracklet(trk)

    assert len(incidents) >= 1
    assert "decel" in incidents[0].incident_id or "Abrupt" in incidents[0].description
    assert incidents[0].involved_tracklet_ids == ["trk_decel"]


def test_kinematic_detector_stationary_vehicle():
    # Vehicle stationary for 20 frames
    centers = [(150.0, 200.0)] * 25
    trk = _make_tracklet("trk_stalled", "Cam_Main", centers)

    incidents = detect_incidents_from_tracklets([trk])
    assert len(incidents) >= 1
    assert "stalled" in incidents[0].description.lower() or "stationary" in incidents[0].description.lower()
