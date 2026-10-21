"""Unit tests for multi-camera vehicle journey reconstruction (M5)."""

from __future__ import annotations

import numpy as np

from incident_search.incident.detector import create_manual_incident
from incident_search.io.schema import Tracklet
from incident_search.search.journey import CrossCameraJourneyReconstructor


def _make_dummy_tracklet(
    tracklet_id: str,
    camera_id: str,
    start_t: float,
    end_t: float,
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        start_time_s=start_t,
        end_time_s=end_t,
        frames=[0, 10],
        boxes=[[10, 10, 50, 50], [15, 15, 55, 55]],
        scores=[0.9, 0.9],
        vehicle_class="car",
    )


def test_journey_reconstruction_chronological_ordering():
    # Incident vehicle at Main_Cam at t = 100s
    q_trk = _make_dummy_tracklet("q_car", "Main_Cam", 95.0, 100.0)
    inc = create_manual_incident("Main_Cam", q_trk, time_s=100.0)

    # Background CCTV observations:
    # Cam_Highway at t = 20s (upstream earliest)
    # Cam_Junction at t = 60s (upstream middle)
    # Cam_Irrelevant at t = 150s (downstream future, should be excluded)
    trk_highway = _make_dummy_tracklet("car_hw", "Cam_Highway", 15.0, 20.0)
    trk_junction = _make_dummy_tracklet("car_junc", "Cam_Junction", 55.0, 60.0)
    trk_future = _make_dummy_tracklet("car_fut", "Cam_Future", 145.0, 150.0)

    # Identical embeddings to simulate strong appearance match
    emb_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    diff_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    emb_map = {
        "q_car": emb_vec,
        "car_hw": emb_vec,
        "car_junc": emb_vec,
        "car_fut": emb_vec,
        "other_car": diff_vec,
    }

    bg_tracklets = [trk_highway, trk_junction, trk_future]

    reconstructor = CrossCameraJourneyReconstructor(default_time_window_s=120.0)
    result = reconstructor.reconstruct_journey(
        incident=inc,
        query_tracklet=q_trk,
        background_tracklets=bg_tracklets,
        embedding_map=emb_map,
        time_window_s=120.0,
    )

    # Cam_Future at t=150s should be pruned (after incident at t=100)
    assert len(result.sightings) == 2
    # Chronological route order: Cam_Highway (20s) -> Cam_Junction (60s) -> Main_Cam (100s)
    assert result.sightings[0].camera_id == "Cam_Highway"
    assert result.sightings[1].camera_id == "Cam_Junction"
    assert "Cam_Highway ➔ Cam_Junction ➔ Main_Cam" in result.route_str
    assert result.pruning_reduction > 0.0
