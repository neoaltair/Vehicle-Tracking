"""Unit tests for Pydantic data schemas and serialization."""

import tempfile
from pathlib import Path

import pytest

from incident_search.io.schema import (
    Camera,
    Incident,
    SearchResult,
    SearchResultItem,
    Tracklet,
    load_tracklets,
    save_tracklets,
)


def test_camera_schema():
    cam = Camera(
        camera_id="cam_01",
        video_paths=["/path/to/vdo.mp4"],
        fps=25.0,
        time_offset_s=1.5,
        location={"lat": 40.7128, "lon": -74.0060},
    )
    assert cam.camera_id == "cam_01"
    assert cam.fps == 25.0
    assert cam.time_offset_s == 1.5
    assert cam.location == {"lat": 40.7128, "lon": -74.0060}


def test_tracklet_roundtrip():
    t = Tracklet(
        tracklet_id="cam01_trk001",
        camera_id="cam01",
        start_time_s=10.0,
        end_time_s=12.5,
        frames=[100, 101, 102],
        boxes=[[10.0, 20.0, 50.0, 60.0], [12.0, 22.0, 52.0, 62.0], [14.0, 24.0, 54.0, 64.0]],
        scores=[0.92, 0.95, 0.91],
        vehicle_class="car",
        gt_vehicle_id="v_1001",
        crop_paths=["crops/cam01_trk001_f101.jpg"],
    )
    json_str = t.to_json()
    t2 = Tracklet.from_json(json_str)

    assert t2.tracklet_id == t.tracklet_id
    assert t2.camera_id == t.camera_id
    assert t2.frames == [100, 101, 102]
    assert len(t2.boxes) == 3
    assert t2.gt_vehicle_id == "v_1001"


def test_save_load_tracklets():
    tracklets = [
        Tracklet(
            tracklet_id=f"trk_{i}",
            camera_id="c01",
            start_time_s=float(i),
            end_time_s=float(i + 1),
            frames=[i, i + 1],
            boxes=[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
            scores=[0.8, 0.85],
            vehicle_class="car",
        )
        for i in range(3)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "tracklets.json"
        save_tracklets(tracklets, file_path)
        loaded = load_tracklets(file_path)
        assert len(loaded) == 3
        assert loaded[1].tracklet_id == "trk_1"


def test_search_result_pruning_reduction():
    sr = SearchResult(
        candidates_before_pruning=100,
        candidates_after_pruning=25,
        mode="appearance_graph_prior",
        results=[
            SearchResultItem(
                tracklet_id="trk_01",
                score=0.95,
                appearance_score=0.85,
                prior_score=0.10,
                camera_id="c02",
                start_time_s=15.0,
                end_time_s=18.0,
            )
        ],
    )
    assert pytest.approx(sr.pruning_reduction, rel=1e-3) == 0.75
    json_str = sr.to_json()
    sr_loaded = SearchResult.from_json(json_str)
    assert len(sr_loaded.results) == 1
    assert sr_loaded.results[0].tracklet_id == "trk_01"


def test_incident_schema():
    inc = Incident(
        incident_id="inc_001",
        camera_id="cam_01",
        time_s=42.5,
        involved_tracklet_ids=["trk_01", "trk_02"],
        confidence=0.88,
        description="Sudden deceleration and overlap",
    )
    json_str = inc.to_json()
    inc2 = Incident.from_json(json_str)
    assert inc2.incident_id == "inc_001"
    assert len(inc2.involved_tracklet_ids) == 2
