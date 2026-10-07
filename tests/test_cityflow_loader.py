from pathlib import Path

from incident_search.io.cityflow import (
    discover_cityflow_cameras,
    gt_detections_to_tracklets,
    parse_cityflow_gt,
)


def test_parse_cityflow_gt_and_tracklets(tmp_path: Path):
    cam_dir = tmp_path / "train" / "S01" / "c001" / "gt"
    cam_dir.mkdir(parents=True)
    gt_path = cam_dir / "gt.txt"
    gt_path.write_text(
        "1,10,100,120,40,20,1.5,2.5\n"
        "2,10,102,121,40,20,1.6,2.6\n"
        "4,11,200,220,50,30,3.0,4.0\n",
        encoding="utf-8",
    )

    cameras = discover_cityflow_cameras(tmp_path, default_fps=10.0)
    assert len(cameras) == 1
    assert cameras[0].scenario_id == "S01"
    assert cameras[0].camera_id == "c001"

    detections = parse_cityflow_gt(gt_path, "S01", "c001", fps=10.0)
    assert len(detections) == 3
    assert detections[0].time_s == 0.0
    assert detections[1].time_s == 0.1
    assert detections[0].bbox_xyxy == (100.0, 120.0, 140.0, 140.0)

    tracklets = gt_detections_to_tracklets(detections, min_frames=2)
    assert len(tracklets) == 1
    assert tracklets[0].tracklet_id == "S01_c001_gt10"
    assert tracklets[0].gt_vehicle_id == "10"
    assert tracklets[0].frames == [1, 2]
