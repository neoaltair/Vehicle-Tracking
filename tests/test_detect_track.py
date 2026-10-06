"""Unit tests for detection and tracking module."""

import numpy as np

from incident_search.detect_track.tracker import (
    filter_tracklets_by_length,
    sample_best_crops,
)


def test_filter_tracklets_by_length():
    raw = {
        "trk_short": {"frames": [1, 2, 3], "boxes": [], "scores": []},
        "trk_exact": {"frames": list(range(10)), "boxes": [], "scores": []},
        "trk_long": {"frames": list(range(25)), "boxes": [], "scores": []},
    }
    filtered = filter_tracklets_by_length(raw, min_len=10)
    assert "trk_short" not in filtered
    assert "trk_exact" in filtered
    assert "trk_long" in filtered
    assert len(filtered) == 2


def test_sample_best_crops():
    frames = list(range(20))
    # Boxes with varying areas
    boxes = [[0.0, 0.0, float(i * 10 + 10), float(i * 10 + 10)] for i in range(20)]
    scores = [0.5 + (i * 0.02) for i in range(20)]
    dummy_crops = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(20)]

    best = sample_best_crops(frames, boxes, scores, dummy_crops, k=5)
    assert len(best) == 5

    # Should retain high quality frames (higher score & larger area near end)
    selected_frames = [f for f, _, _ in best]
    assert 19 in selected_frames
    assert 18 in selected_frames


def test_sample_crops_when_fewer_than_k():
    frames = [1, 2, 3]
    boxes = [[0.0, 0.0, 50.0, 50.0]] * 3
    scores = [0.9] * 3
    dummy_crops = [np.zeros((10, 10, 3), dtype=np.uint8)] * 3

    best = sample_best_crops(frames, boxes, scores, dummy_crops, k=8)
    assert len(best) == 3
