import numpy as np

from incident_search.eval.baseline import cosine_score_matrix
from incident_search.eval.metrics import evaluate_retrieval
from incident_search.eval.protocol import (
    assert_identity_disjoint,
    build_upstream_protocol,
    make_identity_splits,
    true_match_survival,
)
from incident_search.io.schema import Tracklet


def _tracklet(tracklet_id: str, camera_id: str, start: float, end: float, identity: str) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        start_time_s=start,
        end_time_s=end,
        frames=[int(start), int(end)],
        boxes=[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
        scores=[1.0, 1.0],
        vehicle_class="car",
        gt_vehicle_id=identity,
    )


def test_upstream_protocol_excludes_same_camera_and_future():
    query = _tracklet("q", "c003", 30.0, 35.0, "v1")
    gallery = [
        _tracklet("same_id_earlier", "c001", 10.0, 12.0, "v1"),
        _tracklet("same_id_same_cam", "c003", 20.0, 21.0, "v1"),
        _tracklet("same_id_future", "c002", 40.0, 41.0, "v1"),
        _tracklet("negative", "c002", 11.0, 12.0, "v2"),
    ]

    eligible, positives = build_upstream_protocol([query], gallery, time_window_s=25.0)
    assert eligible.tolist() == [[True, False, False, True]]
    assert positives.tolist() == [[True, False, False, False]]


def test_identity_splits_are_disjoint():
    tracklets = [_tracklet(f"t{i}", "c001", float(i), float(i + 1), f"v{i}") for i in range(10)]
    splits = make_identity_splits(tracklets, seed=7)
    assert_identity_disjoint(splits)
    assert len(splits.train | splits.val | splits.test) == 10


def test_retrieval_metrics_multiple_positives():
    scores = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    positives = np.array([[False, True, False, True]])
    metrics, per_query = evaluate_retrieval(scores, positives)

    # Positives are ranked 2 and 4: AP = (1/2 + 2/4) / 2 = 0.5
    assert metrics.num_queries == 1
    assert metrics.mAP == 0.5
    assert metrics.rank1 == 0.0
    assert metrics.rank5 == 1.0
    assert per_query[0]["num_positives"] == 2


def test_cosine_score_matrix():
    queries = [_tracklet("q", "c001", 1.0, 2.0, "v1")]
    gallery = [
        _tracklet("g1", "c002", 0.0, 1.0, "v1"),
        _tracklet("g2", "c003", 0.0, 1.0, "v2"),
    ]
    embedding_map = {
        "q": np.array([1.0, 0.0], dtype=np.float32),
        "g1": np.array([0.5, 0.5], dtype=np.float32),
        "g2": np.array([0.0, 1.0], dtype=np.float32),
    }
    scores = cosine_score_matrix(queries, gallery, embedding_map)
    assert scores.shape == (1, 2)
    assert scores.tolist() == [[0.5, 0.0]]


def test_true_match_survival():
    before_positive = np.array([[True, False, False], [False, True, False]])
    after_eligible = np.array([[False, True, False], [False, True, False]])
    assert true_match_survival(before_positive, after_eligible) == 0.5
