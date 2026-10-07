"""Evaluation utilities for retrieval experiments."""

from incident_search.eval.baseline import (
    cosine_score_matrix,
    load_tracklet_embedding_map,
    write_per_query_csv,
)
from incident_search.eval.metrics import RetrievalMetrics, evaluate_retrieval
from incident_search.eval.protocol import (
    SplitIds,
    assert_identity_disjoint,
    build_upstream_protocol,
    make_identity_splits,
)

__all__ = [
    "RetrievalMetrics",
    "cosine_score_matrix",
    "evaluate_retrieval",
    "load_tracklet_embedding_map",
    "write_per_query_csv",
    "SplitIds",
    "assert_identity_disjoint",
    "build_upstream_protocol",
    "make_identity_splits",
]
