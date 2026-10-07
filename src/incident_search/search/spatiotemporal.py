"""Spatio-temporal candidate filtering and fused retrieval scoring for M4."""

from __future__ import annotations

import numpy as np

from incident_search.graph.camera_graph import CameraTransitionGraph
from incident_search.io.schema import Tracklet


def build_graph_prior_matrix(
    queries: list[Tracklet],
    gallery: list[Tracklet],
    camera_graph: CameraTransitionGraph,
    min_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute travel-time prior probabilities and connectivity mask for query-gallery pairs.

    Recall the upstream relationship:
    If candidate C_c is an upstream observation of query C_q, then the vehicle traveled
    from candidate camera C_c to query camera C_q.
    Therefore, the transition is: src = candidate.camera_id -> dst = query.camera_id.
    Travel time delta_t = query.start_time_s - candidate.end_time_s (or query.start - candidate.start).

    Args:
        queries: List of query tracklets.
        gallery: List of candidate gallery tracklets.
        camera_graph: Fitted CameraTransitionGraph.
        min_prob: Threshold below which prior is considered zero.

    Returns:
        prior_matrix: Array of shape (num_queries, num_gallery) with prior densities.
        connected_mask: Boolean mask where edge exists and prior > min_prob.
    """
    n_q = len(queries)
    n_g = len(gallery)
    priors = np.zeros((n_q, n_g), dtype=np.float32)
    connected = np.zeros((n_q, n_g), dtype=bool)

    for i, q in enumerate(queries):
        for j, g in enumerate(gallery):
            if q.camera_id == g.camera_id:
                continue
            if g.end_time_s > q.start_time_s:
                continue

            dt = q.start_time_s - g.end_time_s
            if dt <= 0:
                dt = q.start_time_s - g.start_time_s
            if dt <= 0:
                continue

            # Upstream transition: candidate camera -> query camera
            prob = camera_graph.get_prior(g.camera_id, q.camera_id, dt)
            priors[i, j] = prob
            if prob > min_prob:
                connected[i, j] = True

    return priors, connected


def compute_fused_scores(
    appearance_scores: np.ndarray,
    prior_matrix: np.ndarray,
    alpha: float = 0.1,
    normalize_per_query: bool = True,
) -> np.ndarray:
    """Fuse appearance cosine similarity and travel-time spatio-temporal prior.

    score = appearance_score + alpha * normalized_prior

    Args:
        appearance_scores: Array of shape (num_queries, num_gallery), e.g. cosine similarities in [-1, 1].
        prior_matrix: Array of shape (num_queries, num_gallery) with prior values >= 0.
        alpha: Weight balancing appearance and spatio-temporal prior.
        normalize_per_query: If True, normalize prior by max_prior for each query.

    Returns:
        fused_scores: Array of shape (num_queries, num_gallery).
    """
    priors = np.asarray(prior_matrix, dtype=np.float32)
    if normalize_per_query:
        norm_priors = np.zeros_like(priors)
        max_vals = np.max(priors, axis=1, keepdims=True)
        valid = max_vals > 0
        np.divide(priors, max_vals, out=norm_priors, where=valid)
    else:
        max_val = np.max(priors)
        norm_priors = (priors / max_val) if max_val > 0 else priors

    return appearance_scores + (alpha * norm_priors)
