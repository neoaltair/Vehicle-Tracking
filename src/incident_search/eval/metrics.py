"""Transparent NumPy retrieval metrics for the project protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics over valid queries."""

    num_queries: int
    mAP: float
    rank1: float
    rank5: float
    rank10: float


def _average_precision(sorted_positive: np.ndarray) -> float:
    """Compute AP for one ranked list with one or more positives."""
    positive_positions = np.flatnonzero(sorted_positive)
    if len(positive_positions) == 0:
        return 0.0

    precisions = []
    for pos in positive_positions:
        precisions.append(float(np.sum(sorted_positive[: pos + 1]) / (pos + 1)))
    return float(np.mean(precisions))


def evaluate_retrieval(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    eligible_mask: np.ndarray | None = None,
    ranks: Iterable[int] = (1, 5, 10),
) -> tuple[RetrievalMetrics, list[dict[str, float | int]]]:
    """Evaluate mAP and CMC/Recall@K for multi-positive retrieval.

    Args:
        scores: Array of shape `(num_queries, num_gallery)`; higher is better.
        positive_mask: Boolean array with the same shape marking valid positives.
        eligible_mask: Optional boolean array with the same shape. Ineligible
            candidates are removed before ranking.
        ranks: Rank cutoffs to include in per-query rows.

    Returns:
        Aggregate metrics and per-query rows. Queries with no eligible positive
        are skipped because mAP/CMC are undefined for them.
    """
    scores = np.asarray(scores, dtype=np.float32)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if scores.shape != positive_mask.shape:
        raise ValueError("scores and positive_mask must have identical shapes")

    if eligible_mask is None:
        eligible_mask = np.ones_like(positive_mask, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible_mask, dtype=bool)
        if eligible_mask.shape != scores.shape:
            raise ValueError("eligible_mask must match scores shape")

    rank_list = sorted(set(int(rank) for rank in ranks))
    per_query: list[dict[str, float | int]] = []

    for query_idx in range(scores.shape[0]):
        eligible = eligible_mask[query_idx]
        positives = positive_mask[query_idx] & eligible
        if not np.any(positives):
            continue

        eligible_scores = scores[query_idx, eligible]
        eligible_positive = positives[eligible]
        order = np.argsort(-eligible_scores, kind="mergesort")
        sorted_positive = eligible_positive[order]

        row: dict[str, float | int] = {
            "query_index": query_idx,
            "num_candidates": int(len(sorted_positive)),
            "num_positives": int(np.sum(sorted_positive)),
            "ap": _average_precision(sorted_positive),
        }
        for rank in rank_list:
            row[f"rank{rank}"] = float(np.any(sorted_positive[:rank]))
        per_query.append(row)

    if not per_query:
        return RetrievalMetrics(num_queries=0, mAP=0.0, rank1=0.0, rank5=0.0, rank10=0.0), []

    def _mean(key: str) -> float:
        return float(np.mean([float(row.get(key, 0.0)) for row in per_query]))

    metrics = RetrievalMetrics(
        num_queries=len(per_query),
        mAP=_mean("ap"),
        rank1=_mean("rank1"),
        rank5=_mean("rank5"),
        rank10=_mean("rank10"),
    )
    return metrics, per_query
