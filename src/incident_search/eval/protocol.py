"""Query/gallery construction and split checks for upstream retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from incident_search.io.schema import Tracklet


@dataclass(frozen=True)
class SplitIds:
    """Identity-disjoint split assignment."""

    train: set[str]
    val: set[str]
    test: set[str]


def _tracklet_identity(tracklet: Tracklet) -> str:
    if tracklet.gt_vehicle_id is None:
        raise ValueError(f"Tracklet {tracklet.tracklet_id} is missing gt_vehicle_id")
    return tracklet.gt_vehicle_id


def make_identity_splits(
    tracklets: list[Tracklet],
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> SplitIds:
    """Create deterministic identity-disjoint train/val/test IDs."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    identities = sorted({_tracklet_identity(tracklet) for tracklet in tracklets})
    rng = np.random.default_rng(seed)
    shuffled = list(identities)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_fraction))
    n_val = int(round(n_total * val_fraction))

    train = set(shuffled[:n_train])
    val = set(shuffled[n_train : n_train + n_val])
    test = set(shuffled[n_train + n_val :])
    return SplitIds(train=train, val=val, test=test)


def assert_identity_disjoint(split_ids: SplitIds) -> None:
    """Raise if any identity appears in more than one split."""
    overlaps = {
        "train_val": split_ids.train & split_ids.val,
        "train_test": split_ids.train & split_ids.test,
        "val_test": split_ids.val & split_ids.test,
    }
    bad = {name: ids for name, ids in overlaps.items() if ids}
    if bad:
        raise ValueError(f"Identity leakage across splits: {bad}")


def filter_tracklets_by_identities(tracklets: list[Tracklet], identities: set[str]) -> list[Tracklet]:
    """Return tracklets whose GT identity is in `identities`."""
    return [tracklet for tracklet in tracklets if _tracklet_identity(tracklet) in identities]


def build_upstream_protocol(
    queries: list[Tracklet],
    gallery: list[Tracklet],
    time_window_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build eligible and positive masks for upstream cross-camera retrieval.

    A gallery tracklet is eligible when it is from a different camera and ended
    no later than the query start time. If `time_window_s` is provided, its end
    time must also be within `[Tq - time_window_s, Tq]`.

    A positive is an eligible gallery tracklet with the same GT vehicle identity.
    """
    eligible = np.zeros((len(queries), len(gallery)), dtype=bool)
    positives = np.zeros_like(eligible)

    for query_idx, query in enumerate(queries):
        query_identity = _tracklet_identity(query)
        query_time = query.start_time_s
        lower_bound = None if time_window_s is None else query_time - time_window_s

        for gallery_idx, candidate in enumerate(gallery):
            if candidate.tracklet_id == query.tracklet_id:
                continue
            if candidate.camera_id == query.camera_id:
                continue
            if candidate.end_time_s > query_time:
                continue
            if lower_bound is not None and candidate.end_time_s < lower_bound:
                continue

            eligible[query_idx, gallery_idx] = True
            positives[query_idx, gallery_idx] = _tracklet_identity(candidate) == query_identity

    return eligible, positives


def pruning_reduction(before_mask: np.ndarray, after_mask: np.ndarray) -> float:
    """Compute average `1 - after / before` over queries with nonzero candidates."""
    before_counts = np.sum(before_mask, axis=1)
    after_counts = np.sum(after_mask, axis=1)
    valid = before_counts > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(1.0 - (after_counts[valid] / before_counts[valid])))


def true_match_survival(before_positive: np.ndarray, after_eligible: np.ndarray) -> float:
    """Fraction of queries where a pre-pruning positive survives pruning."""
    has_positive_before = np.any(before_positive, axis=1)
    if not np.any(has_positive_before):
        return 0.0
    survives = np.any(before_positive & after_eligible, axis=1)
    return float(np.mean(survives[has_positive_before]))
