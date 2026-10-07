"""Appearance-only retrieval baseline helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from incident_search.io.schema import Tracklet
from incident_search.reid.extractor import load_embeddings


def load_tracklet_embedding_map(
    tracklets: list[Tracklet],
    cache_dir: str | Path,
    model_name: str = "sbs_R50-ibn",
) -> dict[str, np.ndarray]:
    """Load cached tracklet embeddings for the supplied tracklets."""
    embedding_map: dict[str, np.ndarray] = {}
    for tracklet in tracklets:
        cached = load_embeddings(cache_dir, tracklet.tracklet_id, model_name=model_name)
        if cached is None:
            continue
        _frame_embeddings, tracklet_embedding = cached
        embedding_map[tracklet.tracklet_id] = tracklet_embedding.astype(np.float32)
    return embedding_map


def cosine_score_matrix(
    queries: list[Tracklet],
    gallery: list[Tracklet],
    embedding_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Build an exact cosine-score matrix from L2-normalized embeddings."""
    missing_queries = [tracklet.tracklet_id for tracklet in queries if tracklet.tracklet_id not in embedding_map]
    missing_gallery = [tracklet.tracklet_id for tracklet in gallery if tracklet.tracklet_id not in embedding_map]
    if missing_queries:
        raise ValueError(f"Missing query embeddings for {len(missing_queries)} tracklets")
    if missing_gallery:
        raise ValueError(f"Missing gallery embeddings for {len(missing_gallery)} tracklets")

    query_mat = np.stack([embedding_map[tracklet.tracklet_id] for tracklet in queries])
    gallery_mat = np.stack([embedding_map[tracklet.tracklet_id] for tracklet in gallery])
    return (query_mat @ gallery_mat.T).astype(np.float32)


def write_per_query_csv(rows: list[dict[str, float | int]], path: str | Path) -> None:
    """Write per-query metric rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
