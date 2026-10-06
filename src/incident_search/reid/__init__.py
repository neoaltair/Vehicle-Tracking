"""Vehicle re-identification and embedding extraction module."""

from incident_search.reid.extractor import (
    FastReIDExtractor,
    aggregate_embeddings,
    l2_normalize,
    load_embeddings,
    save_embeddings,
)

__all__ = [
    "FastReIDExtractor",
    "aggregate_embeddings",
    "l2_normalize",
    "save_embeddings",
    "load_embeddings",
]
