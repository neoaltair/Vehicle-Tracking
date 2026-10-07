"""Data schemas for Incident-Triggered Multi-Camera Vehicle Search.

All modules communicate through these Pydantic v2 models.
Supports JSON serialization for tracklets, incidents, queries, and search results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Camera(BaseModel):
    """Camera specification within the surveillance network."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(description="Unique camera identifier across network")
    video_paths: list[str] = Field(description="List of video file paths for this camera")
    fps: float = Field(default=30.0, description="Frames per second")
    time_offset_s: float = Field(
        default=0.0,
        description="Offset in seconds relative to common global synchronized clock",
    )
    location: dict[str, float] | None = Field(
        default=None,
        description="Optional GPS (lat, lon) or map coordinates (x, y)",
    )


class Tracklet(BaseModel):
    """Single-camera vehicle tracklet across consecutive or continuous frames."""

    model_config = ConfigDict(extra="forbid")

    tracklet_id: str = Field(description="Unique tracklet ID (e.g. cam01_trk0042)")
    camera_id: str = Field(description="Camera where observation occurred")
    start_time_s: float = Field(description="Global timestamp of first observation in seconds")
    end_time_s: float = Field(description="Global timestamp of last observation in seconds")
    frames: list[int] = Field(description="Frame indices relative to camera video")
    boxes: list[list[float]] = Field(
        description="Bounding boxes [[x1, y1, x2, y2]] in pixel coordinates"
    )
    scores: list[float] = Field(description="Detection/tracking confidence scores per frame")
    vehicle_class: str = Field(
        default="car",
        description="Detected vehicle class (car, bus, truck, motorcycle)",
    )
    gt_vehicle_id: str | None = Field(
        default=None,
        description="Ground-truth vehicle identity (for eval/training priors only)",
    )
    crop_paths: list[str] = Field(
        default_factory=list,
        description="Relative or absolute file paths to saved sampled vehicle crops",
    )

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize tracklet to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Tracklet:
        """Deserialize tracklet from JSON string."""
        return cls.model_validate_json(json_str)


class EmbeddingRecord(BaseModel):
    """Metadata and paths/embeddings for a tracklet."""

    model_config = ConfigDict(extra="forbid")

    tracklet_id: str = Field(description="Associated tracklet ID")
    model_name: str = Field(description="Name of the re-ID backbone architecture")
    model_version: str = Field(description="Weights checkpoint identifier or URL")
    embedding_dim: int = Field(default=2048, description="Feature embedding dimensionality")
    embedding_path: str | None = Field(
        default=None,
        description="Path to saved .npz file storing frame and aggregated embeddings",
    )


class SearchQuery(BaseModel):
    """Incident search query specification."""

    model_config = ConfigDict(extra="forbid")

    query_camera_id: str = Field(description="Camera ID where query vehicle was observed")
    query_time_s: float = Field(description="Global time of incident observation in seconds")
    query_tracklet_id: str | None = Field(
        default=None,
        description="Tracklet ID if query is an extracted tracklet",
    )
    crop_path: str | None = Field(
        default=None,
        description="Path to a single crop image if query is a single frame/crop",
    )
    gt_vehicle_id: str | None = Field(
        default=None,
        description="Optional ground-truth ID for evaluation metrics only",
    )


class SearchResultItem(BaseModel):
    """Individual ranked retrieval candidate."""

    model_config = ConfigDict(extra="forbid")

    tracklet_id: str = Field(description="Retrieved tracklet ID")
    score: float = Field(description="Final combined retrieval score")
    appearance_score: float = Field(description="Cosine similarity of re-ID embeddings")
    prior_score: float = Field(
        default=0.0,
        description="Normalized spatio-temporal transition prior score",
    )
    camera_id: str = Field(description="Camera ID of retrieved tracklet")
    start_time_s: float = Field(description="Global start time of candidate observation")
    end_time_s: float = Field(description="Global end time of candidate observation")
    crop_paths: list[str] = Field(default_factory=list, description="Thumbnails / crop paths")


class SearchResult(BaseModel):
    """Full search result output with pre- and post-pruning metrics."""

    model_config = ConfigDict(extra="forbid")

    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="Ranked candidate list in descending score order",
    )
    candidates_before_pruning: int = Field(description="Initial candidate count in pool")
    candidates_after_pruning: int = Field(description="Surviving candidate count after pruning")
    mode: str = Field(
        default="appearance_only",
        description="Search mode used (appearance_only, appearance_window, appearance_graph_prior)",
    )

    @property
    def pruning_reduction(self) -> float:
        """Fraction of candidate search space pruned away: 1 - (after / before)."""
        if self.candidates_before_pruning <= 0:
            return 0.0
        return 1.0 - (self.candidates_after_pruning / self.candidates_before_pruning)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize search result to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> SearchResult:
        """Deserialize search result from JSON string."""
        return cls.model_validate_json(json_str)


class Incident(BaseModel):
    """Rule-based or ground-truth incident detection event."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(description="Unique incident identifier")
    camera_id: str = Field(description="Camera ID where incident occurred")
    time_s: float = Field(description="Global time of incident occurrence in seconds")
    involved_tracklet_ids: list[str] = Field(
        description="Tracklet IDs involved in the incident"
    )
    confidence: float = Field(description="Detector confidence score in [0.0, 1.0]")
    description: str | None = Field(default=None, description="Optional incident description")

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize incident to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Incident:
        """Deserialize incident from JSON string."""
        return cls.model_validate_json(json_str)


def save_tracklets(tracklets: list[Tracklet], path: str | Path) -> None:
    """Save a list of tracklets to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [t.model_dump() for t in tracklets]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_tracklets(path: str | Path) -> list[Tracklet]:
    """Load a list of tracklets from a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return [Tracklet.model_validate(item) for item in data]
