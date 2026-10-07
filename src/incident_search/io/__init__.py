"""Input/Output and schema definitions."""

from incident_search.io.cityflow import (
    CityFlowCamera,
    CityFlowDetection,
    discover_cityflow_cameras,
    gt_detections_to_tracklets,
    load_cityflow_gt_tracklets,
    parse_cityflow_gt,
    save_tracklet_crops,
)
from incident_search.io.schema import (
    Camera,
    EmbeddingRecord,
    Incident,
    SearchQuery,
    SearchResult,
    SearchResultItem,
    Tracklet,
    load_tracklets,
    save_tracklets,
)

__all__ = [
    "CityFlowCamera",
    "CityFlowDetection",
    "discover_cityflow_cameras",
    "gt_detections_to_tracklets",
    "load_cityflow_gt_tracklets",
    "parse_cityflow_gt",
    "save_tracklet_crops",
    "Camera",
    "Tracklet",
    "EmbeddingRecord",
    "SearchQuery",
    "SearchResult",
    "SearchResultItem",
    "Incident",
    "save_tracklets",
    "load_tracklets",
]
