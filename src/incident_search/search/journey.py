"""Cross-camera vehicle journey reconstruction and incident search."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from incident_search.graph.camera_graph import CameraTransitionGraph
from incident_search.io.schema import Incident, SearchResultItem, Tracklet
from incident_search.search.spatiotemporal import compute_fused_scores

logger = logging.getLogger(__name__)


@dataclass
class CameraSighting:
    """Best matched sighting of the query vehicle at a single camera."""

    camera_id: str
    tracklet: Tracklet
    score: float
    appearance_score: float
    prior_score: float
    timestamp_s: float
    time_delta_s: float  # Difference from query incident time (seconds)
    crop_path: str | None = None
    alternatives: list[SearchResultItem] = field(default_factory=list)


@dataclass
class JourneyResult:
    """Reconstructed cross-camera route and timeline for an incident vehicle."""

    incident: Incident
    query_tracklet: Tracklet
    query_crop_path: str | None
    sightings: list[CameraSighting]  # Sorted chronologically (earliest to latest)
    candidates_evaluated: int
    candidates_pruned: int
    pruning_reduction: float

    @property
    def route_str(self) -> str:
        """Formatted string of the reconstructed camera route."""
        cam_names = [s.camera_id for s in self.sightings] + [self.incident.camera_id]
        return " ➔ ".join(cam_names)


class CrossCameraJourneyReconstructor:
    """Reconstructs the multi-camera journey of an incident vehicle across CCTV cameras."""

    def __init__(
        self,
        camera_graph: CameraTransitionGraph | None = None,
        default_time_window_s: float = 300.0,
        alpha: float = 0.05,
    ) -> None:
        self.camera_graph = camera_graph or CameraTransitionGraph()
        self.default_time_window_s = default_time_window_s
        self.alpha = alpha

    def reconstruct_journey(
        self,
        incident: Incident,
        query_tracklet: Tracklet,
        background_tracklets: list[Tracklet],
        embedding_map: dict[str, np.ndarray],
        time_window_s: float | None = None,
        alpha: float | None = None,
        min_score: float = 0.20,
    ) -> JourneyResult:
        """Reconstruct the upstream journey of the incident vehicle across background CCTV cameras."""
        window_s = time_window_s if time_window_s is not None else self.default_time_window_s
        weight_alpha = alpha if alpha is not None else self.alpha

        q_emb = embedding_map.get(query_tracklet.tracklet_id)
        if q_emb is None:
            raise ValueError(f"No embedding found for query vehicle {query_tracklet.tracklet_id}")

        # Group background tracklets by camera (excluding incident camera)
        by_camera: dict[str, list[Tracklet]] = {}
        for trk in background_tracklets:
            if trk.camera_id == incident.camera_id:
                continue
            if trk.tracklet_id not in embedding_map:
                continue
            by_camera.setdefault(trk.camera_id, []).append(trk)

        total_candidates = sum(len(trks) for trks in by_camera.values())
        surviving_count = 0
        sightings: list[CameraSighting] = []

        q_mat = q_emb[None, :]

        for cam_id, trks in by_camera.items():
            # Temporal filtering: candidate observation must be upstream (prior to incident)
            eligible_trks: list[Tracklet] = []
            for trk in trks:
                # dt is the time elapsed between candidate observation and incident
                dt = incident.time_s - trk.end_time_s
                if dt < -5.0:  # Observation occurred after incident
                    continue
                if window_s is not None and dt > window_s:
                    continue
                eligible_trks.append(trk)

            if not eligible_trks:
                continue

            surviving_count += len(eligible_trks)

            # Appearance scores
            c_embs = np.stack([embedding_map[t.tracklet_id] for t in eligible_trks])
            app_scores = (q_mat @ c_embs.T).astype(np.float32)

            # Prior scores from graph if available
            priors = np.zeros((1, len(eligible_trks)), dtype=np.float32)
            for j, t in enumerate(eligible_trks):
                dt = abs(incident.time_s - t.start_time_s)
                priors[0, j] = self.camera_graph.get_prior(cam_id, incident.camera_id, dt)

            fused_scores = compute_fused_scores(app_scores, priors, alpha=weight_alpha)[0]

            ranked_items: list[SearchResultItem] = []
            for j, t in enumerate(eligible_trks):
                score = float(fused_scores[j])
                if score >= min_score:
                    ranked_items.append(
                        SearchResultItem(
                            tracklet_id=t.tracklet_id,
                            score=score,
                            appearance_score=float(app_scores[0, j]),
                            prior_score=float(priors[0, j]),
                            camera_id=t.camera_id,
                            start_time_s=t.start_time_s,
                            end_time_s=t.end_time_s,
                            crop_paths=t.crop_paths,
                        )
                    )

            if not ranked_items:
                continue

            ranked_items.sort(key=lambda x: x.score, reverse=True)
            best_match = ranked_items[0]
            best_trk = next(t for t in eligible_trks if t.tracklet_id == best_match.tracklet_id)

            crop_p = None
            valid_crops = [p for p in best_trk.crop_paths if Path(p).exists()]
            if valid_crops:
                crop_p = valid_crops[len(valid_crops) // 2]

            dt_diff = round(incident.time_s - best_trk.start_time_s, 1)

            sighting = CameraSighting(
                camera_id=cam_id,
                tracklet=best_trk,
                score=best_match.score,
                appearance_score=best_match.appearance_score,
                prior_score=best_match.prior_score,
                timestamp_s=best_trk.start_time_s,
                time_delta_s=dt_diff,
                crop_path=crop_p,
                alternatives=ranked_items[1:5],
            )
            sightings.append(sighting)

        # Sort sightings chronologically (earliest observation first)
        sightings.sort(key=lambda s: s.timestamp_s)

        query_crop = None
        q_valid_crops = [p for p in query_tracklet.crop_paths if Path(p).exists()]
        if q_valid_crops:
            query_crop = q_valid_crops[len(q_valid_crops) // 2]

        pruning_red = 1.0 - (surviving_count / max(1, total_candidates))

        return JourneyResult(
            incident=incident,
            query_tracklet=query_tracklet,
            query_crop_path=query_crop,
            sightings=sightings,
            candidates_evaluated=total_candidates,
            candidates_pruned=total_candidates - surviving_count,
            pruning_reduction=round(pruning_red, 3),
        )
