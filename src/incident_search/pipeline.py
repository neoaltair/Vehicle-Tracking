"""End-to-end multi-camera vehicle tracking and incident search pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from incident_search.detect_track import run_detect_track
from incident_search.detect_track.live_stream import stream_detect_track
from incident_search.graph.camera_graph import CameraTransitionGraph
from incident_search.incident.detector import (
    create_manual_incident,
    detect_incidents_from_tracklets,
)
from incident_search.io.schema import Incident, SearchResult, SearchResultItem, Tracklet
from incident_search.reid.extractor import FastReIDExtractor, aggregate_embeddings
from incident_search.search.journey import (
    CrossCameraJourneyReconstructor,
    JourneyResult,
)
from incident_search.search.spatiotemporal import compute_fused_scores

logger = logging.getLogger(__name__)


@dataclass
class VideoCameraInput:
    """Specification of a camera input video."""

    camera_id: str
    video_path: str | Path
    time_offset_s: float = 0.0
    is_main: bool = False


class IncidentSearchPipeline:
    """Multi-camera surveillance pipeline: 1 main video + N background CCTV cameras."""

    def __init__(
        self,
        weights_path: str | Path = "outputs/weights/veri_sbs_R50-ibn.pth",
        device: str | None = None,
        camera_graph: CameraTransitionGraph | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = device
        self._extractor: FastReIDExtractor | None = None
        self.camera_graph = camera_graph or CameraTransitionGraph()
        self.tracklets: list[Tracklet] = []
        self.embedding_map: dict[str, np.ndarray] = {}
        self.main_camera_id: str | None = None
        self.background_camera_ids: list[str] = []
        self.annotated_videos: dict[str, str] = {}
        self.raw_video_paths: dict[str, str] = {}
        self.camera_offsets: dict[str, float] = {}
        self.reconstructor = CrossCameraJourneyReconstructor(camera_graph=self.camera_graph)

    @property
    def extractor(self) -> FastReIDExtractor:
        if self._extractor is None:
            self._extractor = FastReIDExtractor(
                weights_path=str(self.weights_path) if self.weights_path.exists() else None,
                device=self.device,
            )
        return self._extractor

    def process_camera_video(
        self,
        camera_id: str,
        video_path: str | Path,
        time_offset_s: float = 0.0,
        is_main: bool = False,
        output_dir: str | Path = "outputs/pipeline",
        max_frames: int | None = None,
        config: dict[str, Any] | None = None,
        save_annotated_video: bool = True,
    ) -> list[Tracklet]:
        """Detect and track vehicles in a video, then compute ReID embeddings."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing camera %s (is_main=%s) from %s", camera_id, is_main, video_path)
        if is_main:
            self.main_camera_id = camera_id
        elif camera_id not in self.background_camera_ids:
            self.background_camera_ids.append(camera_id)

        trks = run_detect_track(
            video_path=video_path,
            camera_id=camera_id,
            config=config or {},
            output_dir=out_dir,
            max_frames=max_frames,
            save_annotated_video=True,
            time_offset_s=time_offset_s,
        )

        annotated_path = out_dir / f"{camera_id}_annotated.mp4"
        if annotated_path.exists():
            self.annotated_videos[camera_id] = str(annotated_path.as_posix())
        self.raw_video_paths[camera_id] = str(Path(video_path).as_posix())
        self.camera_offsets[camera_id] = float(time_offset_s)

        for trk in trks:
            crops = [p for p in trk.crop_paths if Path(p).exists()]
            if not crops:
                continue
            frame_embs = self.extractor.embed_crops(crops)
            emb = aggregate_embeddings(frame_embs, method="mean")
            self.embedding_map[trk.tracklet_id] = emb

        self.tracklets.extend(trks)
        return trks

    def stream_camera_video(
        self,
        camera_id: str,
        video_path: str | Path,
        time_offset_s: float = 0.0,
        is_main: bool = False,
        output_dir: str | Path = "outputs/pipeline",
        max_frames: int | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Stream detection frames live while tracking and embedding vehicles."""
        if is_main:
            self.main_camera_id = camera_id
        elif camera_id not in self.background_camera_ids:
            self.background_camera_ids.append(camera_id)

        gen = stream_detect_track(
            video_path=video_path,
            camera_id=camera_id,
            config=config,
            output_dir=output_dir,
            max_frames=max_frames,
            time_offset_s=time_offset_s,
        )

        trks = None
        try:
            while True:
                item = next(gen)
                yield item
        except StopIteration as e:
            trks = e.value

        if trks:
            out_p = Path(output_dir)
            annotated_path = out_p / f"{camera_id}_annotated.mp4"
            if annotated_path.exists():
                self.annotated_videos[camera_id] = str(annotated_path.as_posix())
            self.raw_video_paths[camera_id] = str(Path(video_path).as_posix())
            self.camera_offsets[camera_id] = float(time_offset_s)

            for trk in trks:
                crops = [p for p in trk.crop_paths if Path(p).exists()]
                if not crops:
                    continue
                frame_embs = self.extractor.embed_crops(crops)
                emb = aggregate_embeddings(frame_embs, method="mean")
                self.embedding_map[trk.tracklet_id] = emb
            self.tracklets.extend(trks)

    def detect_incidents_in_camera(self, camera_id: str, fps: float = 30.0) -> list[Incident]:
        """Detect automated kinematic incidents (sudden stops, stalls) in a specific camera feed."""
        cam_trks = [t for t in self.tracklets if t.camera_id == camera_id]
        return detect_incidents_from_tracklets(cam_trks, fps=fps)

    def trigger_incident_search(
        self,
        query_tracklet: Tracklet,
        incident: Incident | None = None,
        time_window_s: float = 300.0,
        alpha: float = 0.05,
        min_score: float = 0.20,
    ) -> JourneyResult:
        """Given a suspect vehicle from the Main Video, reconstruct its journey across background CCTV cameras."""
        if incident is None:
            incident = create_manual_incident(
                camera_id=query_tracklet.camera_id,
                tracklet=query_tracklet,
                time_s=query_tracklet.start_time_s,
                description=f"Suspect vehicle search for {query_tracklet.tracklet_id}",
            )

        bg_tracklets = [t for t in self.tracklets if t.camera_id != query_tracklet.camera_id]

        return self.reconstructor.reconstruct_journey(
            incident=incident,
            query_tracklet=query_tracklet,
            background_tracklets=bg_tracklets,
            embedding_map=self.embedding_map,
            time_window_s=time_window_s,
            alpha=alpha,
            min_score=min_score,
        )

    def search_vehicle(
        self,
        query_tracklet: Tracklet,
        time_window_s: float | None = 300.0,
        alpha: float = 0.05,
        min_score: float = 0.0,
        top_k: int = 10,
    ) -> SearchResult:
        """Search across all background cameras for a target vehicle."""
        if query_tracklet.tracklet_id not in self.embedding_map:
            raise ValueError(f"No embedding found for query tracklet {query_tracklet.tracklet_id}")

        q_emb = self.embedding_map[query_tracklet.tracklet_id]

        candidates: list[Tracklet] = []
        c_embs: list[np.ndarray] = []

        for trk in self.tracklets:
            if trk.tracklet_id == query_tracklet.tracklet_id:
                continue
            if trk.camera_id == query_tracklet.camera_id:
                continue
            if trk.tracklet_id in self.embedding_map:
                candidates.append(trk)
                c_embs.append(self.embedding_map[trk.tracklet_id])

        if not candidates:
            return SearchResult(
                results=[],
                candidates_before_pruning=0,
                candidates_after_pruning=0,
                mode="appearance_spatiotemporal",
            )

        candidates_before = len(candidates)
        g_mat = np.stack(c_embs)
        q_mat = q_emb[None, :]

        app_scores = (q_mat @ g_mat.T).astype(np.float32)

        eligible_indices = []
        priors = np.zeros((1, len(candidates)), dtype=np.float32)

        for j, cand in enumerate(candidates):
            dt = abs(cand.start_time_s - query_tracklet.start_time_s)
            if time_window_s is not None and dt > time_window_s:
                continue

            prob = self.camera_graph.get_prior(cand.camera_id, query_tracklet.camera_id, dt)
            priors[0, j] = prob
            eligible_indices.append(j)

        if not eligible_indices:
            return SearchResult(
                results=[],
                candidates_before_pruning=candidates_before,
                candidates_after_pruning=0,
                mode="appearance_spatiotemporal",
            )

        fused = compute_fused_scores(app_scores, priors, alpha=alpha)[0]

        items: list[SearchResultItem] = []
        for idx in eligible_indices:
            score = float(fused[idx])
            if score >= min_score:
                cand = candidates[idx]
                items.append(
                    SearchResultItem(
                        tracklet_id=cand.tracklet_id,
                        score=score,
                        appearance_score=float(app_scores[0, idx]),
                        prior_score=float(priors[0, idx]),
                        camera_id=cand.camera_id,
                        start_time_s=cand.start_time_s,
                        end_time_s=cand.end_time_s,
                        crop_paths=cand.crop_paths,
                    )
                )

        items.sort(key=lambda x: x.score, reverse=True)
        items = items[:top_k]

        return SearchResult(
            results=items,
            candidates_before_pruning=candidates_before,
            candidates_after_pruning=len(eligible_indices),
            mode="appearance_spatiotemporal",
        )
