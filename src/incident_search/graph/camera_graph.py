"""Camera transition graph and spatio-temporal travel-time prior modeling.

Prevents leakage by strictly fitting transitions only on training identities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy.stats import lognorm

from incident_search.eval.protocol import _tracklet_identity
from incident_search.io.schema import Tracklet

logger = logging.getLogger(__name__)


@dataclass
class EdgeTravelTimeModel:
    """Estimated travel-time distribution for a directed camera pair (u -> v)."""

    src_camera: str
    dst_camera: str
    sample_count: int
    mean_dt: float
    std_dt: float
    min_dt: float
    max_dt: float
    fitted_shape: float | None = None  # log-normal s (sigma)
    fitted_scale: float | None = None  # log-normal exp(mu)

    def pdf(self, dt: float) -> float:
        """Evaluate travel-time probability density function at dt seconds."""
        if dt <= 0:
            return 0.0
        if self.fitted_shape is not None and self.fitted_scale is not None:
            # lognorm in scipy: s=shape, scale=scale, loc=0
            return float(lognorm.pdf(dt, s=self.fitted_shape, scale=self.fitted_scale))
        # Fallback to Gaussian or uniform window if lognorm failed or 1 sample
        if self.std_dt > 1e-4:
            variance = self.std_dt ** 2
            return float(np.exp(-0.5 * ((dt - self.mean_dt) ** 2) / variance) / (np.sqrt(2 * np.pi) * self.std_dt))
        return 1.0 if abs(dt - self.mean_dt) <= 5.0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_camera": self.src_camera,
            "dst_camera": self.dst_camera,
            "sample_count": self.sample_count,
            "mean_dt": round(self.mean_dt, 3),
            "std_dt": round(self.std_dt, 3),
            "min_dt": round(self.min_dt, 3),
            "max_dt": round(self.max_dt, 3),
            "fitted_shape": round(self.fitted_shape, 4) if self.fitted_shape is not None else None,
            "fitted_scale": round(self.fitted_scale, 4) if self.fitted_scale is not None else None,
        }


@dataclass
class CameraTransitionGraph:
    """Directed camera network graph with empirical travel-time statistics."""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    edge_models: dict[tuple[str, str], EdgeTravelTimeModel] = field(default_factory=dict)
    min_edge_samples: int = 5
    max_travel_time_s: float = 3600.0  # Ignore transitions taking longer than 1 hour

    def add_transition(self, src_cam: str, dst_cam: str, delta_t: float) -> None:
        """Add an observed transition duration delta_t from src_cam to dst_cam."""
        if delta_t <= 0 or delta_t > self.max_travel_time_s:
            return
        if not self.graph.has_node(src_cam):
            self.graph.add_node(src_cam)
        if not self.graph.has_node(dst_cam):
            self.graph.add_node(dst_cam)

        if not self.graph.has_edge(src_cam, dst_cam):
            self.graph.add_edge(src_cam, dst_cam, samples=[])

        self.graph[src_cam][dst_cam]["samples"].append(delta_t)

    def fit_models(self) -> None:
        """Fit travel-time distribution models for all edges with sufficient samples."""
        self.edge_models.clear()
        for u, v, data in self.graph.edges(data=True):
            samples = data.get("samples", [])
            if len(samples) < self.min_edge_samples:
                continue

            arr = np.asarray(samples, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))

            fitted_shape = None
            fitted_scale = None

            # Fit log-normal distribution if positive and variance exists
            if len(arr) >= 3 and np.all(arr > 0) and std_val > 1e-4:
                try:
                    # Fix loc=0 for standard 2-parameter log-normal
                    shape, loc, scale = lognorm.fit(arr, floc=0)
                    fitted_shape = float(shape)
                    fitted_scale = float(scale)
                except Exception as e:
                    logger.debug("Failed to fit lognormal on edge %s->%s: %s", u, v, e)

            model = EdgeTravelTimeModel(
                src_camera=u,
                dst_camera=v,
                sample_count=len(arr),
                mean_dt=mean_val,
                std_dt=std_val,
                min_dt=min_val,
                max_dt=max_val,
                fitted_shape=fitted_shape,
                fitted_scale=fitted_scale,
            )
            self.edge_models[(u, v)] = model

    def has_edge(self, u: str, v: str) -> bool:
        """Check if a modeled transition edge exists between u and v."""
        return (u, v) in self.edge_models

    def get_prior(self, src_cam: str, dst_cam: str, delta_t: float) -> float:
        """Return transition prior density / likelihood for candidate transition src_cam -> dst_cam in delta_t seconds."""
        model = self.edge_models.get((src_cam, dst_cam))
        if model is None:
            return 0.0
        return model.pdf(delta_t)


def extract_transitions_from_tracklets(
    tracklets: Iterable[Tracklet],
    max_transition_time_s: float = 3600.0,
) -> list[tuple[str, str, float, str]]:
    """Extract consecutive camera-to-camera transitions for each vehicle identity.

    Returns:
        List of tuples: (src_camera, dst_camera, delta_t, vehicle_identity)
    """
    by_identity: dict[str, list[Tracklet]] = {}
    for t in tracklets:
        identity = _tracklet_identity(t)
        by_identity.setdefault(identity, []).append(t)

    transitions: list[tuple[str, str, float, str]] = []

    for identity, trk_list in by_identity.items():
        # Sort chronologically by start_time_s
        sorted_trks = sorted(trk_list, key=lambda x: x.start_time_s)
        for i in range(len(sorted_trks) - 1):
            curr_trk = sorted_trks[i]
            next_trk = sorted_trks[i + 1]

            if curr_trk.camera_id == next_trk.camera_id:
                continue

            # Travel time: from departure at curr_trk to arrival at next_trk
            # (or start-to-start / end-to-start)
            # Typically delta_t = next_trk.start_time_s - curr_trk.end_time_s
            dt = next_trk.start_time_s - curr_trk.end_time_s
            if dt <= 0:
                # If slight overlap or instant transition, fall back to start-to-start
                dt = next_trk.start_time_s - curr_trk.start_time_s

            if 0 < dt <= max_transition_time_s:
                transitions.append((curr_trk.camera_id, next_trk.camera_id, float(dt), identity))

    return transitions


def build_camera_graph_from_training_tracklets(
    train_tracklets: list[Tracklet],
    min_edge_samples: int = 5,
    max_transition_time_s: float = 3600.0,
) -> CameraTransitionGraph:
    """Build and fit a CameraTransitionGraph strictly from training tracklets.

    Args:
        train_tracklets: Tracklets from the training split ONLY.
        min_edge_samples: Minimum observed transitions to model an edge.
        max_transition_time_s: Max travel time considered reasonable.

    Returns:
        Fitted CameraTransitionGraph instance.
    """
    graph = CameraTransitionGraph(
        min_edge_samples=min_edge_samples,
        max_travel_time_s=max_transition_time_s,
    )
    transitions = extract_transitions_from_tracklets(
        train_tracklets,
        max_transition_time_s=max_transition_time_s,
    )
    for src_cam, dst_cam, dt, _ in transitions:
        graph.add_transition(src_cam, dst_cam, dt)

    graph.fit_models()
    return graph
