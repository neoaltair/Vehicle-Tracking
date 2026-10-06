"""Vehicle Re-Identification (Re-ID) embedding extraction, aggregation, and caching.

Uses FastReID SBS(R50-ibn) pretrained on the VeRi-776 benchmark.
Extracts L2-normalized 2048-dimensional float32 feature vectors.
Supports exactly two aggregation methods: 'single' and 'mean'.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_URL = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth"
)
DEFAULT_WEIGHTS_PATH = "outputs/weights/veri_sbs_R50-ibn.pth"


def l2_normalize(features: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize numpy feature array along specified axis.

    Args:
        features: Input feature array.
        axis: Axis along which to compute L2 norm.
        eps: Small epsilon to avoid division by zero.

    Returns:
        L2-normalized float32 numpy array.
    """
    features = features.astype(np.float32)
    norm = np.linalg.norm(features, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return features / norm


def aggregate_embeddings(
    frame_embeddings: np.ndarray,
    method: str = "mean",
    best_idx: int = 0,
) -> np.ndarray:
    """Aggregate per-frame embeddings for a tracklet into a single tracklet embedding.

    Strictly supports exactly two methods:
    1. 'single': selects one best frame embedding (at best_idx) and ensures L2 unit norm.
    2. 'mean': averages L2-normalized frame embeddings and re-normalizes to unit norm.

    Args:
        frame_embeddings: Array of shape (N, D) containing per-frame feature vectors.
        method: Aggregation method ('single' or 'mean').
        best_idx: Index of best frame to use when method='single'.

    Returns:
        Aggregated feature vector of shape (D,), L2-normalized.
    """
    if method not in ("single", "mean"):
        raise ValueError(
            f"Invalid aggregation method '{method}'. Must be strictly 'single' or 'mean'."
        )

    if frame_embeddings.ndim == 1:
        return l2_normalize(frame_embeddings, axis=-1)

    if len(frame_embeddings) == 0:
        raise ValueError("Cannot aggregate empty frame embeddings array.")

    # Ensure all input frame embeddings are L2 normalized
    normed_frames = l2_normalize(frame_embeddings, axis=-1)

    if method == "single":
        idx = max(0, min(len(normed_frames) - 1, best_idx))
        return normed_frames[idx]

    # method == 'mean'
    mean_vec = np.mean(normed_frames, axis=0)
    return l2_normalize(mean_vec, axis=-1)


class FastReIDExtractor:
    """Vehicle Re-ID feature extractor using FastReID SBS(R50-ibn) on VeRi-776."""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        weights_url: str = DEFAULT_WEIGHTS_URL,
        device: str = "cpu",
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        """Initialize the Re-ID extractor.

        Args:
            weights_path: Local path to saved model checkpoint.
            weights_url: URL to download weights if missing.
            device: 'cuda' or 'cpu'.
            image_size: Input (height, width) for model forward pass.
        """
        self.weights_path = Path(weights_path)
        self.weights_url = weights_url
        self.device = torch.device(device if (torch.cuda.is_available() and device == "cuda") else "cpu")
        self.image_size = image_size
        self.model_name = "sbs_R50-ibn"
        self.model_version = "veri_sbs_R50-ibn_v0.1.1"

        self._ensure_weights()
        self.model = self._build_model()

    def _ensure_weights(self) -> None:
        """Ensure model weights exist locally, downloading if necessary."""
        if not self.weights_path.exists():
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading VeRi-776 pretrained weights from %s ...", self.weights_url)
            urllib.request.urlretrieve(self.weights_url, str(self.weights_path))
            logger.info("Downloaded weights to %s (%d bytes)", self.weights_path, self.weights_path.stat().st_size)

    def _build_model(self) -> torch.nn.Module:
        """Construct FastReID Baseline model with ResNet50-IBN backbone and GeM pooling."""
        cfg = get_cfg()
        cfg.MODEL.META_ARCHITECTURE = "Baseline"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.BACKBONE.DEPTH = "50x"
        cfg.MODEL.BACKBONE.WITH_IBN = True
        cfg.MODEL.BACKBONE.WITH_NL = True
        cfg.MODEL.HEADS.POOL_LAYER = "GeneralizedMeanPoolingP"
        cfg.MODEL.HEADS.NECK_FEAT = "after"
        cfg.MODEL.HEADS.NUM_CLASSES = 575
        cfg.MODEL.DEVICE = str(self.device)

        model = build_model(cfg)
        Checkpointer(model).load(str(self.weights_path))
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess an image (BGR numpy array) to model input tensor.

        Args:
            img: BGR image array (H, W, 3).

        Returns:
            Preprocessed float32 tensor of shape (3, H, W).
        """
        # Resize to test size (256, 256)
        h, w = self.image_size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        # PyTorch format: (3, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)

        # FastReID Baseline internal normalization expects raw [0, 255] RGB values
        # as pixel_mean and pixel_std inside the model handle normalization
        return tensor

    @torch.no_grad()
    def embed_crops(self, crops: list[np.ndarray | str | Path]) -> np.ndarray:
        """Extract L2-normalized re-ID feature vectors for a list of crops.

        Args:
            crops: List of BGR numpy image arrays or file paths.

        Returns:
            Float32 numpy array of shape (N, 2048), L2-normalized.
        """
        if not crops:
            return np.empty((0, 2048), dtype=np.float32)

        tensors = []
        for crop in crops:
            if isinstance(crop, (str, Path)):
                img = cv2.imread(str(crop))
                if img is None:
                    raise FileNotFoundError(f"Could not read crop image: {crop}")
            else:
                img = crop

            tensors.append(self.preprocess_image(img))

        batch_tensor = torch.stack(tensors).to(self.device)
        features = self.model({"images": batch_tensor})

        # Convert to numpy and L2 normalize
        feats_np = features.cpu().numpy().astype(np.float32)
        return l2_normalize(feats_np, axis=-1)


def save_embeddings(
    cache_dir: str | Path,
    tracklet_id: str,
    frame_embeddings: np.ndarray,
    tracklet_embedding: np.ndarray,
    model_name: str = "sbs_R50-ibn",
) -> Path:
    """Save frame and tracklet embeddings to disk as a compressed .npz archive.

    Args:
        cache_dir: Base directory for embedding cache.
        tracklet_id: Unique tracklet identifier.
        frame_embeddings: Array of shape (N, D).
        tracklet_embedding: Aggregated array of shape (D,).
        model_name: Subfolder for model identifier.

    Returns:
        Path to saved .npz file.
    """
    out_dir = Path(cache_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tracklet_id}.npz"

    np.savez_compressed(
        out_path,
        frame_embeddings=frame_embeddings.astype(np.float32),
        tracklet_embedding=tracklet_embedding.astype(np.float32),
    )
    return out_path


def load_embeddings(
    cache_dir: str | Path,
    tracklet_id: str,
    model_name: str = "sbs_R50-ibn",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load cached frame and tracklet embeddings from disk.

    Args:
        cache_dir: Base directory for embedding cache.
        tracklet_id: Tracklet identifier.
        model_name: Model identifier subfolder.

    Returns:
        Tuple of (frame_embeddings, tracklet_embedding) or None if not found.
    """
    target = Path(cache_dir) / model_name / f"{tracklet_id}.npz"
    if not target.exists():
        return None

    with np.load(target) as data:
        frame_feats = data["frame_embeddings"]
        trk_feat = data["tracklet_embedding"]
    return frame_feats, trk_feat
