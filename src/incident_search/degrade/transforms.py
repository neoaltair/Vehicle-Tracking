"""Controlled deterministic query image degradation transforms for vehicle search robustness study.

Degradation families:
1. Downscale + upsample: factors {1.0, 0.5, 0.25, 0.125}
2. Gaussian blur: sigma levels {0.0, 1.0, 2.0, 4.0}
3. Crop (central area retention): fractions {1.0, 0.75, 0.50, 0.25}
4. Occlusion (random rectangular mask): area fractions {0.0, 0.25, 0.50}

Transformations operate deterministically using a numpy random generator seed.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_downscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Downscale image by factor and upsample back to original dimensions using bilinear interpolation."""
    if factor >= 1.0:
        return image.copy()
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return upscaled


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with specified standard deviation sigma."""
    if sigma <= 0.0:
        return image.copy()
    # Kernel size must be positive and odd
    ksize = int(round(sigma * 4)) | 1
    ksize = max(3, ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Crop central portion of the image preserving keep_ratio area, resize back to original size."""
    if keep_ratio >= 1.0:
        return image.copy()
    h, w = image.shape[:2]
    crop_h = max(1, int(round(h * np.sqrt(keep_ratio))))
    crop_w = max(1, int(round(w * np.sqrt(keep_ratio))))
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_occlusion(image: np.ndarray, occlusion_ratio: float, seed: int = 0) -> np.ndarray:
    """Occlude a random rectangular patch of the image covering occlusion_ratio of total area with gray."""
    if occlusion_ratio <= 0.0:
        return image.copy()
    h, w = image.shape[:2]
    total_area = h * w
    occ_area = total_area * min(1.0, occlusion_ratio)

    rng = np.random.default_rng(seed)
    # Random aspect ratio between 0.5 and 2.0
    aspect = float(rng.uniform(0.5, 2.0))
    occ_h = int(np.clip(round(np.sqrt(occ_area / aspect)), 1, h))
    occ_w = int(np.clip(round(occ_h * aspect), 1, w))

    y1 = int(rng.integers(0, max(1, h - occ_h + 1)))
    x1 = int(rng.integers(0, max(1, w - occ_w + 1)))

    out = image.copy()
    # Gray fill (128)
    out[y1 : y1 + occ_h, x1 : x1 + occ_w] = 128
    return out


def degrade_crop(
    image: np.ndarray,
    family: str,
    severity: float,
    seed: int = 0,
) -> np.ndarray:
    """Apply specified degradation family and severity level to a single vehicle crop."""
    if family == "clean" or severity == 0.0 or (family == "downscale" and severity >= 1.0) or (family == "crop" and severity >= 1.0):
        return image.copy()
    if family == "downscale":
        return apply_downscale(image, factor=severity)
    elif family == "blur":
        return apply_gaussian_blur(image, sigma=severity)
    elif family == "crop":
        return apply_crop(image, keep_ratio=severity)
    elif family == "occlusion":
        return apply_occlusion(image, occlusion_ratio=severity, seed=seed)
    else:
        raise ValueError(f"Unknown degradation family: {family}")
