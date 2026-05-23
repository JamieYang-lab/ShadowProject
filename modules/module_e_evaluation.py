from __future__ import annotations

import numpy as np

from modules.module_c_shadow_sketch import compute_centroid


def _binary(mask) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def mask_iou(mask_a, mask_b) -> float:
    a = _binary(mask_a)
    b = _binary(mask_b)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def direction_angle_error_deg(dir_a, dir_b) -> float:
    a = np.asarray(dir_a, dtype=np.float64)
    b = np.asarray(dir_b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Direction vectors must be non-zero.")
    cosine = float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def shadow_centroid(mask) -> tuple[float, float]:
    return compute_centroid(mask)
