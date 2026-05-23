from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _config_section(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("receiver_selection", config)


def as_binary_mask(mask) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def resize_binary_mask(mask, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(shape[0]), int(shape[1])
    binary = as_binary_mask(mask).astype(np.uint8)
    if binary.shape[:2] == (target_h, target_w):
        return binary.astype(bool)
    resized = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def validate_object_mask(object_mask, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = _config_section(config)
    binary = as_binary_mask(object_mask)
    min_area = int(cfg.get("min_object_area", 4))
    if int(binary.sum()) < min_area:
        raise ValueError(f"Object mask area is too small: {int(binary.sum())} < {min_area}")
    return binary


def lower_image_mask(shape: tuple[int, int], lower_y_ratio: float = 0.35) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Mask shape must have positive height and width.")
    start_y = int(round(float(lower_y_ratio) * height))
    mask = np.zeros((height, width), dtype=bool)
    mask[min(max(start_y, 0), height) :, :] = True
    return mask


def validate_point_map(point_map) -> np.ndarray:
    array = np.asarray(point_map, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("point_map must have shape [H, W, 3].")
    return array


def resize_point_map_to_shape(point_map, shape: tuple[int, int]) -> np.ndarray:
    array = validate_point_map(point_map)
    target_h, target_w = int(shape[0]), int(shape[1])
    if array.shape[:2] == (target_h, target_w):
        return array
    return cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def estimate_plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 3:
        raise ValueError("At least 3 finite points are required to estimate a plane.")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("Estimated plane normal has zero length.")
    normal = normal / norm
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), offset


def plane_distance_mask(point_map, base_mask, config: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = _config_section(config)
    point_map = resize_point_map_to_shape(point_map, as_binary_mask(base_mask).shape)
    base = as_binary_mask(base_mask)
    finite = np.isfinite(point_map).all(axis=-1)
    sample_points = point_map[base & finite]
    normal, offset = estimate_plane_from_points(sample_points.reshape(-1, 3))
    distances = np.abs(np.tensordot(point_map, normal, axes=([-1], [0])) + offset)
    threshold = float(cfg.get("plane_distance_threshold", 0.08))
    return (distances <= threshold) & finite, {"normal": normal.tolist(), "offset": float(offset), "threshold": threshold}


def load_sam_masks(sam_mask_dir: str | Path | None, shape: tuple[int, int]) -> list[np.ndarray]:
    if sam_mask_dir is None:
        return []
    directory = Path(sam_mask_dir)
    if not directory.exists():
        return []
    masks: list[np.ndarray] = []
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as image:
            masks.append(resize_binary_mask(np.asarray(image.convert("L")), shape))
    return masks


def select_sam_receiver_masks(
    sam_masks: list[np.ndarray],
    object_mask,
    lower_mask,
    plane_mask=None,
    config: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    cfg = _config_section(config)
    object_binary = as_binary_mask(object_mask)
    lower_binary = as_binary_mask(lower_mask)
    plane_binary = as_binary_mask(plane_mask) if plane_mask is not None else None
    selected: list[np.ndarray] = []

    min_area = int(cfg.get("sam_min_area", 100))
    lower_overlap_min = float(cfg.get("sam_lower_overlap_min", 0.25))
    object_overlap_max = float(cfg.get("sam_object_overlap_max", 0.2))
    plane_overlap_min = float(cfg.get("sam_plane_overlap_min", 0.25))

    for mask in sam_masks:
        binary = resize_binary_mask(mask, object_binary.shape)
        area = int(binary.sum())
        if area < min_area:
            continue
        lower_overlap = float((binary & lower_binary).sum() / max(area, 1))
        object_overlap = float((binary & object_binary).sum() / max(area, 1))
        if lower_overlap < lower_overlap_min:
            continue
        if object_overlap > object_overlap_max:
            continue
        if plane_binary is not None:
            plane_overlap = float((binary & plane_binary).sum() / max(area, 1))
            if plane_overlap < plane_overlap_min:
                continue
        selected.append(binary)
    return selected


def combine_receiver_masks(base_mask, selected_sam_masks: list[np.ndarray], config: dict[str, Any] | None = None) -> np.ndarray:
    combined = as_binary_mask(base_mask).copy()
    if selected_sam_masks:
        sam_union = np.zeros_like(combined, dtype=bool)
        for mask in selected_sam_masks:
            sam_union |= resize_binary_mask(mask, combined.shape)
        mode = _config_section(config).get("combine_sam_with_plane", "union")
        if mode == "intersection":
            combined &= sam_union
        else:
            combined |= sam_union
    return combined


def compute_receiver_mask(
    object_mask,
    image_shape: tuple[int, int],
    point_map=None,
    sam_masks: list[np.ndarray] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config_section(config)
    object_binary = resize_binary_mask(validate_object_mask(object_mask, cfg), image_shape)
    lower = lower_image_mask(image_shape, float(cfg.get("lower_y_ratio", 0.35)))
    non_object = ~object_binary
    base = non_object & lower
    plane_info = None

    plane_mask = None
    if point_map is not None:
        try:
            point_map_resized = resize_point_map_to_shape(point_map, image_shape)
            sample_lower = lower_image_mask(image_shape, float(cfg.get("plane_sample_lower_ratio", 0.5)))
            plane_mask, plane_info = plane_distance_mask(point_map_resized, sample_lower & non_object, cfg)
            base = base & plane_mask
        except ValueError:
            plane_mask = None
            plane_info = None

    selected_sam = select_sam_receiver_masks(sam_masks or [], object_binary, lower, plane_mask, cfg)
    receiver = combine_receiver_masks(base, selected_sam, cfg)
    receiver &= non_object
    return {
        "receiver_mask": receiver.astype(bool),
        "lower_mask": lower.astype(bool),
        "plane_mask": None if plane_mask is None else plane_mask.astype(bool),
        "selected_sam_masks": selected_sam,
        "plane": plane_info,
    }


def save_receiver_mask_png(path: str | Path, receiver_mask) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(as_binary_mask(receiver_mask).astype(np.uint8) * 255).save(output)
    return output
