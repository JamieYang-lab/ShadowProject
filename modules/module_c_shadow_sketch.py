from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def _as_binary_mask(mask: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask_array = np.asarray(mask.convert("L"))
    else:
        mask_array = np.asarray(mask)

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    return (mask_array > 0).astype(np.uint8)


def _minimum_mask_area(config: dict | None = None) -> int:
    if not config:
        return 1
    section = config.get("shadow_sketch", config)
    return int(section.get("minimum_mask_area", 1))


def compute_centroid(mask: np.ndarray | Image.Image, minimum_area: int = 1) -> tuple[float, float]:
    binary = _as_binary_mask(mask)
    ys, xs = np.nonzero(binary)
    if len(xs) < minimum_area:
        raise ValueError(f"Mask area is too small: {len(xs)} < {minimum_area}")
    return float(xs.mean()), float(ys.mean())


def estimate_shadow_direction_from_masks(
    object_mask: np.ndarray | Image.Image,
    shadow_mask: np.ndarray | Image.Image,
    minimum_area: int = 1,
) -> list[float]:
    object_centroid = np.asarray(compute_centroid(object_mask, minimum_area), dtype=np.float64)
    shadow_centroid = np.asarray(compute_centroid(shadow_mask, minimum_area), dtype=np.float64)
    direction = shadow_centroid - object_centroid
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("Object and shadow centroids are identical; direction is undefined.")
    return (direction / norm).astype(float).tolist()


def estimate_light_direction_from_masks(
    object_mask: np.ndarray | Image.Image,
    shadow_mask: np.ndarray | Image.Image,
    minimum_area: int = 1,
) -> list[float]:
    shadow_direction = np.asarray(
        estimate_shadow_direction_from_masks(object_mask, shadow_mask, minimum_area),
        dtype=np.float64,
    )
    return (-shadow_direction).astype(float).tolist()


def _resize_mask_to_shape(mask: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    target_height, target_width = int(image_shape[0]), int(image_shape[1])
    if mask.shape[:2] == (target_height, target_width):
        return mask
    return cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def generate_shadow_sketch_from_direction(
    object_mask: np.ndarray | Image.Image,
    light_direction_2d: Iterable[float],
    image_shape: tuple[int, ...],
    config: dict | None = None,
) -> np.ndarray:
    section = config.get("shadow_sketch", config) if config else {}
    minimum_area = _minimum_mask_area(section)
    blur_kernel_size = _odd_kernel_size(section.get("blur_kernel_size", 21))
    projection_length_scale = float(section.get("projection_length_scale", 1.5))
    num_projection_steps = int(section.get("num_projection_steps", 32))

    binary = _as_binary_mask(object_mask)
    if int(binary.sum()) < minimum_area:
        raise ValueError(f"Object mask area is too small: {int(binary.sum())} < {minimum_area}")

    binary = _resize_mask_to_shape(binary, image_shape)
    height, width = binary.shape[:2]

    light_direction = np.asarray(list(light_direction_2d), dtype=np.float64)
    if light_direction.shape[0] != 2:
        raise ValueError("light_direction_2d must contain exactly 2 values.")

    norm = float(np.linalg.norm(light_direction))
    if norm == 0.0:
        raise ValueError("light_direction_2d cannot be zero.")

    shadow_direction = -light_direction / norm
    ys, xs = np.nonzero(binary)
    object_extent = max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
    max_distance = max(1.0, object_extent * projection_length_scale)

    sketch = np.zeros((height, width), dtype=np.float32)
    for step in range(1, max(2, num_projection_steps) + 1):
        alpha = step / float(max(2, num_projection_steps))
        shift = np.rint(shadow_direction * max_distance * alpha).astype(np.int32)
        transform = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
        shifted = cv2.warpAffine(binary, transform, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        sketch = np.maximum(sketch, shifted.astype(np.float32) * (1.0 - 0.65 * alpha))

    sketch[binary > 0] = 0.0
    if blur_kernel_size > 1:
        sketch = cv2.GaussianBlur(sketch, (blur_kernel_size, blur_kernel_size), 0)

    if float(sketch.max()) > 0.0:
        sketch = sketch / float(sketch.max())

    return np.clip(np.rint(sketch * 255.0), 0, 255).astype(np.uint8)


def save_shadow_sketch(sketch: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(sketch, dtype=np.uint8)).save(output)
    return output
