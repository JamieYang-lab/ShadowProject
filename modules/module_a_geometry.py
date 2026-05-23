from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"
MOGE_INSTALL_MESSAGE = (
    "MoGe/MoGe-2 is not available. Install it with "
    "`pip install git+https://github.com/microsoft/MoGe.git`, then set "
    "`geometry.model_name_or_path` in configs/geometry_config.yaml to a Hugging Face "
    "model id such as `Ruicheng/moge-2-vitl-normal` or to a local checkpoint path. "
    "Do not commit downloaded model weights to this repository."
)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_point_map(path: str | Path, point_map: np.ndarray) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    array = np.asarray(point_map, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("point_map must have shape [H, W, 3].")
    np.save(output_path, array)
    return output_path


def load_point_map(path: str | Path) -> np.ndarray:
    array = np.load(Path(path))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Loaded point map must have shape [H, W, 3].")
    return array


def normalize_depth_for_png(depth: np.ndarray) -> np.ndarray:
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("depth must have shape [H, W].")

    finite_mask = np.isfinite(depth_array)
    output = np.zeros(depth_array.shape, dtype=np.uint8)
    if not finite_mask.any():
        return output

    valid = depth_array[finite_mask]
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        output[finite_mask] = 255
        return output

    normalized = (depth_array[finite_mask] - min_value) / (max_value - min_value)
    output[finite_mask] = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    return output


def save_depth_png(path: str | Path, depth: np.ndarray) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    Image.fromarray(normalize_depth_for_png(depth)).save(output_path)
    return output_path


def normalize_normal_for_png(normal: np.ndarray) -> np.ndarray:
    normal_array = np.asarray(normal, dtype=np.float32)
    if normal_array.ndim != 3 or normal_array.shape[-1] != 3:
        raise ValueError("normal must have shape [H, W, 3].")
    return np.clip(np.rint((normal_array + 1.0) * 127.5), 0, 255).astype(np.uint8)


def save_normal_png(path: str | Path, normal: np.ndarray) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    Image.fromarray(normalize_normal_for_png(normal)).save(output_path)
    return output_path


def _geometry_section(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("geometry", config)


def _resize_image_if_needed(image: Image.Image, resize_max_side: int | None) -> Image.Image:
    if resize_max_side is None:
        return image
    max_side = int(resize_max_side)
    if max_side <= 0:
        raise ValueError("resize_max_side must be positive or null.")

    width, height = image.size
    current_max_side = max(width, height)
    if current_max_side <= max_side:
        return image

    scale = max_side / float(current_max_side)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _extract_output(output: dict, *names: str):
    for name in names:
        if name in output and output[name] is not None:
            return _tensor_to_numpy(output[name])
    return None


def run_moge_on_image(image_path: str | Path, config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    geometry_config = _geometry_section(config)
    model_name_or_path = geometry_config.get("model_name_or_path") or DEFAULT_MOGE_MODEL
    device_name = geometry_config.get("device", "cuda")
    resize_max_side = geometry_config.get("resize_max_side")

    try:
        import torch
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise ImportError(MOGE_INSTALL_MESSAGE) from exc

    try:
        device = torch.device(device_name)
        model = MoGeModel.from_pretrained(model_name_or_path).to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MoGe model '{model_name_or_path}' on device '{device_name}'. "
            "Check that the model package is installed, the model name/path is valid, "
            "and weights are available outside the git repository."
        ) from exc

    with Image.open(image_path) as image:
        rgb_image = _resize_image_if_needed(image.convert("RGB"), resize_max_side)
    image_array = np.asarray(rgb_image, dtype=np.float32) / 255.0

    input_tensor = torch.tensor(image_array, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        output = model.infer(input_tensor)

    point_map = _extract_output(output, "points", "point_map")
    if point_map is None:
        raise RuntimeError("MoGe inference output did not contain 'points' or 'point_map'.")

    result: dict[str, np.ndarray] = {"point_map": np.asarray(point_map, dtype=np.float32)}
    depth = _extract_output(output, "depth")
    normal = _extract_output(output, "normal", "normals")
    if depth is not None:
        result["depth"] = np.asarray(depth, dtype=np.float32)
    if normal is not None:
        result["normal"] = np.asarray(normal, dtype=np.float32)

    return result
