from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


SAM_INSTALL_MESSAGE = (
    "SAM/SAM2 is not available or its checkpoint is missing. For SAM2, install the "
    "official package from https://github.com/facebookresearch/sam2 and set "
    "`segmentation.checkpoint_path` plus, when needed, `segmentation.model_cfg` in "
    "configs/segmentation_config.yaml. For SAM v1, install "
    "`git+https://github.com/facebookresearch/segment-anything.git` and set "
    "`segmentation.model_type` to a SAM registry key such as `vit_h`, `vit_l`, or "
    "`vit_b` with a local checkpoint path. Do not commit model weights to this repo."
)


def _as_binary_mask(mask) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def encode_binary_mask(mask) -> np.ndarray:
    return (_as_binary_mask(mask).astype(np.uint8) * 255)


def save_mask_png(path: str | Path, mask) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encode_binary_mask(mask)).save(output_path)
    return output_path


def load_mask_png(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _mask_bbox(mask) -> list[int]:
    binary = _as_binary_mask(mask)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def normalize_mask_record(record: dict) -> dict:
    mask = _as_binary_mask(record.get("mask", record.get("segmentation")))
    area = int(record.get("area", int(mask.sum())))
    bbox = record.get("bbox") or _mask_bbox(mask)
    score = record.get("score", record.get("predicted_iou", record.get("stability_score")))
    return {
        "mask": mask,
        "area": area,
        "bbox": [int(value) for value in bbox],
        "score": None if score is None else float(score),
    }


def filter_masks(masks: Iterable[dict], min_mask_area: int = 0, max_masks: int | None = None) -> list[dict]:
    normalized = [normalize_mask_record(mask) for mask in masks]
    filtered = [mask for mask in normalized if int(mask["area"]) >= int(min_mask_area)]
    filtered.sort(key=lambda item: (item["score"] is not None, item["score"] or 0.0, item["area"]), reverse=True)
    if max_masks is not None:
        filtered = filtered[: max(0, int(max_masks))]
    return filtered


def _color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
        (131, 56, 236),
        (255, 190, 11),
        (0, 150, 199),
    ]
    return palette[index % len(palette)]


def save_mask_overlay(image_path: str | Path, masks: Iterable[dict], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        base = image.convert("RGBA")

    for index, record in enumerate(masks):
        normalized = normalize_mask_record(record)
        mask = normalized["mask"]
        if mask.shape[:2] != (base.height, base.width):
            mask_image = Image.fromarray(encode_binary_mask(mask)).resize(base.size, Image.Resampling.NEAREST)
            mask = np.asarray(mask_image) > 0

        color = _color_for_index(index)
        overlay = Image.new("RGBA", base.size, color + (0,))
        alpha = np.zeros(mask.shape, dtype=np.uint8)
        alpha[mask] = 105
        overlay.putalpha(Image.fromarray(alpha))
        base = Image.alpha_composite(base, overlay)

        draw = ImageDraw.Draw(base)
        x, y, w, h = normalized["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=color + (255,), width=2)

    base.convert("RGB").save(output)
    return output


def _segmentation_section(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("segmentation", config)


def _load_image_array(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def _run_sam2(image: np.ndarray, config: dict[str, Any]) -> list[dict]:
    checkpoint_path = config.get("checkpoint_path")
    model_cfg = config.get("model_cfg")
    device = config.get("device", "cuda")

    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise ImportError(SAM_INSTALL_MESSAGE) from exc

    try:
        if checkpoint_path:
            if not model_cfg:
                raise ValueError("SAM2 checkpoint mode requires `segmentation.model_cfg` in segmentation_config.yaml.")
            model = build_sam2(model_cfg, checkpoint_path, device=device)
            generator = SAM2AutomaticMaskGenerator(model, min_mask_region_area=int(config.get("min_mask_area", 100)))
        elif hasattr(SAM2AutomaticMaskGenerator, "from_pretrained"):
            model_id = config.get("model_name_or_path", "facebook/sam2-hiera-large")
            generator = SAM2AutomaticMaskGenerator.from_pretrained(
                model_id,
                min_mask_region_area=int(config.get("min_mask_area", 100)),
                output_mode="binary_mask",
            )
        else:
            raise ValueError("SAM2 requires checkpoint_path/model_cfg or a SAM2AutomaticMaskGenerator.from_pretrained API.")

        with torch.inference_mode():
            return generator.generate(image)
    except Exception as exc:
        raise RuntimeError(f"Failed to run SAM2 inference. {SAM_INSTALL_MESSAGE}") from exc


def _run_sam_v1(image: np.ndarray, config: dict[str, Any]) -> list[dict]:
    checkpoint_path = config.get("checkpoint_path")
    model_type = config.get("model_type", "vit_h")
    device = config.get("device", "cuda")
    if not checkpoint_path:
        raise ImportError(SAM_INSTALL_MESSAGE)

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(SAM_INSTALL_MESSAGE) from exc

    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=int(config.get("min_mask_area", 100)))
        return generator.generate(image)
    except Exception as exc:
        raise RuntimeError(f"Failed to run SAM inference. {SAM_INSTALL_MESSAGE}") from exc


def run_sam_on_image(image_path: str | Path, config: dict[str, Any] | None = None) -> list[dict]:
    segmentation_config = _segmentation_section(config)
    mode = segmentation_config.get("mode", "automatic")
    if mode != "automatic":
        raise NotImplementedError("Only automatic SAM/SAM2 mask generation is supported in scaffold v2.2.")

    image = _load_image_array(image_path)
    model_type = str(segmentation_config.get("model_type", "sam2")).lower()

    if model_type == "sam2" or model_type.startswith("sam2"):
        raw_masks = _run_sam2(image, segmentation_config)
    else:
        raw_masks = _run_sam_v1(image, segmentation_config)

    return filter_masks(
        raw_masks,
        min_mask_area=int(segmentation_config.get("min_mask_area", 100)),
        max_masks=int(segmentation_config.get("max_masks_per_image", 50)),
    )
