from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_a_receiver_selection import compute_receiver_mask, load_sam_masks


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "receiver_config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "visualizations" / "receiver_masks"
PANEL_TITLE_HEIGHT = 24
PANEL_GAP = 8


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 0


def _overlay(base: Image.Image, mask, color: tuple[int, int, int], alpha: int = 120) -> Image.Image:
    output = base.convert("RGBA")
    binary = np.asarray(mask) > 0
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha_mask = np.zeros(binary.shape, dtype=np.uint8)
    alpha_mask[binary] = alpha
    overlay.putalpha(Image.fromarray(alpha_mask))
    return Image.alpha_composite(output, overlay).convert("RGB")


def _multi_overlay(base: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    colors = [(230, 57, 70), (42, 157, 143), (69, 123, 157), (244, 162, 97), (131, 56, 236)]
    output = base
    for idx, mask in enumerate(masks):
        output = _overlay(output, mask, colors[idx % len(colors)], alpha=85)
    return output


def _heatmap(array: np.ndarray | None, size: tuple[int, int]) -> Image.Image:
    if array is None:
        return Image.new("RGB", size, (32, 32, 32))
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if float(values.max()) > float(values.min()):
        values = (values - float(values.min())) / (float(values.max()) - float(values.min()))
    else:
        values = np.zeros(values.shape, dtype=np.float32)
    gray = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)).resize(size, Image.Resampling.BILINEAR)


def _load_prior(path: str | Path | None) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _add_title(panel: Image.Image, title: str) -> Image.Image:
    output = Image.new("RGB", (panel.width, panel.height + PANEL_TITLE_HEIGHT), (18, 20, 24))
    output.paste(panel, (0, PANEL_TITLE_HEIGHT))
    draw = ImageDraw.Draw(output)
    draw.text((8, 6), title, fill=(235, 235, 235), font=ImageFont.load_default())
    return output


def _board(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    titled = [_add_title(panel, title) for title, panel in panels]
    width = sum(panel.width for panel in titled) + PANEL_GAP * (len(titled) - 1)
    height = max(panel.height for panel in titled)
    output = Image.new("RGB", (width, height), (18, 20, 24))
    x = 0
    for panel in titled:
        output.paste(panel, (x, 0))
        x += panel.width + PANEL_GAP
    return output


def visualize_receiver_mask(
    sample_id: str,
    composite_path: str | Path,
    object_mask_path: str | Path,
    point_map_path: str | Path | None = None,
    sam_mask_dir: str | Path | None = None,
    receiver_mask_path: str | Path | None = None,
    sg_prior_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    config = _load_config(config_path)
    composite = _load_rgb(composite_path)
    size = composite.size
    image_shape = (composite.height, composite.width)
    object_mask = _load_mask(object_mask_path, size)
    point_map = np.load(point_map_path) if point_map_path is not None and Path(point_map_path).exists() else None
    sam_masks = load_sam_masks(sam_mask_dir, image_shape)
    result = compute_receiver_mask(object_mask, image_shape, point_map=point_map, sam_masks=sam_masks, config=config)
    receiver_mask = result["receiver_mask"]
    if receiver_mask_path is not None and Path(receiver_mask_path).exists():
        receiver_mask = _load_mask(receiver_mask_path, size)

    point_z = None if point_map is None else point_map[..., 2]
    prior_before = _load_prior(sg_prior_path)
    prior_after = None
    if prior_before is not None:
        receiver_resized = cv2.resize(receiver_mask.astype(np.float32), (prior_before.shape[1], prior_before.shape[0]), interpolation=cv2.INTER_NEAREST)
        prior_after = prior_before * receiver_resized

    panels = [
        ("composite", composite),
        ("object", _overlay(composite, object_mask, (0, 220, 90))),
        ("point z", _heatmap(point_z, size)),
        ("sam masks", _multi_overlay(composite, sam_masks)),
        ("receiver", _overlay(composite, receiver_mask, (80, 170, 255))),
        ("prior before", _heatmap(prior_before, size)),
        ("prior after", _heatmap(prior_after, size)),
    ]
    output_path = Path(output_dir) / f"{sample_id}_receiver.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _board(panels).save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize receiver / ground candidate mask for one sample.")
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--composite_path", required=True)
    parser.add_argument("--object_mask_path", required=True)
    parser.add_argument("--point_map_path", default=None)
    parser.add_argument("--sam_mask_dir", default=None)
    parser.add_argument("--receiver_mask_path", default=None)
    parser.add_argument("--sg_prior_path", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    output_path = visualize_receiver_mask(
        sample_id=args.sample_id,
        composite_path=args.composite_path,
        object_mask_path=args.object_mask_path,
        point_map_path=args.point_map_path,
        sam_mask_dir=args.sam_mask_dir,
        receiver_mask_path=args.receiver_mask_path,
        sg_prior_path=args.sg_prior_path,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(f"Saved receiver visualization: {output_path}")


if __name__ == "__main__":
    main()
