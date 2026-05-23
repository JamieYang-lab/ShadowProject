from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.desobav2_dataset import DESOBAv2Dataset


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"


PANEL_TITLE_HEIGHT = 24
PANEL_GAP = 8
ARROW_WIDTH = 4


def _resolve_config_path(dataset: DESOBAv2Dataset, section: str, key: str, default: str) -> Path:
    path = dataset.config.get(section, {}).get(key, default)
    output = Path(path)
    if not output.is_absolute():
        output = dataset.project_root / output
    return output.resolve()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").resize(size, Image.Resampling.NEAREST)


def _mask_overlay(base: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: int = 120) -> Image.Image:
    output = base.copy().convert("RGBA")
    mask_array = np.asarray(mask) > 0
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha_mask = np.zeros(mask_array.shape, dtype=np.uint8)
    alpha_mask[mask_array] = alpha
    overlay.putalpha(Image.fromarray(alpha_mask))
    return Image.alpha_composite(output, overlay).convert("RGB")


def _sketch_overlay(base: Image.Image, sketch: Image.Image) -> Image.Image:
    output = base.copy().convert("RGBA")
    sketch_array = np.asarray(sketch.convert("L"))
    overlay = Image.new("RGBA", base.size, (60, 140, 255, 0))
    overlay.putalpha(Image.fromarray(np.clip(sketch_array, 0, 180).astype(np.uint8)))
    return Image.alpha_composite(output, overlay).convert("RGB")


def _normalize_2d(vector) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape[0] < 2:
        raise ValueError("Arrow vector must contain at least 2 values.")
    values = values[:2]
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Arrow vector cannot be zero.")
    return values / norm


def _draw_arrow(
    image: Image.Image,
    origin_xy,
    vector,
    color: tuple[int, int, int],
    label: str,
    length: float | None = None,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    origin = np.asarray(origin_xy, dtype=np.float64)
    direction = _normalize_2d(vector)
    arrow_length = length or max(24.0, min(width, height) * 0.28)
    end = origin + direction * arrow_length
    end[0] = np.clip(end[0], 4, width - 5)
    end[1] = np.clip(end[1], 4, height - 5)

    draw.line([tuple(origin), tuple(end)], fill=color, width=ARROW_WIDTH)

    head = 10.0
    angle = np.arctan2(direction[1], direction[0])
    for delta in (2.55, -2.55):
        side = end + head * np.asarray([np.cos(angle + delta), np.sin(angle + delta)])
        draw.line([tuple(end), tuple(side)], fill=color, width=ARROW_WIDTH)

    label_xy = (int(min(max(end[0] + 5, 2), width - 80)), int(min(max(end[1] + 5, 2), height - 16)))
    draw.text(label_xy, label, fill=color, font=ImageFont.load_default())
    return output


def _add_panel_title(panel: Image.Image, title: str) -> Image.Image:
    output = Image.new("RGB", (panel.width, panel.height + PANEL_TITLE_HEIGHT), (28, 31, 36))
    output.paste(panel, (0, PANEL_TITLE_HEIGHT))
    draw = ImageDraw.Draw(output)
    draw.text((8, 6), title, fill=(235, 235, 235), font=ImageFont.load_default())
    return output


def _make_debug_board(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    titled = [_add_panel_title(panel, title) for title, panel in panels]
    width = sum(panel.width for panel in titled) + PANEL_GAP * (len(titled) - 1)
    height = max(panel.height for panel in titled)
    board = Image.new("RGB", (width, height), (18, 20, 24))

    x = 0
    for panel in titled:
        board.paste(panel, (x, 0))
        x += panel.width + PANEL_GAP
    return board


def _load_sketch(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing shadow sketch: {path}")
    with Image.open(path) as image:
        return image.convert("L").resize(size, Image.Resampling.BILINEAR)


def visualize_sample(sample: dict, pseudo_dir: Path, sg_dir: Path, sketch_dir: Path, output_dir: Path) -> Path:
    sample_id = sample["sample_id"]
    pseudo_path = pseudo_dir / f"{sample_id}.json"
    sg_path = sg_dir / f"{sample_id}.json"
    sketch_path = sketch_dir / f"{sample_id}.png"

    if not pseudo_path.exists():
        raise FileNotFoundError(f"Missing pseudo light JSON: {pseudo_path}")
    if not sg_path.exists():
        raise FileNotFoundError(f"Missing SG params JSON: {sg_path}")

    composite = _load_rgb(sample["composite_path"])
    object_mask = _load_mask(sample["object_mask_path"], composite.size)
    shadow_mask = _load_mask(sample["shadow_mask_path"], composite.size)
    shadow_sketch = _load_sketch(sketch_path, composite.size)
    pseudo = _load_json(pseudo_path)
    sg_record = _load_json(sg_path)

    object_overlay = _mask_overlay(composite, object_mask, (0, 220, 90), alpha=125)
    shadow_overlay = _mask_overlay(composite, shadow_mask, (255, 70, 70), alpha=125)

    arrow_panel = composite.copy()
    origin = pseudo["object_centroid_xy"]
    arrow_panel = _draw_arrow(arrow_panel, origin, pseudo["light_direction_2d"], (255, 210, 40), "pseudo")
    direct_lobe = next((lobe for lobe in sg_record["sg_lobes"] if lobe.get("type") == "direct"), sg_record["sg_lobes"][0])
    arrow_panel = _draw_arrow(arrow_panel, origin, direct_lobe["mu"][:2], (50, 210, 255), "sg direct")

    sketch_overlay = _sketch_overlay(composite, shadow_sketch)
    panels = [
        ("composite", composite),
        ("object mask", object_overlay),
        ("shadow mask", shadow_overlay),
        ("light arrows", arrow_panel),
        ("sketch overlay", sketch_overlay),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}_preprocessing.png"
    _make_debug_board(panels).save(output_path)
    return output_path


def visualize_preprocessing(
    config_path: str | Path = DEFAULT_CONFIG,
    sample_id: str | None = None,
    max_samples: int = 10,
) -> list[Path]:
    dataset = DESOBAv2Dataset(config_path=config_path, strict=False)
    pseudo_dir = _resolve_config_path(dataset, "intermediate", "pseudo_light_dir", "data/intermediate/pseudo_light")
    sg_dir = _resolve_config_path(dataset, "intermediate", "sg_params_dir", "data/intermediate/sg_params")
    sketch_dir = _resolve_config_path(dataset, "intermediate", "shadow_sketch_dir", "data/intermediate/shadow_sketch")
    output_dir = _resolve_config_path(dataset, "outputs", "visualization_dir", "data/outputs/visualizations")

    samples = list(dataset)
    if sample_id is not None:
        samples = [sample for sample in samples if sample["sample_id"] == sample_id]
        if not samples:
            raise ValueError(f"Sample id not found: {sample_id}")
    else:
        samples = samples[: max(0, int(max_samples))]

    return [visualize_sample(sample, pseudo_dir, sg_dir, sketch_dir, output_dir) for sample in samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize preprocessing outputs for DESOBAv2 samples.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to dataset_config.yaml.")
    parser.add_argument("--sample-id", default=None, help="Optional sample id to visualize.")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples to visualize when sample id is omitted.")
    args = parser.parse_args()

    written = visualize_preprocessing(args.config, args.sample_id, args.max_samples)
    print(f"Wrote {len(written)} visualization files.")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
