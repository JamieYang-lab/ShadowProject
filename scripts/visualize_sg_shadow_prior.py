from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.desobav2_dataset import DESOBAv2Dataset


DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"
PANEL_TITLE_HEIGHT = 24
PANEL_GAP = 8


def _resolve_project_path(dataset: DESOBAv2Dataset, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (dataset.project_root / path).resolve()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").resize(size, Image.Resampling.NEAREST)


def _overlay_mask(base: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: int = 120) -> Image.Image:
    output = base.copy().convert("RGBA")
    binary = np.asarray(mask) > 0
    overlay = Image.new("RGBA", base.size, color + (0,))
    alpha_mask = np.zeros(binary.shape, dtype=np.uint8)
    alpha_mask[binary] = alpha
    overlay.putalpha(Image.fromarray(alpha_mask))
    return Image.alpha_composite(output, overlay).convert("RGB")


def _heatmap(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if float(values.max()) > float(values.min()):
        values = (values - float(values.min())) / (float(values.max()) - float(values.min()))
    else:
        values = np.zeros(values.shape, dtype=np.float32)
    gray = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)


def _load_optional_prior(path: Path) -> np.ndarray | None:
    if path.exists():
        if path.suffix.lower() == ".npy":
            return np.load(path)
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return None


def _point_z_panel(point_map_path: Path, size: tuple[int, int]) -> Image.Image:
    if not point_map_path.exists():
        return Image.new("RGB", size, (32, 32, 32))
    point_map = np.load(point_map_path)
    return _heatmap(point_map[..., 2], size)


def _text_panel(size: tuple[int, int], metadata: dict | None) -> Image.Image:
    panel = Image.new("RGB", size, (28, 31, 36))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    lines = ["SG metadata unavailable"]
    if metadata:
        lines = [
            f"direct_mu: {np.round(metadata.get('direct_mu', []), 3).tolist()}",
            f"direct_lambda: {metadata.get('direct_lambda')}",
            f"direct_amp: {metadata.get('direct_amplitude')}",
            f"diffuse_lambda: {metadata.get('diffuse_lambda')}",
            f"diffuse_amp: {metadata.get('diffuse_amplitude')}",
        ]
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(235, 235, 235), font=font)
        y += 18
    return panel


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


def visualize_sg_shadow_prior(
    dataset_config: str | Path = DEFAULT_DATASET_CONFIG,
    max_samples: int = 10,
    sample_id: str | None = None,
) -> list[Path]:
    dataset = DESOBAv2Dataset(dataset_config, strict=False)
    point_map_dir = _resolve_project_path(dataset, "data/intermediate/point_maps")
    old_sketch_dir = _resolve_project_path(dataset, "data/intermediate/shadow_sketch")
    prior_dir = _resolve_project_path(dataset, "data/intermediate/sg_shadow_prior")
    debug_dir = prior_dir / "debug"
    output_dir = _resolve_project_path(dataset, "data/outputs/visualizations/sg_shadow_prior")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(dataset)
    if sample_id is not None:
        samples = [sample for sample in samples if sample["sample_id"] == sample_id]
        if not samples:
            raise ValueError(f"Sample id not found: {sample_id}")
    else:
        samples = samples[: max(0, int(max_samples))]

    written: list[Path] = []
    for sample in samples:
        sid = sample["sample_id"]
        composite = _load_rgb(sample["composite_path"])
        size = composite.size
        object_mask = _load_mask(sample["object_mask_path"], size)
        shadow_mask = _load_mask(sample["shadow_mask_path"], size)

        old_sketch = _load_optional_prior(old_sketch_dir / f"{sid}.png")
        combined = _load_optional_prior(prior_dir / f"{sid}.npy")
        direct = _load_optional_prior(debug_dir / f"{sid}_direct.png")
        metadata = None
        sg_path = _resolve_project_path(dataset, f"data/intermediate/sg_params/{sid}.json")
        if sg_path.exists():
            with open(sg_path, "r", encoding="utf-8") as handle:
                sg = json.load(handle)
            direct_lobe = next((lobe for lobe in sg["sg_lobes"] if lobe.get("type") == "direct"), sg["sg_lobes"][0])
            diffuse_lobe = next((lobe for lobe in sg["sg_lobes"] if lobe.get("type") == "diffuse"), sg["sg_lobes"][1])
            metadata = {
                "direct_mu": direct_lobe["mu"],
                "direct_lambda": direct_lobe["lambda"],
                "direct_amplitude": direct_lobe["amplitude"],
                "diffuse_lambda": diffuse_lobe["lambda"],
                "diffuse_amplitude": diffuse_lobe["amplitude"],
            }

        blank = Image.new("RGB", size, (32, 32, 32))
        panels = [
            ("composite", composite),
            ("object", _overlay_mask(composite, object_mask, (0, 220, 90))),
            ("point z", _point_z_panel(point_map_dir / f"{sid}.npy", size)),
            ("gt shadow", _overlay_mask(composite, shadow_mask, (255, 70, 70))),
            ("old sketch", _heatmap(old_sketch, size) if old_sketch is not None else blank),
            ("sg prior", _heatmap(combined, size) if combined is not None else blank),
            ("direct", _heatmap(direct, size) if direct is not None else blank),
            ("metadata", _text_panel(size, metadata)),
        ]
        output_path = output_dir / f"{sid}_sg_prior.png"
        _board(panels).save(output_path)
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SG-aware physics shadow priors.")
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG), help="Path to dataset_config.yaml.")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples to visualize.")
    parser.add_argument("--sample-id", default=None, help="Optional sample id to visualize.")
    args = parser.parse_args()

    written = visualize_sg_shadow_prior(args.dataset_config, args.max_samples, args.sample_id)
    print(f"Wrote {len(written)} SG shadow prior visualizations.")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
