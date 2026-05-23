from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_a_segmentation import filter_masks, run_sam_on_image, save_mask_overlay, save_mask_png


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "segmentation_config.yaml"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_segmentation_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Segmentation config must contain a YAML mapping.")
    return config


def _segmentation_section(config: dict) -> dict:
    return config.get("segmentation", config)


def resolve_segmentation_path(config_path: str | Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parents[1] / path).resolve()


def list_input_images(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.exists():
        return []
    return sorted(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_sample_output_paths(sample_id: str, mask_root: str | Path, visualization_dir: str | Path) -> dict[str, Path]:
    return {
        "mask_dir": Path(mask_root) / sample_id,
        "overlay": Path(visualization_dir) / f"{sample_id}_sam_overlay.png",
    }


def save_candidate_masks(mask_dir: str | Path, masks: list[dict], overwrite: bool = False) -> list[Path]:
    output_dir = Path(mask_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, mask in enumerate(masks):
        output_path = output_dir / f"mask_{index:03d}.png"
        if output_path.exists() and not overwrite:
            continue
        save_mask_png(output_path, mask["mask"])
        written.append(output_path)
    return written


def run_sam_batch(config_path: str | Path = DEFAULT_CONFIG, limit: int | None = None, overwrite: bool = False) -> dict:
    config = load_segmentation_config(config_path)
    segmentation = _segmentation_section(config)

    input_dir = resolve_segmentation_path(config_path, segmentation.get("input_image_dir", "data/desobav2/composite"))
    output_mask_dir = resolve_segmentation_path(config_path, segmentation.get("output_mask_dir", "data/intermediate/sam_masks"))
    visualization_dir = resolve_segmentation_path(
        config_path,
        segmentation.get("output_visualization_dir", "data/outputs/visualizations/sam"),
    )
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    images = list_input_images(input_dir)
    if limit is not None:
        images = images[: max(0, int(limit))]

    summary = {"processed": 0, "skipped": 0, "failed": 0, "mask_files": [], "overlays": []}
    for image_path in images:
        sample_id = image_path.stem
        output_paths = build_sample_output_paths(sample_id, output_mask_dir, visualization_dir)
        existing_masks = sorted(output_paths["mask_dir"].glob("mask_*.png")) if output_paths["mask_dir"].exists() else []
        if existing_masks and output_paths["overlay"].exists() and not overwrite:
            print(f"Skip existing SAM outputs: {sample_id}")
            summary["skipped"] += 1
            continue

        print(f"Running SAM/SAM2 segmentation: {image_path}")
        try:
            masks = run_sam_on_image(image_path, config)
            masks = filter_masks(
                masks,
                min_mask_area=int(segmentation.get("min_mask_area", 100)),
                max_masks=int(segmentation.get("max_masks_per_image", 50)),
            )
            mask_files = save_candidate_masks(output_paths["mask_dir"], masks, overwrite=overwrite)
            save_mask_overlay(image_path, masks, output_paths["overlay"])
        except Exception:
            summary["failed"] += 1
            raise

        summary["processed"] += 1
        summary["mask_files"].extend(mask_files)
        summary["overlays"].append(output_paths["overlay"])
        print(f"Saved {len(mask_files)} masks to {output_paths['mask_dir']}")
        print(f"Saved overlay: {output_paths['overlay']}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional SAM/SAM2 candidate mask generation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to segmentation_config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks and overlay visualizations.")
    args = parser.parse_args()

    summary = run_sam_batch(args.config, args.limit, args.overwrite)
    print(
        "SAM batch complete: "
        f"processed={summary['processed']} skipped={summary['skipped']} failed={summary['failed']}"
    )


if __name__ == "__main__":
    main()
