from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_a_geometry import (
    ensure_dir,
    run_moge_on_image,
    save_depth_png,
    save_normal_png,
    save_point_map,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_config.yaml"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_geometry_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Geometry config must contain a YAML mapping.")
    return config


def _geometry_section(config: dict) -> dict:
    return config.get("geometry", config)


def resolve_geometry_path(config_path: str | Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parents[1] / path).resolve()


def list_input_images(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    if not directory.exists():
        return []
    return sorted(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_output_paths(sample_id: str, point_map_dir: str | Path, depth_dir: str | Path, normal_dir: str | Path) -> dict[str, Path]:
    return {
        "point_map": Path(point_map_dir) / f"{sample_id}.npy",
        "depth_png": Path(depth_dir) / f"{sample_id}.png",
        "normal_png": Path(normal_dir) / f"{sample_id}.png",
    }


def run_moge_batch(config_path: str | Path = DEFAULT_CONFIG, limit: int | None = None, overwrite: bool = False) -> dict:
    config = load_geometry_config(config_path)
    geometry = _geometry_section(config)

    input_dir = resolve_geometry_path(config_path, geometry.get("input_image_dir", "data/desobav2/composite"))
    point_map_dir = resolve_geometry_path(config_path, geometry.get("output_point_map_dir", "data/intermediate/point_maps"))
    depth_dir = resolve_geometry_path(config_path, geometry.get("output_depth_dir", "data/intermediate/depth_maps"))
    normal_dir = resolve_geometry_path(config_path, geometry.get("output_normal_dir", "data/intermediate/normal_maps"))
    save_depth = bool(geometry.get("save_depth_png", True))
    save_normal = bool(geometry.get("save_normal_png", True))

    ensure_dir(point_map_dir)
    if save_depth:
        ensure_dir(depth_dir)
    if save_normal:
        ensure_dir(normal_dir)

    images = list_input_images(input_dir)
    if limit is not None:
        images = images[: max(0, int(limit))]

    summary = {"processed": 0, "skipped": 0, "failed": 0, "outputs": []}
    for image_path in images:
        sample_id = image_path.stem
        output_paths = build_output_paths(sample_id, point_map_dir, depth_dir, normal_dir)
        if output_paths["point_map"].exists() and not overwrite:
            print(f"Skip existing point map: {output_paths['point_map']}")
            summary["skipped"] += 1
            continue

        print(f"Running MoGe inference: {image_path}")
        try:
            result = run_moge_on_image(image_path, config)
            save_point_map(output_paths["point_map"], result["point_map"])
            if save_depth and "depth" in result:
                save_depth_png(output_paths["depth_png"], result["depth"])
            if save_normal and "normal" in result:
                save_normal_png(output_paths["normal_png"], result["normal"])
        except Exception:
            summary["failed"] += 1
            raise

        summary["processed"] += 1
        summary["outputs"].append(output_paths)
        print(f"Saved point map: {output_paths['point_map']}")
        if save_depth and "depth" in result:
            print(f"Saved depth PNG: {output_paths['depth_png']}")
        if save_normal and "normal" in result:
            print(f"Saved normal PNG: {output_paths['normal_png']}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional MoGe/MoGe-2 geometry inference on composite images.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to geometry_config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing point map outputs.")
    args = parser.parse_args()

    summary = run_moge_batch(args.config, args.limit, args.overwrite)
    print(
        "MoGe batch complete: "
        f"processed={summary['processed']} skipped={summary['skipped']} failed={summary['failed']}"
    )


if __name__ == "__main__":
    main()
