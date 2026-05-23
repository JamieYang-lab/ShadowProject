from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_a_receiver_selection import compute_receiver_mask, load_sam_masks, save_receiver_mask_png


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "receiver_config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate" / "receiver_masks"


def load_receiver_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Receiver config must contain a YAML mapping.")
    return config


def _load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _image_shape(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def generate_receiver_mask(
    sample_id: str,
    composite_path: str | Path,
    object_mask_path: str | Path,
    point_map_path: str | Path | None = None,
    sam_mask_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    config = load_receiver_config(config_path)
    image_shape = _image_shape(composite_path)
    object_mask = _load_mask(object_mask_path)
    point_map = None
    if point_map_path is not None and Path(point_map_path).exists():
        point_map = np.load(point_map_path)
    sam_masks = load_sam_masks(sam_mask_dir, image_shape)
    result = compute_receiver_mask(object_mask, image_shape, point_map=point_map, sam_masks=sam_masks, config=config)
    output_path = Path(output_dir) / f"{sample_id}.png"
    save_receiver_mask_png(output_path, result["receiver_mask"])
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate receiver / ground candidate mask for one sample.")
    parser.add_argument("--sample_id", required=True, help="Sample id.")
    parser.add_argument("--composite_path", required=True, help="Path to composite image.")
    parser.add_argument("--object_mask_path", required=True, help="Path to object mask image.")
    parser.add_argument("--point_map_path", default=None, help="Optional point map .npy path.")
    parser.add_argument("--sam_mask_dir", default=None, help="Optional SAM mask directory.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to receiver_config.yaml.")
    args = parser.parse_args()

    output_path = generate_receiver_mask(
        sample_id=args.sample_id,
        composite_path=args.composite_path,
        object_mask_path=args.object_mask_path,
        point_map_path=args.point_map_path,
        sam_mask_dir=args.sam_mask_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    print(f"Saved receiver mask: {output_path}")


if __name__ == "__main__":
    main()
