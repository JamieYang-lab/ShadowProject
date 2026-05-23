from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.desobav2_dataset import DESOBAv2Dataset
from modules.module_c_shadow_sketch import (
    estimate_light_direction_from_masks,
    generate_shadow_sketch_from_direction,
    save_shadow_sketch,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"
DEFAULT_SKETCH_CONFIG = PROJECT_ROOT / "configs" / "sketch_config.yaml"


def _resolve_output(dataset: DESOBAv2Dataset, key: str, default: str) -> Path:
    path = dataset.config.get("intermediate", {}).get(key, default)
    output = Path(path)
    if not output.is_absolute():
        output = dataset.project_root / output
    return output.resolve()


def _load_sketch_config(path: str | Path = DEFAULT_SKETCH_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _load_light_direction(sample: dict, pseudo_dir: Path) -> list[float]:
    pseudo_path = pseudo_dir / f"{sample['sample_id']}.json"
    if pseudo_path.exists():
        with open(pseudo_path, "r", encoding="utf-8") as handle:
            pseudo = json.load(handle)
        return pseudo["light_direction_2d"]

    object_mask = _load_mask(sample["object_mask_path"])
    shadow_mask = _load_mask(sample["shadow_mask_path"])
    return estimate_light_direction_from_masks(object_mask, shadow_mask)


def generate_shadow_sketches(
    config_path: str | Path = DEFAULT_CONFIG,
    sketch_config_path: str | Path = DEFAULT_SKETCH_CONFIG,
) -> list[Path]:
    dataset = DESOBAv2Dataset(config_path=config_path, strict=False)
    pseudo_dir = _resolve_output(dataset, "pseudo_light_dir", "data/intermediate/pseudo_light")
    output_dir = _resolve_output(dataset, "shadow_sketch_dir", "data/intermediate/shadow_sketch")
    output_dir.mkdir(parents=True, exist_ok=True)
    sketch_config = _load_sketch_config(sketch_config_path)

    written: list[Path] = []
    for sample in dataset:
        with Image.open(sample["composite_path"]) as composite:
            image_shape = (composite.height, composite.width)
        object_mask = _load_mask(sample["object_mask_path"])
        light_direction = _load_light_direction(sample, pseudo_dir)
        sketch = generate_shadow_sketch_from_direction(object_mask, light_direction, image_shape, sketch_config)
        output_path = output_dir / f"{sample['sample_id']}.png"
        save_shadow_sketch(sketch, output_path)
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coarse shadow sketch PNGs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to dataset_config.yaml.")
    parser.add_argument("--sketch-config", default=str(DEFAULT_SKETCH_CONFIG), help="Path to sketch_config.yaml.")
    args = parser.parse_args()

    written = generate_shadow_sketches(args.config, args.sketch_config)
    print(f"Wrote {len(written)} shadow sketch files.")


if __name__ == "__main__":
    main()
