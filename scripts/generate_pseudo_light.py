from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.desobav2_dataset import DESOBAv2Dataset
from modules.module_c_shadow_sketch import compute_centroid


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _output_dir_from_config(dataset: DESOBAv2Dataset) -> Path:
    path = dataset.config.get("intermediate", {}).get("pseudo_light_dir", "data/intermediate/pseudo_light")
    output = Path(path)
    if not output.is_absolute():
        output = dataset.project_root / output
    return output.resolve()


def build_pseudo_light_record(sample: dict, minimum_area: int = 1) -> dict:
    object_mask = _load_mask(sample["object_mask_path"])
    shadow_mask = _load_mask(sample["shadow_mask_path"])

    object_centroid = np.asarray(compute_centroid(object_mask, minimum_area), dtype=np.float64)
    shadow_centroid = np.asarray(compute_centroid(shadow_mask, minimum_area), dtype=np.float64)
    shadow_direction = shadow_centroid - object_centroid
    distance = float(np.linalg.norm(shadow_direction))
    if distance == 0.0:
        raise ValueError(f"Sample '{sample['sample_id']}' has identical object and shadow centroids.")

    light_direction = -shadow_direction
    height, width = object_mask.shape[:2]
    diagonal = float(np.hypot(width, height))
    confidence = float(np.clip(distance / diagonal, 0.0, 1.0))

    return {
        "sample_id": sample["sample_id"],
        "object_centroid_xy": object_centroid.astype(float).tolist(),
        "shadow_centroid_xy": shadow_centroid.astype(float).tolist(),
        "shadow_direction_2d": shadow_direction.astype(float).tolist(),
        "light_direction_2d": light_direction.astype(float).tolist(),
        "confidence": confidence,
    }


def generate_pseudo_light(config_path: str | Path = DEFAULT_CONFIG) -> list[Path]:
    dataset = DESOBAv2Dataset(config_path=config_path, strict=False)
    output_dir = _output_dir_from_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for sample in dataset:
        record = build_pseudo_light_record(sample)
        output_path = output_dir / f"{sample['sample_id']}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo light directions from DESOBAv2 masks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to dataset_config.yaml.")
    args = parser.parse_args()

    written = generate_pseudo_light(args.config)
    print(f"Wrote {len(written)} pseudo light files.")


if __name__ == "__main__":
    main()
