from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.desobav2_dataset import DESOBAv2Dataset


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"


def _mask_area(path: Path) -> int:
    mask = np.asarray(Image.open(path).convert("L"))
    return int((mask > 0).sum())


def _area_stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def inspect_dataset(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    dataset = DESOBAv2Dataset(config_path=config_path, strict=False)
    object_areas = [_mask_area(sample["object_mask_path"]) for sample in dataset]
    shadow_areas = [_mask_area(sample["shadow_mask_path"]) for sample in dataset]

    return {
        "valid_samples": len(dataset),
        "missing_files": dataset.count_missing_files(),
        "missing_by_field": dataset.missing_by_field(),
        "object_mask_area": _area_stats(object_areas),
        "shadow_mask_area": _area_stats(shadow_areas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DESOBAv2 dataset scaffold inputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to dataset_config.yaml.")
    args = parser.parse_args()

    summary = inspect_dataset(args.config)
    print(f"Valid samples: {summary['valid_samples']}")
    print(f"Missing files: {summary['missing_files']}")
    print(f"Missing by field: {summary['missing_by_field']}")
    print(f"Object mask area: {summary['object_mask_area']}")
    print(f"Shadow mask area: {summary['shadow_mask_area']}")


if __name__ == "__main__":
    main()
