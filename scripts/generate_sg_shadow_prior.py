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
from modules.module_c_sg_shadow_prior import compute_sg_shadow_prior


DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"
DEFAULT_SKETCH_CONFIG = PROJECT_ROOT / "configs" / "sketch_config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate" / "sg_shadow_prior"


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_project_path(dataset: DESOBAv2Dataset, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (dataset.project_root / path).resolve()


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _save_prior_png(path: Path, prior: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(np.asarray(prior) * 255.0), 0, 255).astype(np.uint8)).save(path)
    return path


def _load_sg_params(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    return record["sg_lobes"]


def generate_sg_shadow_priors(
    dataset_config: str | Path = DEFAULT_DATASET_CONFIG,
    sketch_config: str | Path = DEFAULT_SKETCH_CONFIG,
    limit: int | None = None,
    overwrite: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    dataset = DESOBAv2Dataset(dataset_config, strict=False)
    config = _load_yaml(sketch_config)
    prior_config = config.get("sg_shadow_prior", {})
    point_map_dir = _resolve_project_path(dataset, "data/intermediate/point_maps")
    sg_dir = _resolve_project_path(dataset, "data/intermediate/sg_params")
    receiver_mask_dir = _resolve_project_path(dataset, "data/intermediate/receiver_masks")
    output_root = _resolve_project_path(dataset, output_dir)
    debug_dir = output_root / "debug"
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(prior_config.get("save_per_lobe_debug", True)):
        debug_dir.mkdir(parents=True, exist_ok=True)

    samples = list(dataset)
    if limit is not None:
        samples = samples[: max(0, int(limit))]

    written: list[Path] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        npy_path = output_root / f"{sample_id}.npy"
        png_path = output_root / f"{sample_id}.png"
        if npy_path.exists() and png_path.exists() and not overwrite:
            print(f"Skip existing SG shadow prior: {sample_id}")
            continue

        point_map_path = point_map_dir / f"{sample_id}.npy"
        sg_path = sg_dir / f"{sample_id}.json"
        if not point_map_path.exists():
            print(f"Skip missing point map: {point_map_path}")
            continue
        if not sg_path.exists():
            print(f"Skip missing SG params: {sg_path}")
            continue

        object_mask = _load_mask(sample["object_mask_path"])
        point_map = np.load(point_map_path)
        sg_lobes = _load_sg_params(sg_path)
        receiver_mask_path = receiver_mask_dir / f"{sample_id}.png"
        receiver_mask = _load_mask(receiver_mask_path) if receiver_mask_path.exists() else None
        result = compute_sg_shadow_prior(
            object_mask,
            point_map,
            sg_lobes,
            config,
            receiver_mask=receiver_mask,
            return_debug=True,
        )

        np.save(npy_path, result["combined"].astype(np.float32))
        _save_prior_png(png_path, result["combined"])
        written.append(npy_path)

        if bool(prior_config.get("save_per_lobe_debug", True)):
            _save_prior_png(debug_dir / f"{sample_id}_direct.png", result["direct"])
            if "diffuse_modulated" in result:
                _save_prior_png(debug_dir / f"{sample_id}_diffuse_modulated.png", result["diffuse_modulated"])

        print(f"Saved SG shadow prior: {npy_path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SG-aware physics shadow priors.")
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG), help="Path to dataset_config.yaml.")
    parser.add_argument("--sketch-config", default=str(DEFAULT_SKETCH_CONFIG), help="Path to sketch_config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for SG shadow priors.")
    args = parser.parse_args()

    written = generate_sg_shadow_priors(args.dataset_config, args.sketch_config, args.limit, args.overwrite, args.output_dir)
    print(f"Wrote {len(written)} SG shadow prior files.")


if __name__ == "__main__":
    main()
