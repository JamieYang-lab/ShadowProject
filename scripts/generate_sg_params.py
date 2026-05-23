from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.desobav2_dataset import DESOBAv2Dataset
from modules.module_b_sg_light import initialize_sg_from_light_direction


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"
DEFAULT_SG_CONFIG = PROJECT_ROOT / "configs" / "sg_light_config.yaml"


def _resolve_output(dataset: DESOBAv2Dataset, key: str, default: str) -> Path:
    path = dataset.config.get("intermediate", {}).get(key, default)
    output = Path(path)
    if not output.is_absolute():
        output = dataset.project_root / output
    return output.resolve()


def _load_sg_config(path: str | Path = DEFAULT_SG_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config.get("sg_light", config)


def light_2d_to_3d(light_direction_2d) -> list[float]:
    direction_2d = np.asarray(light_direction_2d, dtype=np.float64)
    if direction_2d.shape[0] != 2:
        raise ValueError("light_direction_2d must contain exactly 2 values.")
    if float(np.linalg.norm(direction_2d)) == 0.0:
        raise ValueError("light_direction_2d cannot be zero.")
    direction_3d = np.asarray([direction_2d[0], direction_2d[1], 1.0], dtype=np.float64)
    return (direction_3d / float(np.linalg.norm(direction_3d))).astype(float).tolist()


def build_sg_record(sample_id: str, light_direction_2d, sg_config: dict | None = None) -> dict:
    sg_config = sg_config or _load_sg_config()
    direct_config = sg_config.get("direct", {})
    diffuse_config = sg_config.get("diffuse", {})
    light_direction_3d = light_2d_to_3d(light_direction_2d)

    lobes = initialize_sg_from_light_direction(
        light_direction_3d,
        direct_sharpness=float(direct_config.get("default_lambda", 80.0)),
        direct_amplitude=float(direct_config.get("default_amplitude", 1.0)),
        diffuse_sharpness=float(diffuse_config.get("default_lambda", 5.0)),
        diffuse_amplitude=float(diffuse_config.get("default_amplitude", 0.3)),
    )

    return {
        "sample_id": sample_id,
        "sg_lobes": [lobe.to_dict() for lobe in lobes],
    }


def generate_sg_params(
    config_path: str | Path = DEFAULT_CONFIG,
    sg_config_path: str | Path = DEFAULT_SG_CONFIG,
) -> list[Path]:
    dataset = DESOBAv2Dataset(config_path=config_path, strict=False)
    pseudo_dir = _resolve_output(dataset, "pseudo_light_dir", "data/intermediate/pseudo_light")
    output_dir = _resolve_output(dataset, "sg_params_dir", "data/intermediate/sg_params")
    output_dir.mkdir(parents=True, exist_ok=True)
    sg_config = _load_sg_config(sg_config_path)

    written: list[Path] = []
    for pseudo_path in sorted(pseudo_dir.glob("*.json")):
        with open(pseudo_path, "r", encoding="utf-8") as handle:
            pseudo = json.load(handle)
        record = build_sg_record(pseudo["sample_id"], pseudo["light_direction_2d"], sg_config)
        output_path = output_dir / f"{pseudo['sample_id']}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate K=2 SG lighting parameters from pseudo light JSON.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to dataset_config.yaml.")
    parser.add_argument("--sg-config", default=str(DEFAULT_SG_CONFIG), help="Path to sg_light_config.yaml.")
    args = parser.parse_args()

    written = generate_sg_params(args.config, args.sg_config)
    print(f"Wrote {len(written)} SG parameter files.")


if __name__ == "__main__":
    main()
