from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_b_weather import load_weather_config
from modules.module_b_weather_to_sg import save_sg_params_json, weather_to_sg_params


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "weather_config.yaml"


def _resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parents[1] / path).resolve()


def _default_weather_path(config_path: str | Path, config: dict, sample_id: str) -> Path:
    output_dir = _resolve_path(config_path, config.get("output_weather_dir", "data/intermediate/weather"))
    return output_dir / f"{sample_id}.json"


def _default_sg_path(config_path: str | Path, config: dict, sample_id: str) -> Path:
    output_dir = _resolve_path(config_path, config.get("output_sg_dir", "data/intermediate/sg_params"))
    return output_dir / f"{sample_id}.json"


def generate_weather_aware_sg(
    weather_config_path: str | Path,
    sample_id: str,
    sun_vector: list[float],
    weather_json: str | Path | None = None,
) -> tuple[dict, Path]:
    config = load_weather_config(weather_config_path)
    weather_path = Path(weather_json) if weather_json is not None else _default_weather_path(weather_config_path, config, sample_id)
    if not weather_path.is_absolute():
        weather_path = _resolve_path(weather_config_path, weather_path)

    with open(weather_path, "r", encoding="utf-8") as handle:
        weather_features = json.load(handle)

    sg_params = weather_to_sg_params(sample_id, sun_vector, weather_features, config)
    output_path = _default_sg_path(weather_config_path, config, sample_id)
    save_sg_params_json(output_path, sg_params)
    return sg_params, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weather-aware K=2 SG lighting parameters.")
    parser.add_argument("--weather-config", default=str(DEFAULT_CONFIG), help="Path to weather_config.yaml.")
    parser.add_argument("--sample-id", required=True, help="Sample id.")
    parser.add_argument("--sun-vector", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"), help="NOAA/pvlib sun vector.")
    parser.add_argument("--weather-json", default=None, help="Optional path to weather feature JSON.")
    args = parser.parse_args()

    sg_params, output_path = generate_weather_aware_sg(
        weather_config_path=args.weather_config,
        sample_id=args.sample_id,
        sun_vector=args.sun_vector,
        weather_json=args.weather_json,
    )
    direct, diffuse = sg_params["sg_lobes"]
    print(f"Saved weather-aware SG params: {output_path}")
    print(
        "SG summary: "
        f"source={sg_params['source']} mode={sg_params['weather']['weather_mode']} "
        f"direct_lambda={direct['lambda']} direct_amp={direct['amplitude']} "
        f"diffuse_lambda={diffuse['lambda']} diffuse_amp={diffuse['amplitude']}"
    )


if __name__ == "__main__":
    main()
