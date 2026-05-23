from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_b_light_engine import get_solar_angles, solar_angles_to_world_vector
from modules.module_b_weather import build_weather_features, fetch_current_weather, load_weather_config
from modules.module_b_weather_to_sg import SOURCE_NAME, save_sg_params_json, weather_to_sg_params
from utils.math_utils import world_to_camera_vector


DEFAULT_METADATA_SG_CONFIG = PROJECT_ROOT / "configs" / "metadata_sg_config.yaml"
DEFAULT_WEATHER_CONFIG = PROJECT_ROOT / "configs" / "weather_config.yaml"


REQUIRED_METADATA_FIELDS = [
    "sample_id",
    "latitude",
    "longitude",
    "timestamp",
    "timezone",
    "camera_heading",
    "camera_pitch",
    "camera_roll",
]


def load_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        metadata = json.load(handle)
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise KeyError(f"Missing required metadata fields: {missing}")
    return metadata


def load_metadata_sg_config(path: str | Path = DEFAULT_METADATA_SG_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Metadata SG config must contain a YAML mapping.")
    return config.get("metadata_sg", config)


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def compute_solar_vectors_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    elevation_deg, azimuth_deg = get_solar_angles(
        longitude=float(metadata["longitude"]),
        latitude=float(metadata["latitude"]),
        timestamp=str(metadata["timestamp"]),
        timezone=str(metadata["timezone"]),
    )
    if elevation_deg <= 0:
        raise ValueError(
            "Solar-only SG direction is unavailable because the sun is below the "
            f"horizon (elevation_deg={elevation_deg:.4f})."
        )

    sun_vector_world = solar_angles_to_world_vector(elevation_deg, azimuth_deg)
    sun_vector_camera = world_to_camera_vector(
        vec_world=sun_vector_world,
        heading_deg=float(metadata["camera_heading"]),
        pitch_deg=float(metadata.get("camera_pitch", 0.0)),
        roll_deg=float(metadata.get("camera_roll", 0.0)),
    )

    return {
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "sun_vector_world": np.asarray(sun_vector_world, dtype=float).tolist(),
        "sun_vector_camera": np.asarray(sun_vector_camera, dtype=float).tolist(),
    }


def _offline_weather_values(
    metadata_config: dict[str, Any],
    cloudiness: float | None,
    visibility: float | None,
) -> dict[str, Any]:
    defaults = metadata_config.get("offline_defaults", {})
    return {
        "cloudiness": float(defaults.get("cloudiness", 20) if cloudiness is None else cloudiness),
        "visibility": float(defaults.get("visibility", 10000) if visibility is None else visibility),
        "weather_main": str(defaults.get("weather_main", "Clear")),
        "weather_description": str(defaults.get("weather_description", "manual/offline")),
    }


def build_weather_features_from_metadata(
    metadata: dict[str, Any],
    weather_config: dict[str, Any],
    metadata_config: dict[str, Any],
    offline: bool,
    cloudiness: float | None = None,
    visibility: float | None = None,
) -> dict[str, Any]:
    if offline:
        parsed = _offline_weather_values(metadata_config, cloudiness, visibility)
    else:
        parsed = fetch_current_weather(float(metadata["latitude"]), float(metadata["longitude"]), weather_config)

    return build_weather_features(
        sample_id=str(metadata["sample_id"]),
        cloudiness=parsed["cloudiness"],
        visibility=parsed["visibility"],
        weather_main=parsed["weather_main"],
        weather_description=parsed["weather_description"],
        config=weather_config,
    )


def generate_noaa_weather_sg_from_metadata(
    metadata_path: str | Path,
    weather_config_path: str | Path = DEFAULT_WEATHER_CONFIG,
    offline: bool = False,
    cloudiness: float | None = None,
    visibility: float | None = None,
    output_dir: str | Path = "data/intermediate/sg_params",
    metadata_sg_config_path: str | Path = DEFAULT_METADATA_SG_CONFIG,
) -> tuple[dict[str, Any], Path]:
    metadata_config = load_metadata_sg_config(metadata_sg_config_path)
    metadata = load_metadata(metadata_path)
    weather_config = load_weather_config(weather_config_path)
    solar = compute_solar_vectors_from_metadata(metadata)
    weather = build_weather_features_from_metadata(
        metadata=metadata,
        weather_config=weather_config,
        metadata_config=metadata_config,
        offline=offline,
        cloudiness=cloudiness,
        visibility=visibility,
    )

    sg_params = weather_to_sg_params(
        sample_id=str(metadata["sample_id"]),
        sun_vector=solar["sun_vector_camera"],
        weather_features=weather,
        config=weather_config,
    )
    sg_params.update(
        {
            "source": SOURCE_NAME,
            "coordinate_frame": "camera",
            "metadata": metadata,
            "solar": solar,
        }
    )

    output_root = _resolve_project_path(output_dir)
    output_path = output_root / f"{metadata['sample_id']}.json"
    save_sg_params_json(output_path, sg_params)
    return sg_params, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate camera-space NOAA + weather-aware SG params from metadata JSON.")
    parser.add_argument("--metadata", required=True, help="Path to image metadata JSON.")
    parser.add_argument("--weather-config", default=str(DEFAULT_WEATHER_CONFIG), help="Path to weather_config.yaml.")
    parser.add_argument("--offline", action="store_true", help="Use offline cloudiness/visibility instead of OpenWeatherMap.")
    parser.add_argument("--cloudiness", type=float, default=None, help="Offline cloudiness percentage.")
    parser.add_argument("--visibility", type=float, default=None, help="Offline visibility in meters.")
    parser.add_argument("--output-dir", default="data/intermediate/sg_params", help="Output directory for SG JSON.")
    parser.add_argument("--metadata-sg-config", default=str(DEFAULT_METADATA_SG_CONFIG), help="Path to metadata_sg_config.yaml.")
    args = parser.parse_args()

    sg_params, output_path = generate_noaa_weather_sg_from_metadata(
        metadata_path=args.metadata,
        weather_config_path=args.weather_config,
        offline=args.offline,
        cloudiness=args.cloudiness,
        visibility=args.visibility,
        output_dir=args.output_dir,
        metadata_sg_config_path=args.metadata_sg_config,
    )
    direct = sg_params["sg_lobes"][0]
    print(f"Saved NOAA weather SG params: {output_path}")
    print(
        "SG summary: "
        f"frame={sg_params['coordinate_frame']} mode={sg_params['weather']['weather_mode']} "
        f"direct_mu={np.round(direct['mu'], 4).tolist()}"
    )


if __name__ == "__main__":
    main()
