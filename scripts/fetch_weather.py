from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_b_weather import (
    build_weather_features,
    fetch_current_weather,
    load_weather_config,
)
from modules.module_b_weather_to_sg import save_weather_features_json


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "weather_config.yaml"


def _resolve_output_dir(config_path: str | Path, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parents[1] / path).resolve()


def fetch_or_build_weather_features(
    config_path: str | Path,
    sample_id: str,
    lat: float,
    lon: float,
    offline: bool = False,
    cloudiness: float | None = None,
    visibility: float | None = None,
    weather_main: str = "Clear",
    weather_description: str = "manual/offline",
) -> tuple[dict, Path]:
    config = load_weather_config(config_path)

    if offline:
        parsed = {
            "cloudiness": float(config.get("default_cloudiness", 20) if cloudiness is None else cloudiness),
            "visibility": float(config.get("default_visibility", 10000) if visibility is None else visibility),
            "weather_main": weather_main,
            "weather_description": weather_description,
        }
    else:
        parsed = fetch_current_weather(lat, lon, config)

    features = build_weather_features(
        sample_id=sample_id,
        cloudiness=parsed["cloudiness"],
        visibility=parsed["visibility"],
        weather_main=parsed["weather_main"],
        weather_description=parsed["weather_description"],
        config=config,
    )

    output_dir = _resolve_output_dir(config_path, config.get("output_weather_dir", "data/intermediate/weather"))
    output_path = output_dir / f"{sample_id}.json"
    save_weather_features_json(output_path, features)
    return features, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or build OpenWeatherMap weather features.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to weather_config.yaml.")
    parser.add_argument("--sample-id", required=True, help="Sample id for output JSON.")
    parser.add_argument("--lat", required=True, type=float, help="Latitude.")
    parser.add_argument("--lon", required=True, type=float, help="Longitude.")
    parser.add_argument("--offline", action="store_true", help="Use manual/default weather values instead of API.")
    parser.add_argument("--cloudiness", type=float, default=None, help="Offline cloudiness percentage.")
    parser.add_argument("--visibility", type=float, default=None, help="Offline visibility in meters.")
    parser.add_argument("--weather-main", default="Clear", help="Offline weather main label.")
    parser.add_argument("--weather-description", default="manual/offline", help="Offline weather description.")
    args = parser.parse_args()

    features, output_path = fetch_or_build_weather_features(
        config_path=args.config,
        sample_id=args.sample_id,
        lat=args.lat,
        lon=args.lon,
        offline=args.offline,
        cloudiness=args.cloudiness,
        visibility=args.visibility,
        weather_main=args.weather_main,
        weather_description=args.weather_description,
    )
    print(f"Saved weather features: {output_path}")
    print(
        "Weather summary: "
        f"mode={features['weather_mode']} cloudiness={features['cloudiness']} "
        f"visibility={features['visibility']}"
    )


if __name__ == "__main__":
    main()
