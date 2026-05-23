import sys
from pathlib import Path
import re
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pvlib
import yaml

from utils.math_utils import (
    normalize,
    deg2rad,
    world_to_camera_vector
)


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    return config


def _require_mapping(config: dict, key: str) -> dict:
    if key not in config:
        raise KeyError(f"Missing required config section: '{key}'")

    value = config[key]
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")

    return value


def _require_value(section: dict, section_name: str, key: str):
    if key not in section:
        raise KeyError(f"Missing required config field: '{section_name}.{key}'")
    return section[key]


def _require_float(section: dict, section_name: str, key: str) -> float:
    value = _require_value(section, section_name, key)

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field '{section_name}.{key}' must be numeric.") from exc


def _require_timestamp(timestamp: str) -> str:
    if not isinstance(timestamp, str):
        raise ValueError("Config field 'capture.timestamp' must be a string.")

    if not TIMESTAMP_REGEX.fullmatch(timestamp):
        raise ValueError(
            "Config field 'capture.timestamp' must match 'YYYY-MM-DD HH:MM:SS' "
            "without any timezone offset."
        )

    try:
        datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(
            "Config field 'capture.timestamp' must be a valid local time in "
            "'YYYY-MM-DD HH:MM:SS' format."
        ) from exc

    return timestamp


def _require_timezone(timezone: str) -> str:
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("Config field 'capture.timezone' must be a non-empty string.")

    try:
        pd.Timestamp("2024-01-01 00:00:00").tz_localize(timezone)
    except Exception as exc:
        raise ValueError(
            f"Config field 'capture.timezone' is not a valid timezone: '{timezone}'"
        ) from exc

    return timezone


def validate_config_contract(config: dict) -> dict:
    location = _require_mapping(config, "location")
    capture = _require_mapping(config, "capture")
    camera = _require_mapping(config, "camera")

    validated = {
        "location": {
            "longitude": _require_float(location, "location", "longitude"),
            "latitude": _require_float(location, "location", "latitude"),
        },
        "capture": {
            "timestamp": _require_timestamp(_require_value(capture, "capture", "timestamp")),
            "timezone": _require_timezone(_require_value(capture, "capture", "timezone")),
        },
        "camera": {
            "heading": _require_float(camera, "camera", "heading"),
            "pitch": float(camera.get("pitch", 0.0)),
            "roll": float(camera.get("roll", 0.0)),
        },
    }

    return validated


def get_solar_angles(longitude: float,
                     latitude: float,
                     timestamp: str,
                     timezone: str):
    localized_time = pd.Timestamp(timestamp).tz_localize(timezone)
    time = pd.DatetimeIndex([localized_time])

    solar_pos = pvlib.solarposition.get_solarposition(
        time,
        latitude=latitude,
        longitude=longitude
    )

    azimuth_deg = float(solar_pos["azimuth"].iloc[0])
    elevation_deg = float(solar_pos["apparent_elevation"].iloc[0])

    return elevation_deg, azimuth_deg


def solar_angles_to_world_vector(elevation_deg: float,
                                 azimuth_deg: float) -> np.ndarray:
    """
    ENU:
        x = East
        y = North
        z = Up

    sun_vec_world = 場景 -> 太陽
    """
    a = deg2rad(azimuth_deg)
    e = deg2rad(elevation_deg)

    sun_vec_world = np.array([
        np.cos(e) * np.sin(a),
        np.cos(e) * np.cos(a),
        np.sin(e)
    ], dtype=np.float64)

    return normalize(sun_vec_world)


def sun_world_to_light_world(sun_vec_world: np.ndarray) -> np.ndarray:
    """
    light_vec_world = 太陽 → 地面
    """
    return normalize(-sun_vec_world)


def world_to_camera_light_vector(light_vec_world: np.ndarray,
                                 heading_deg: float,
                                 pitch_deg: float,
                                 roll_deg: float) -> np.ndarray:
    return world_to_camera_vector(
        vec_world=light_vec_world,
        heading_deg=heading_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg
    )


def run_light_engine(config_path: str) -> dict:
    raw_config = load_config(config_path)
    config = validate_config_contract(raw_config)

    longitude = config["location"]["longitude"]
    latitude = config["location"]["latitude"]
    timestamp = config["capture"]["timestamp"]
    timezone = config["capture"]["timezone"]

    heading = config["camera"]["heading"]
    pitch = config["camera"].get("pitch", 0.0)
    roll = config["camera"].get("roll", 0.0)

    elevation_deg, azimuth_deg = get_solar_angles(
        longitude,
        latitude,
        timestamp,
        timezone
    )

    if elevation_deg <= 0:
        raise ValueError(
            "Solar-only light direction is unavailable because the sun is below the "
            f"horizon (elevation_deg={elevation_deg:.4f})."
        )

    sun_vec_world = solar_angles_to_world_vector(
        elevation_deg,
        azimuth_deg
    )

    light_vec_world = sun_world_to_light_world(sun_vec_world)

    light_vec_camera = world_to_camera_light_vector(
        light_vec_world,
        heading,
        pitch,
        roll
    )

    return {
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
        "sun_vec_world": sun_vec_world,
        "light_vec_world": light_vec_world,
        "light_vec_camera": light_vec_camera,
        "light_source_type": "solar",
        "solar_provider": "pvlib",
    }


if __name__ == "__main__":
    result = run_light_engine("configs/config.yaml")

    print("=== Light Engine Result ===")
    print(f"elevation_deg   : {result['elevation_deg']:.4f}")
    print(f"azimuth_deg     : {result['azimuth_deg']:.4f}")
    print(f"sun_vec_world   : {result['sun_vec_world']}")
    print(f"light_vec_world : {result['light_vec_world']}")
    print(f"light_vec_camera: {result['light_vec_camera']}")
