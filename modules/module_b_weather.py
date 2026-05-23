from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import yaml


DEFAULT_WEATHER_CONFIG = Path("configs/weather_config.yaml")


def load_weather_config(path: str | Path = DEFAULT_WEATHER_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Weather config must contain a YAML mapping.")
    return config


def get_api_key(config: dict[str, Any]) -> str:
    env_name = config.get("api_key_env", "OPENWEATHER_API_KEY")
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(
            f"Missing OpenWeatherMap API key. Set environment variable `{env_name}`. "
            "Do not store API keys in config files or commit them to git."
        )
    return api_key


def fetch_current_weather(lat: float, lon: float, config: dict[str, Any]) -> dict[str, Any]:
    api_key = get_api_key(config)
    params = {
        "lat": float(lat),
        "lon": float(lon),
        "appid": api_key,
        "units": config.get("units", "metric"),
    }
    response = requests.get(config.get("api_base_url"), params=params, timeout=20)
    response.raise_for_status()
    return parse_openweather_response(response.json())


def parse_openweather_response(response_json: dict[str, Any]) -> dict[str, Any]:
    weather_items = response_json.get("weather") or [{}]
    weather = weather_items[0] if weather_items else {}
    return {
        "cloudiness": float((response_json.get("clouds") or {}).get("all", 0.0)),
        "visibility": float(response_json.get("visibility", 0.0)),
        "weather_main": str(weather.get("main", "")),
        "weather_description": str(weather.get("description", "")),
    }


def classify_weather_mode(cloudiness: float, visibility: float, config: dict[str, Any]) -> str:
    thresholds = config.get("mode_thresholds", {})
    sunny_cloud_max = float(thresholds.get("sunny_cloud_max", 30))
    cloudy_cloud_max = float(thresholds.get("cloudy_cloud_max", 75))

    if float(cloudiness) < sunny_cloud_max:
        return "sunny"
    if float(cloudiness) < cloudy_cloud_max:
        return "cloudy"
    return "overcast"


def _base_ratios(weather_mode: str, config: dict[str, Any]) -> tuple[float, float]:
    rules = config.get("sg_rules", {})
    mode_rule = rules.get(weather_mode, rules.get("cloudy", {}))
    direct = float(mode_rule.get("direct_amplitude", 0.6))
    diffuse = float(mode_rule.get("diffuse_amplitude", 0.5))
    return _normalize_ratios(direct, diffuse)


def _normalize_ratios(direct: float, diffuse: float) -> tuple[float, float]:
    direct = max(0.0, float(direct))
    diffuse = max(0.0, float(diffuse))
    total = max(direct + diffuse, 1e-8)
    return direct / total, diffuse / total


def _visibility_adjusted_ratios(
    weather_mode: str,
    visibility: float,
    config: dict[str, Any],
) -> tuple[float, float]:
    rules = config.get("sg_rules", {})
    mode_rule = rules.get(weather_mode, rules.get("cloudy", {}))
    direct = float(mode_rule.get("direct_amplitude", 0.6))
    diffuse = float(mode_rule.get("diffuse_amplitude", 0.5))

    visibility_thresholds = config.get("visibility_thresholds", {})
    medium_threshold = float(visibility_thresholds.get("medium", 3000))
    high_threshold = float(visibility_thresholds.get("high", 8000))
    adjustment = config.get("visibility_adjustment", {})

    visibility = float(visibility)
    if visibility < medium_threshold:
        direct *= float(adjustment.get("low_visibility_direct_scale", 0.5))
        diffuse += float(adjustment.get("low_visibility_diffuse_boost", 0.2))
    elif visibility < high_threshold:
        direct *= float(adjustment.get("medium_visibility_direct_scale", 0.8))
        diffuse += float(adjustment.get("medium_visibility_diffuse_boost", 0.1))

    return _normalize_ratios(direct, diffuse)


def build_weather_features(
    sample_id: str,
    cloudiness: float,
    visibility: float,
    weather_main: str,
    weather_description: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    weather_mode = classify_weather_mode(cloudiness, visibility, config)
    direct_ratio, diffuse_ratio = _visibility_adjusted_ratios(weather_mode, visibility, config)
    return {
        "sample_id": str(sample_id),
        "cloudiness": float(cloudiness),
        "visibility": float(visibility),
        "weather_main": str(weather_main),
        "weather_description": str(weather_description),
        "weather_mode": weather_mode,
        "direct_ratio": float(direct_ratio),
        "diffuse_ratio": float(diffuse_ratio),
    }
