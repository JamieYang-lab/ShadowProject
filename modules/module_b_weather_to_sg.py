from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_NAME = "noaa_openweather_rule_based"


def normalize_vector(vector: Iterable[float]) -> list[float]:
    values = np.asarray(list(vector), dtype=np.float64)
    if values.shape[0] != 3:
        raise ValueError("sun_vector must contain exactly 3 values.")
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (values / norm).astype(float).tolist()


def _clamp_non_negative(value: float) -> float:
    return max(0.0, float(value))


def _mode_rule(weather_features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    mode = weather_features.get("weather_mode", "cloudy")
    rules = config.get("sg_rules", {})
    if mode not in rules:
        return rules.get("cloudy", {})
    return rules[mode]


def _adjust_amplitudes(
    direct_amplitude: float,
    diffuse_amplitude: float,
    visibility: float,
    config: dict[str, Any],
) -> tuple[float, float]:
    visibility_thresholds = config.get("visibility_thresholds", {})
    medium_threshold = float(visibility_thresholds.get("medium", 3000))
    high_threshold = float(visibility_thresholds.get("high", 8000))
    adjustment = config.get("visibility_adjustment", {})

    direct = float(direct_amplitude)
    diffuse = float(diffuse_amplitude)
    visibility = float(visibility)

    if visibility < medium_threshold:
        direct *= float(adjustment.get("low_visibility_direct_scale", 0.5))
        diffuse += float(adjustment.get("low_visibility_diffuse_boost", 0.2))
    elif visibility < high_threshold:
        direct *= float(adjustment.get("medium_visibility_direct_scale", 0.8))
        diffuse += float(adjustment.get("medium_visibility_diffuse_boost", 0.1))

    return _clamp_non_negative(direct), _clamp_non_negative(diffuse)


def weather_to_ratios(weather_features: dict[str, Any]) -> tuple[float, float]:
    direct = _clamp_non_negative(float(weather_features.get("direct_ratio", 0.5)))
    diffuse = _clamp_non_negative(float(weather_features.get("diffuse_ratio", 0.5)))
    total = direct + diffuse
    if total == 0.0:
        return 0.5, 0.5
    return direct / total, diffuse / total


def weather_to_sg_params(
    sample_id: str,
    sun_vector: Iterable[float],
    weather_features: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rule = _mode_rule(weather_features, config)
    visibility = float(weather_features.get("visibility", config.get("default_visibility", 10000)))
    direct_amplitude, diffuse_amplitude = _adjust_amplitudes(
        float(rule.get("direct_amplitude", 0.6)),
        float(rule.get("diffuse_amplitude", 0.5)),
        visibility,
        config,
    )

    return {
        "sample_id": str(sample_id),
        "source": SOURCE_NAME,
        "weather": weather_features,
        "sg_lobes": [
            {
                "type": "direct",
                "mu": normalize_vector(sun_vector),
                "lambda": float(rule.get("direct_lambda", 50.0)),
                "amplitude": direct_amplitude,
            },
            {
                "type": "diffuse",
                "mu": [0.0, 0.0, 1.0],
                "lambda": float(rule.get("diffuse_lambda", 5.0)),
                "amplitude": diffuse_amplitude,
            },
        ],
    }


def save_weather_features_json(path: str | Path, features: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(features, handle, indent=2)
    return output_path


def save_sg_params_json(path: str | Path, sg_params: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(sg_params, handle, indent=2)
    return output_path
