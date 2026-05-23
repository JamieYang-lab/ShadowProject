from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SGLobe:
    type: str
    mu: list[float]
    lambda_: float
    amplitude: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["lambda"] = data.pop("lambda_")
        return data


def normalize_vector(vector: Iterable[float]) -> list[float]:
    values = np.asarray(list(vector), dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (values / norm).astype(float).tolist()


def _clamp_amplitude(amplitude: float) -> float:
    return float(np.clip(float(amplitude), 0.0, 1.0))


def initialize_direct_lobe(
    light_direction: Iterable[float],
    sharpness: float = 80.0,
    amplitude: float = 1.0,
) -> SGLobe:
    if sharpness <= 0:
        raise ValueError("Direct lobe sharpness must be positive.")

    return SGLobe(
        type="direct",
        mu=normalize_vector(light_direction),
        lambda_=float(sharpness),
        amplitude=_clamp_amplitude(amplitude),
    )


def initialize_diffuse_lobe(
    direction: Iterable[float] = (0.0, 0.0, 1.0),
    sharpness: float = 5.0,
    amplitude: float = 0.3,
) -> SGLobe:
    if sharpness <= 0:
        raise ValueError("Diffuse lobe sharpness must be positive.")

    return SGLobe(
        type="diffuse",
        mu=normalize_vector(direction),
        lambda_=float(sharpness),
        amplitude=_clamp_amplitude(amplitude),
    )


def initialize_sg_from_light_direction(
    light_direction: Iterable[float],
    direct_sharpness: float = 80.0,
    direct_amplitude: float = 1.0,
    diffuse_sharpness: float = 5.0,
    diffuse_amplitude: float = 0.3,
) -> list[SGLobe]:
    return [
        initialize_direct_lobe(light_direction, direct_sharpness, direct_amplitude),
        initialize_diffuse_lobe((0.0, 0.0, 1.0), diffuse_sharpness, diffuse_amplitude),
    ]


def flatten_sg_lobes(lobes: Iterable[SGLobe | dict]) -> list[float]:
    flattened: list[float] = []
    for lobe in lobes:
        if isinstance(lobe, SGLobe):
            mu = lobe.mu
            lambda_value = lobe.lambda_
            amplitude = lobe.amplitude
        else:
            mu = lobe["mu"]
            lambda_value = lobe["lambda"]
            amplitude = lobe["amplitude"]

        if len(mu) != 3:
            raise ValueError("Each SG lobe direction must contain 3 values.")

        flattened.extend([float(mu[0]), float(mu[1]), float(mu[2]), float(lambda_value), float(amplitude)])

    return flattened
