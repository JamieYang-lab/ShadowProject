from __future__ import annotations

from typing import Any

import cv2
import numpy as np


RNG_SEED = 1337


def _config_section(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("sg_shadow_prior", config)


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def resize_mask(mask, size: int | tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    target = (int(size), int(size)) if isinstance(size, int) else (int(size[1]), int(size[0]))
    resized = cv2.resize((array > 0).astype(np.uint8), target, interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def resize_point_map(point_map, size: int | tuple[int, int]) -> np.ndarray:
    array = np.asarray(point_map, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("point_map must have shape [H, W, 3].")
    target = (int(size), int(size)) if isinstance(size, int) else (int(size[1]), int(size[0]))
    return cv2.resize(array, target, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def normalize_vector(v) -> np.ndarray:
    vector = np.asarray(v, dtype=np.float64)
    if vector.shape[0] != 3:
        raise ValueError("Vector must contain exactly 3 values.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero-length vector.")
    return (vector / norm).astype(np.float32)


def validate_sg_lobes(sg_lobes: list[dict]) -> None:
    if not isinstance(sg_lobes, list) or len(sg_lobes) < 2:
        raise ValueError("sg_lobes must contain at least direct and diffuse lobes.")
    for lobe in sg_lobes:
        if "type" not in lobe or "mu" not in lobe or "lambda" not in lobe or "amplitude" not in lobe:
            raise ValueError("Each SG lobe must contain type, mu, lambda, and amplitude.")
        normalize_vector(lobe["mu"])
        if float(lobe["lambda"]) <= 0:
            raise ValueError("SG lobe lambda must be positive.")
        if float(lobe["amplitude"]) < 0:
            raise ValueError("SG lobe amplitude must be non-negative.")


def extract_direct_and_diffuse_lobes(sg_lobes: list[dict]) -> tuple[dict, dict]:
    validate_sg_lobes(sg_lobes)
    direct = next((lobe for lobe in sg_lobes if lobe.get("type") == "direct"), sg_lobes[0])
    diffuse = next((lobe for lobe in sg_lobes if lobe.get("type") == "diffuse"), sg_lobes[1])
    return direct, diffuse


def _normalize_prior(prior: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float32)
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    prior = np.clip(prior, 0.0, None)
    max_value = float(prior.max())
    if max_value > 0.0:
        prior = prior / max_value
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def _deterministic_subsample(indices: np.ndarray, max_points: int) -> np.ndarray:
    if len(indices) <= max_points:
        return indices
    rng = np.random.default_rng(RNG_SEED)
    selected = rng.choice(len(indices), size=int(max_points), replace=False)
    return indices[np.sort(selected)]


def _blur_for_lambda(lambda_value: float, config: dict[str, Any]) -> int:
    hard_kernel = _odd_kernel(config.get("hard_blur_kernel", 3))
    soft_kernel = _odd_kernel(config.get("soft_blur_kernel", 31))
    hard_threshold = float(config.get("lambda_hard_threshold", 80.0))
    soft_threshold = float(config.get("lambda_soft_threshold", 5.0))
    lambda_value = float(lambda_value)
    if lambda_value >= hard_threshold:
        return hard_kernel
    if lambda_value <= soft_threshold:
        return soft_kernel
    t = (lambda_value - soft_threshold) / max(hard_threshold - soft_threshold, 1e-8)
    kernel = int(round(soft_kernel + t * (hard_kernel - soft_kernel)))
    return _odd_kernel(kernel)


def _finite_point_mask(point_map: np.ndarray) -> np.ndarray:
    return np.isfinite(point_map).all(axis=-1)


def compute_direct_shadow_prior(
    object_mask,
    point_map,
    direct_lobe: dict,
    config: dict[str, Any] | None,
    receiver_mask=None,
) -> np.ndarray:
    cfg = _config_section(config)
    coarse_size = int(cfg.get("coarse_size", 64))
    min_object_area = int(cfg.get("min_object_area", 4))
    object_binary = resize_mask(object_mask, coarse_size)
    point_map_coarse = resize_point_map(point_map, coarse_size)
    finite_mask = _finite_point_mask(point_map_coarse)

    if int(object_binary.sum()) < min_object_area:
        raise ValueError(f"Object mask area is too small: {int(object_binary.sum())} < {min_object_area}")

    receiver_binary = np.ones((coarse_size, coarse_size), dtype=bool)
    if receiver_mask is not None:
        receiver_binary = resize_mask(receiver_mask, coarse_size)
    receiver_binary &= ~object_binary
    receiver_binary &= finite_mask

    object_indices = np.argwhere(object_binary & finite_mask)
    receiver_indices = np.argwhere(receiver_binary)
    if len(object_indices) == 0 or len(receiver_indices) == 0:
        return np.zeros((coarse_size, coarse_size), dtype=np.float32)

    object_indices = _deterministic_subsample(object_indices, int(cfg.get("max_object_points", 512)))
    receiver_indices = _deterministic_subsample(receiver_indices, int(cfg.get("max_receiver_points", 4096)))
    object_points = point_map_coarse[object_indices[:, 0], object_indices[:, 1]].astype(np.float32)
    receiver_points = point_map_coarse[receiver_indices[:, 0], receiver_indices[:, 1]].astype(np.float32)

    mu = normalize_vector(direct_lobe["mu"])
    direction = -mu
    tau = np.deg2rad(float(cfg.get("angular_tolerance_deg", 5.0)))
    tan_tau_sq = float(np.tan(tau) ** 2)
    min_mu_z = float(cfg.get("min_mu_z", 0.1))
    max_distance_base = float(cfg.get("max_forward_distance_base", 3.0))
    forward_limit = max_distance_base / max(abs(float(mu[2])), min_mu_z)

    prior = np.zeros((coarse_size, coarse_size), dtype=np.float32)
    chunk_size = 128
    for start in range(0, len(object_points), chunk_size):
        object_chunk = object_points[start : start + chunk_size]
        vectors = receiver_points[None, :, :] - object_chunk[:, None, :]
        p = np.einsum("nmc,c->nm", vectors, direction)
        dist2 = np.einsum("nmc,nmc->nm", vectors, vectors)
        q2 = np.maximum(dist2 - p * p, 0.0)
        candidates = (p > 0.0) & (p <= forward_limit) & (q2 <= tan_tau_sq * p * p)
        if candidates.any():
            candidate_strength = np.where(candidates, 1.0 / (1.0 + np.maximum(p, 0.0)), 0.0)
            receiver_strength = candidate_strength.max(axis=0)
            hit_indices = receiver_indices[receiver_strength > 0.0]
            prior[hit_indices[:, 0], hit_indices[:, 1]] = np.maximum(
                prior[hit_indices[:, 0], hit_indices[:, 1]],
                receiver_strength[receiver_strength > 0.0],
            )

    prior *= float(direct_lobe.get("amplitude", 1.0))
    kernel = _blur_for_lambda(float(direct_lobe["lambda"]), cfg)
    if kernel > 1:
        prior = cv2.GaussianBlur(prior, (kernel, kernel), 0)

    if receiver_mask is not None:
        prior *= resize_mask(receiver_mask, coarse_size).astype(np.float32)

    return _normalize_prior(prior) if bool(cfg.get("normalize_output", True)) else np.clip(prior, 0.0, 1.0).astype(np.float32)


def apply_diffuse_modulation(direct_prior: np.ndarray, diffuse_lobe: dict, config: dict[str, Any] | None) -> np.ndarray:
    cfg = _config_section(config)
    direct = np.asarray(direct_prior, dtype=np.float32)
    diffuse_amplitude = float(diffuse_lobe.get("amplitude", 0.0))
    diffuse_lambda = float(diffuse_lobe.get("lambda", 5.0))

    attenuation = 1.0 / (1.0 + diffuse_amplitude)
    soft_kernel = _blur_for_lambda(diffuse_lambda, cfg)
    ambient_kernel = max(soft_kernel, _odd_kernel(cfg.get("soft_blur_kernel", 31)))
    ambient = cv2.GaussianBlur(direct, (ambient_kernel, ambient_kernel), 0) if ambient_kernel > 1 else direct.copy()
    if float(ambient.max()) > 0.0:
        ambient = ambient / float(ambient.max())

    diffuse_weight = diffuse_amplitude / (1.0 + diffuse_amplitude)
    combined = direct * attenuation + ambient * diffuse_weight * 0.55
    combined_kernel = max(1, int(round(soft_kernel * diffuse_weight)))
    combined_kernel = _odd_kernel(combined_kernel)
    if combined_kernel > 1:
        combined = cv2.GaussianBlur(combined, (combined_kernel, combined_kernel), 0)
    return _normalize_prior(combined) if bool(cfg.get("normalize_output", True)) else np.clip(combined, 0.0, 1.0).astype(np.float32)


def compute_sg_shadow_prior(
    object_mask,
    point_map,
    sg_lobes: list[dict],
    config: dict[str, Any] | None,
    receiver_mask=None,
    return_debug: bool = False,
) -> dict[str, Any]:
    direct_lobe, diffuse_lobe = extract_direct_and_diffuse_lobes(sg_lobes)
    cfg = _config_section(config)
    direct = compute_direct_shadow_prior(object_mask, point_map, direct_lobe, cfg, receiver_mask)
    diffuse_modulated = apply_diffuse_modulation(direct, diffuse_lobe, cfg)
    combined = diffuse_modulated
    if receiver_mask is not None:
        combined *= resize_mask(receiver_mask, int(cfg.get("coarse_size", 64))).astype(np.float32)
        combined = _normalize_prior(combined) if bool(cfg.get("normalize_output", True)) else combined

    result: dict[str, Any] = {
        "combined": combined.astype(np.float32),
        "direct": direct.astype(np.float32),
        "metadata": {
            "direct_mu": normalize_vector(direct_lobe["mu"]).astype(float).tolist(),
            "direct_lambda": float(direct_lobe["lambda"]),
            "direct_amplitude": float(direct_lobe["amplitude"]),
            "diffuse_lambda": float(diffuse_lobe["lambda"]),
            "diffuse_amplitude": float(diffuse_lobe["amplitude"]),
        },
    }
    if return_debug:
        result["diffuse_modulated"] = diffuse_modulated.astype(np.float32)
    return result
