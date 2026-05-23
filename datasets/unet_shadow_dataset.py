from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.desobav2_dataset import DESOBAv2Dataset
from modules.module_b_sg_light import flatten_sg_lobes


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(image, dtype=np.float32) > 0).astype(np.float32)


def _load_sg_prior(path_base: Path, size: int) -> np.ndarray:
    npy_path = path_base.with_suffix(".npy")
    png_path = path_base.with_suffix(".png")
    if npy_path.exists():
        prior = np.load(npy_path).astype(np.float32)
    elif png_path.exists():
        with Image.open(png_path) as image:
            prior = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    else:
        raise FileNotFoundError(f"Missing SG shadow prior: {npy_path} or {png_path}")

    if prior.ndim != 2:
        raise ValueError("SG shadow prior must have shape [H, W].")
    prior = cv2.resize(prior, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.clip(prior.astype(np.float32), 0.0, 1.0)


def _robust_normalize_point_map(point_map: np.ndarray, size: int) -> np.ndarray:
    point_map = np.asarray(point_map, dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError("point_map must have shape [H, W, 3].")
    point_map = cv2.resize(point_map, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    normalized = np.zeros_like(point_map, dtype=np.float32)
    for channel in range(3):
        values = point_map[..., channel]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        valid = values[finite]
        median = float(np.median(valid))
        q1, q3 = np.percentile(valid, [25, 75])
        scale = float(q3 - q1)
        if scale < 1e-6:
            scale = float(np.std(valid))
        if scale < 1e-6:
            scale = 1.0
        channel_values = np.nan_to_num((values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        normalized[..., channel] = np.clip(channel_values, -5.0, 5.0) / 5.0
    return normalized.astype(np.float32)


def _load_sg_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing SG params: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    if "sg_lobes" not in record:
        raise ValueError(f"SG params JSON missing 'sg_lobes': {path}")
    vector = np.asarray(flatten_sg_lobes(record["sg_lobes"]), dtype=np.float32)
    if vector.shape != (10,):
        raise ValueError(f"Expected K=2 SG vector with shape [10], got {vector.shape}.")
    return vector


class UNetShadowDataset(Dataset):
    def __init__(self, config_path: str | Path = "configs/dataset_config.yaml", size: int = 512, strict: bool = True, limit: int | None = None):
        self.config_path = Path(config_path)
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("size must be positive.")
        self.base_dataset = DESOBAv2Dataset(config_path, strict=False)
        self.project_root = self.base_dataset.project_root
        self.strict = strict
        self.samples = self._build_samples(strict=strict)
        if limit is not None:
            self.samples = self.samples[: max(0, int(limit))]

    def _path_bundle(self, sample: dict[str, Any]) -> dict[str, Path]:
        sample_id = sample["sample_id"]
        return {
            **sample,
            "point_map_path": _resolve_project_path(self.project_root, f"data/intermediate/point_maps/{sample_id}.npy"),
            "sg_prior_base": _resolve_project_path(self.project_root, f"data/intermediate/sg_shadow_prior/{sample_id}"),
            "sg_params_path": _resolve_project_path(self.project_root, f"data/intermediate/sg_params/{sample_id}.json"),
            "receiver_mask_path": _resolve_project_path(self.project_root, f"data/intermediate/receiver_masks/{sample_id}.png"),
        }

    def _validate_bundle(self, sample: dict[str, Any]) -> None:
        required_paths = [
            sample["composite_path"],
            sample["object_mask_path"],
            sample["shadow_mask_path"],
            sample["point_map_path"],
            sample["sg_params_path"],
        ]
        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing required UNet dataset file: {path}")
        if not sample["sg_prior_base"].with_suffix(".npy").exists() and not sample["sg_prior_base"].with_suffix(".png").exists():
            raise FileNotFoundError(f"Missing SG shadow prior for sample: {sample['sample_id']}")

    def _build_samples(self, strict: bool) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        for sample in self.base_dataset:
            bundle = self._path_bundle(sample)
            try:
                self._validate_bundle(bundle)
            except (FileNotFoundError, ValueError):
                if strict:
                    raise
                continue
            valid.append(bundle)
        return valid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        size = self.size

        composite = _load_rgb(sample["composite_path"], size)
        object_mask = _load_mask(sample["object_mask_path"], size)
        shadow_mask = _load_mask(sample["shadow_mask_path"], size)
        point_map = _robust_normalize_point_map(np.load(sample["point_map_path"]), size)
        sg_prior = _load_sg_prior(sample["sg_prior_base"], size)
        sg_vector = _load_sg_vector(sample["sg_params_path"])
        sg_maps = np.broadcast_to(sg_vector[:, None, None], (10, size, size)).astype(np.float32)

        channels = [
            np.transpose(composite, (2, 0, 1)),
            object_mask[None, ...],
            np.transpose(point_map, (2, 0, 1)),
            sg_prior[None, ...],
            sg_maps,
        ]
        model_input = np.concatenate(channels, axis=0).astype(np.float32)

        return {
            "sample_id": sample["sample_id"],
            "input": torch.from_numpy(model_input),
            "shadow_mask": torch.from_numpy(shadow_mask[None, ...].astype(np.float32)),
            "sg_vector": torch.from_numpy(sg_vector.astype(np.float32)),
            "receiver_mask_path": str(sample["receiver_mask_path"]),
        }
