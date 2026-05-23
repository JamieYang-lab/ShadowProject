from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


DEFAULT_CONFIG_PATH = Path("configs/dataset_config.yaml")


def load_dataset_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Dataset config must contain a YAML mapping.")

    return config


def _resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (base_dir / path).resolve()
    return path.resolve()


def _allowed_extensions(config: dict) -> set[str]:
    extensions = config.get("dataset", {}).get("image_extensions", [".png", ".jpg", ".jpeg"])
    return {str(ext).lower() for ext in extensions}


@dataclass(frozen=True)
class MissingFileRecord:
    sample_id: str
    field: str
    path: Path


class DESOBAv2Dataset:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH, strict: bool = True):
        self.config_path = Path(config_path)
        self.config = load_dataset_config(self.config_path)
        self.project_root = self.config_path.resolve().parents[1]
        self.strict = strict

        dataset_config = self.config.get("dataset")
        if not isinstance(dataset_config, dict):
            raise ValueError("Missing required 'dataset' section in dataset config.")

        subdirs = dataset_config.get("subdirs")
        if not isinstance(subdirs, dict):
            raise ValueError("Missing required 'dataset.subdirs' mapping.")

        self.root = _resolve_path(dataset_config.get("root", "data/desobav2"), self.project_root)
        self.subdirs = {
            "composite": str(subdirs.get("composite", "composite")),
            "target": str(subdirs.get("target", "target")),
            "object_mask": str(subdirs.get("object_mask", "object_mask")),
            "shadow_mask": str(subdirs.get("shadow_mask", "shadow_mask")),
        }
        self.extensions = _allowed_extensions(self.config)
        self.samples, self.missing_files = self._build_index(strict=strict)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

    def _field_dir(self, field: str) -> Path:
        return self.root / self.subdirs[field]

    def _list_ids(self, field: str) -> dict[str, Path]:
        directory = self._field_dir(field)
        if not directory.exists():
            return {}

        paths: dict[str, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in self.extensions:
                paths[path.stem] = path.resolve()
        return paths

    def _build_index(self, strict: bool) -> tuple[list[dict], list[MissingFileRecord]]:
        field_paths = {field: self._list_ids(field) for field in self.subdirs}
        all_ids = sorted(set().union(*(paths.keys() for paths in field_paths.values())))

        if not all_ids and strict and not self.root.exists():
            raise FileNotFoundError(f"DESOBAv2 root does not exist: {self.root}")

        samples: list[dict] = []
        missing_files: list[MissingFileRecord] = []

        for sample_id in all_ids:
            sample_paths: dict[str, Path] = {}
            missing_for_sample: list[MissingFileRecord] = []

            for field in self.subdirs:
                path = field_paths[field].get(sample_id)
                if path is None:
                    expected = self._field_dir(field) / f"{sample_id}.png"
                    missing_for_sample.append(MissingFileRecord(sample_id, field, expected.resolve()))
                else:
                    sample_paths[field] = path

            if missing_for_sample:
                missing_files.extend(missing_for_sample)
                if strict:
                    details = ", ".join(f"{item.field}: {item.path}" for item in missing_for_sample)
                    raise FileNotFoundError(f"Missing files for sample '{sample_id}': {details}")
                continue

            samples.append(
                {
                    "sample_id": sample_id,
                    "composite_path": sample_paths["composite"],
                    "target_path": sample_paths["target"],
                    "object_mask_path": sample_paths["object_mask"],
                    "shadow_mask_path": sample_paths["shadow_mask"],
                    "metadata": {
                        "dataset": "DESOBAv2",
                        "source": "file_stem_match",
                    },
                }
            )

        return samples, missing_files

    def count_missing_files(self) -> int:
        return len(self.missing_files)

    def missing_by_field(self) -> dict[str, int]:
        return summarize_missing_files(self.missing_files)


def summarize_missing_files(records: Iterable[MissingFileRecord]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        summary[record.field] = summary.get(record.field, 0) + 1
    return summary
