import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from datasets.unet_shadow_dataset import UNetShadowDataset


SG_LOBES = [
    {"type": "direct", "mu": [1.0, 0.0, 0.0], "lambda": 80.0, "amplitude": 1.0},
    {"type": "diffuse", "mu": [0.0, 0.0, 1.0], "lambda": 5.0, "amplitude": 0.3},
]


def save_rgb(path: Path, color=(128, 64, 32), size=(24, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def save_mask(path: Path, box=(4, 4, 12, 12), size=(24, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask).save(path)


def write_dataset_config(path: Path, root: Path) -> None:
    config = {
        "dataset": {
            "name": "DESOBAv2",
            "root": str(root),
            "subdirs": {
                "composite": "composite",
                "target": "target",
                "object_mask": "object_mask",
                "shadow_mask": "shadow_mask",
            },
            "image_extensions": [".png"],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def create_sample(project_root: Path, sample_id: str = "0001", omit: set[str] | None = None) -> None:
    omit = omit or set()
    data_root = project_root / "data" / "desobav2"
    if "composite" not in omit:
        save_rgb(data_root / "composite" / f"{sample_id}.png")
    if "target" not in omit:
        save_rgb(data_root / "target" / f"{sample_id}.png")
    if "object_mask" not in omit:
        save_mask(data_root / "object_mask" / f"{sample_id}.png")
    if "shadow_mask" not in omit:
        save_mask(data_root / "shadow_mask" / f"{sample_id}.png", box=(12, 12, 20, 20))
    if "point_map" not in omit:
        point_map = np.random.default_rng(1).normal(size=(16, 16, 3)).astype(np.float32)
        path = project_root / "data" / "intermediate" / "point_maps" / f"{sample_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, point_map)
    if "sg_prior" not in omit:
        prior = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        path = project_root / "data" / "intermediate" / "sg_shadow_prior" / f"{sample_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, prior)
    if "sg_params" not in omit:
        path = project_root / "data" / "intermediate" / "sg_params" / f"{sample_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"sample_id": sample_id, "sg_lobes": SG_LOBES}, handle)


class UNetShadowDatasetTestCase(unittest.TestCase):
    def build_dataset(self, tmp_path: Path, strict: bool = True, size: int = 64) -> UNetShadowDataset:
        config_path = tmp_path / "configs" / "dataset_config.yaml"
        write_dataset_config(config_path, tmp_path / "data" / "desobav2")
        return UNetShadowDataset(config_path=config_path, size=size, strict=strict)

    def test_dataset_length_is_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path, "0001")
            create_sample(tmp_path, "0002")

            dataset = self.build_dataset(tmp_path)

            self.assertEqual(len(dataset), 2)

    def test_input_and_shadow_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path)
            item = self.build_dataset(tmp_path, size=64)[0]

            self.assertEqual(tuple(item["input"].shape), (18, 64, 64))
            self.assertEqual(tuple(item["shadow_mask"].shape), (1, 64, 64))

    def test_image_and_mask_values_are_in_expected_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path)
            item = self.build_dataset(tmp_path, size=64)[0]

            self.assertGreaterEqual(float(item["input"][0:4].min()), 0.0)
            self.assertLessEqual(float(item["input"][0:4].max()), 1.0)
            self.assertGreaterEqual(float(item["shadow_mask"].min()), 0.0)
            self.assertLessEqual(float(item["shadow_mask"].max()), 1.0)

    def test_sg_vector_shape_and_broadcast_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path)
            item = self.build_dataset(tmp_path, size=32)[0]

            self.assertEqual(tuple(item["sg_vector"].shape), (10,))
            sg_maps = item["input"][8:18]
            self.assertTrue(np.allclose(sg_maps[:, 0, 0].numpy(), item["sg_vector"].numpy()))
            self.assertTrue(np.allclose(sg_maps[:, -1, -1].numpy(), item["sg_vector"].numpy()))

    def test_strict_true_raises_if_point_map_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path, omit={"point_map"})

            with self.assertRaises(FileNotFoundError):
                self.build_dataset(tmp_path, strict=True)

    def test_strict_true_raises_if_sg_prior_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path, omit={"sg_prior"})

            with self.assertRaises(FileNotFoundError):
                self.build_dataset(tmp_path, strict=True)

    def test_strict_true_raises_if_sg_params_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path, omit={"sg_params"})

            with self.assertRaises(FileNotFoundError):
                self.build_dataset(tmp_path, strict=True)

    def test_strict_false_skips_invalid_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            create_sample(tmp_path, "valid")
            create_sample(tmp_path, "invalid", omit={"sg_params"})

            dataset = self.build_dataset(tmp_path, strict=False)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["sample_id"], "valid")


if __name__ == "__main__":
    unittest.main()
