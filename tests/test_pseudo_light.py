import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from scripts.generate_pseudo_light import build_pseudo_light_record, generate_pseudo_light
from scripts.generate_sg_params import generate_sg_params
from scripts.generate_shadow_sketch import generate_shadow_sketches
from scripts.inspect_dataset import inspect_dataset


def save_mask(path: Path, box: tuple[int, int, int, int], size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask).save(path)


def save_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (127, 127, 127)).save(path)


def write_config(config_path: Path, root: Path, intermediate: Path) -> None:
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
        },
        "intermediate": {
            "pseudo_light_dir": str(intermediate / "pseudo_light"),
            "sg_params_dir": str(intermediate / "sg_params"),
            "shadow_sketch_dir": str(intermediate / "shadow_sketch"),
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def create_sample(root: Path, sample_id: str, offset: int = 0) -> None:
    save_image(root / "composite" / f"{sample_id}.png")
    save_image(root / "target" / f"{sample_id}.png")
    save_mask(root / "object_mask" / f"{sample_id}.png", (8, 8 + offset, 12, 12 + offset))
    save_mask(root / "shadow_mask" / f"{sample_id}.png", (18, 8 + offset, 24, 12 + offset))


class PseudoLightTestCase(unittest.TestCase):
    def test_shadow_and_light_directions_are_opposites(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            create_sample(root, "0001")
            sample = {
                "sample_id": "0001",
                "object_mask_path": root / "object_mask" / "0001.png",
                "shadow_mask_path": root / "shadow_mask" / "0001.png",
            }

            record = build_pseudo_light_record(sample)

            shadow = np.asarray(record["shadow_direction_2d"], dtype=np.float64)
            light = np.asarray(record["light_direction_2d"], dtype=np.float64)
            self.assertTrue(np.allclose(shadow, -light))
            self.assertGreaterEqual(record["confidence"], 0.0)
            self.assertLessEqual(record["confidence"], 1.0)

    def test_minimal_fixture_pipeline_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            intermediate = tmp_path / "data" / "intermediate"
            config_path = tmp_path / "configs" / "dataset_config.yaml"
            write_config(config_path, root, intermediate)
            create_sample(root, "0002", offset=0)
            create_sample(root, "0001", offset=4)

            summary = inspect_dataset(config_path)
            pseudo_outputs = generate_pseudo_light(config_path)
            sg_outputs = generate_sg_params(config_path)
            sketch_outputs = generate_shadow_sketches(config_path)

            self.assertEqual(summary["valid_samples"], 2)
            self.assertEqual(len(pseudo_outputs), 2)
            self.assertEqual(len(sg_outputs), 2)
            self.assertEqual(len(sketch_outputs), 2)

            for sample_id in ["0001", "0002"]:
                self.assertTrue((intermediate / "pseudo_light" / f"{sample_id}.json").exists())
                self.assertTrue((intermediate / "sg_params" / f"{sample_id}.json").exists())
                self.assertTrue((intermediate / "shadow_sketch" / f"{sample_id}.png").exists())

            with open(intermediate / "sg_params" / "0001.json", "r", encoding="utf-8") as handle:
                sg_record = json.load(handle)
            self.assertEqual(len(sg_record["sg_lobes"]), 2)


if __name__ == "__main__":
    unittest.main()
