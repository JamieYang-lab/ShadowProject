import tempfile
import unittest
from pathlib import Path

from PIL import Image
import yaml

from datasets.desobav2_dataset import DESOBAv2Dataset


def write_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), value).save(path)


def write_dataset_config(config_path: Path, root: Path, intermediate_root: Path | None = None) -> None:
    intermediate_root = intermediate_root or root.parent / "intermediate"
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
            "pseudo_light_dir": str(intermediate_root / "pseudo_light"),
            "sg_params_dir": str(intermediate_root / "sg_params"),
            "shadow_sketch_dir": str(intermediate_root / "shadow_sketch"),
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


class DESOBAv2DatasetTestCase(unittest.TestCase):
    def create_sample(self, root: Path, sample_id: str, omit: set[str] | None = None) -> None:
        omit = omit or set()
        for field in ["composite", "target", "object_mask", "shadow_mask"]:
            if field not in omit:
                write_image(root / field / f"{sample_id}.png")

    def test_stable_sample_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            self.create_sample(root, "sample_b")
            self.create_sample(root, "sample_a")
            self.create_sample(root, "sample_c")
            config_path = tmp_path / "configs" / "dataset_config.yaml"
            write_dataset_config(config_path, root)

            dataset = DESOBAv2Dataset(config_path, strict=True)

            self.assertEqual([sample["sample_id"] for sample in dataset], ["sample_a", "sample_b", "sample_c"])

    def test_missing_file_strict_and_non_strict_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            self.create_sample(root, "valid")
            self.create_sample(root, "missing_shadow", omit={"shadow_mask"})
            config_path = tmp_path / "configs" / "dataset_config.yaml"
            write_dataset_config(config_path, root)

            with self.assertRaises(FileNotFoundError):
                DESOBAv2Dataset(config_path, strict=True)

            dataset = DESOBAv2Dataset(config_path, strict=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["sample_id"], "valid")
            self.assertEqual(dataset.count_missing_files(), 1)
            self.assertEqual(dataset.missing_by_field()["shadow_mask"], 1)

    def test_sample_paths_are_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            self.create_sample(root, "0001")
            config_path = tmp_path / "configs" / "dataset_config.yaml"
            write_dataset_config(config_path, root)

            sample = DESOBAv2Dataset(config_path, strict=True)[0]

            self.assertEqual(sample["object_mask_path"], (root / "object_mask" / "0001.png").resolve())
            self.assertEqual(sample["shadow_mask_path"], (root / "shadow_mask" / "0001.png").resolve())


if __name__ == "__main__":
    unittest.main()
