import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from scripts.generate_pseudo_light import generate_pseudo_light
from scripts.generate_sg_params import generate_sg_params
from scripts.generate_shadow_sketch import generate_shadow_sketches
from scripts.visualize_preprocessing import visualize_preprocessing


def save_mask(path: Path, box: tuple[int, int, int, int], size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask).save(path)


def save_image(path: Path, size: tuple[int, int] = (32, 32), color=(127, 127, 127)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def write_config(config_path: Path, root: Path, intermediate: Path, outputs: Path) -> None:
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
        "outputs": {
            "visualization_dir": str(outputs / "visualizations"),
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def create_sample(root: Path, sample_id: str, y_offset: int = 0) -> None:
    save_image(root / "composite" / f"{sample_id}.png")
    save_image(root / "target" / f"{sample_id}.png")
    save_mask(root / "object_mask" / f"{sample_id}.png", (8, 8 + y_offset, 12, 12 + y_offset))
    save_mask(root / "shadow_mask" / f"{sample_id}.png", (18, 8 + y_offset, 24, 12 + y_offset))


class VisualizePreprocessingTestCase(unittest.TestCase):
    def test_visualize_preprocessing_writes_debug_board_for_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            intermediate = tmp_path / "data" / "intermediate"
            outputs = tmp_path / "data" / "outputs"
            config_path = tmp_path / "configs" / "dataset_config.yaml"
            write_config(config_path, root, intermediate, outputs)
            create_sample(root, "0001")

            generate_pseudo_light(config_path)
            generate_sg_params(config_path)
            generate_shadow_sketches(config_path)

            written = visualize_preprocessing(config_path, sample_id="0001")

            self.assertEqual(len(written), 1)
            self.assertTrue(written[0].exists())
            self.assertEqual(written[0].name, "0001_preprocessing.png")
            with Image.open(written[0]) as debug_image:
                self.assertEqual(debug_image.mode, "RGB")
                self.assertGreater(debug_image.width, debug_image.height)

    def test_visualize_preprocessing_respects_max_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "data" / "desobav2"
            intermediate = tmp_path / "data" / "intermediate"
            outputs = tmp_path / "data" / "outputs"
            config_path = tmp_path / "configs" / "dataset_config.yaml"
            write_config(config_path, root, intermediate, outputs)
            create_sample(root, "0001")
            create_sample(root, "0002", y_offset=4)

            generate_pseudo_light(config_path)
            generate_sg_params(config_path)
            generate_shadow_sketches(config_path)

            written = visualize_preprocessing(config_path, max_samples=1)

            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].name, "0001_preprocessing.png")


if __name__ == "__main__":
    unittest.main()
