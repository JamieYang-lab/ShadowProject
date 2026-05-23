import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from modules.module_a_receiver_selection import (
    compute_receiver_mask,
    load_sam_masks,
    save_receiver_mask_png,
)
from scripts.generate_receiver_mask import generate_receiver_mask


TEST_CONFIG = {
    "receiver_selection": {
        "lower_y_ratio": 0.35,
        "min_object_area": 4,
        "plane_sample_lower_ratio": 0.5,
        "plane_distance_threshold": 0.05,
        "sam_min_area": 20,
        "sam_lower_overlap_min": 0.25,
        "sam_object_overlap_max": 0.2,
        "sam_plane_overlap_min": 0.25,
        "combine_sam_with_plane": "union",
    }
}


def object_mask(shape=(32, 32)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[20:25, 10:15] = 255
    return mask


def fake_point_map(shape=(32, 32), sloped: bool = False) -> np.ndarray:
    h, w = shape
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gz = 0.02 * gx if sloped else np.zeros_like(gx)
    return np.stack([gx, gy, gz], axis=-1)


def save_image(path: Path, size=(32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (127, 127, 127)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def write_receiver_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(TEST_CONFIG, handle, sort_keys=False)


class ReceiverSelectionTestCase(unittest.TestCase):
    def test_receiver_mask_excludes_object_region(self):
        obj = object_mask()
        result = compute_receiver_mask(obj, obj.shape, config=TEST_CONFIG)

        self.assertEqual(int(result["receiver_mask"][obj > 0].sum()), 0)

    def test_lower_image_prior_keeps_lower_and_removes_upper_region(self):
        obj = object_mask()
        result = compute_receiver_mask(obj, obj.shape, config=TEST_CONFIG)
        receiver = result["receiver_mask"]

        self.assertEqual(int(receiver[:8, :].sum()), 0)
        self.assertGreater(int(receiver[20:, :].sum()), 0)

    def test_plane_distance_filtering_returns_valid_binary_mask(self):
        obj = object_mask()
        result = compute_receiver_mask(obj, obj.shape, point_map=fake_point_map(sloped=True), config=TEST_CONFIG)

        self.assertEqual(result["receiver_mask"].dtype, bool)
        self.assertEqual(result["receiver_mask"].shape, obj.shape)
        self.assertIsNotNone(result["plane"])

    def test_sam_candidate_selection_prefers_large_lower_masks(self):
        obj = object_mask()
        large_lower = np.zeros_like(obj, dtype=bool)
        large_lower[18:31, 2:30] = True
        small_upper = np.zeros_like(obj, dtype=bool)
        small_upper[1:3, 1:3] = True

        result = compute_receiver_mask(obj, obj.shape, sam_masks=[small_upper, large_lower], config=TEST_CONFIG)

        self.assertEqual(len(result["selected_sam_masks"]), 1)
        self.assertGreater(int(result["selected_sam_masks"][0].sum()), 20)

    def test_output_shape_matches_input(self):
        obj = object_mask((40, 24))
        result = compute_receiver_mask(obj, obj.shape, config=TEST_CONFIG)

        self.assertEqual(result["receiver_mask"].shape, (40, 24))

    def test_empty_object_mask_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Object mask area is too small"):
            compute_receiver_mask(np.zeros((32, 32), dtype=np.uint8), (32, 32), config=TEST_CONFIG)

    def test_invalid_point_map_falls_back_to_lower_prior(self):
        obj = object_mask()
        result = compute_receiver_mask(obj, obj.shape, point_map=np.zeros((32, 32), dtype=np.float32), config=TEST_CONFIG)

        self.assertIsNone(result["plane"])
        self.assertGreater(int(result["receiver_mask"].sum()), 0)

    def test_file_output_integration_returns_uint8_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            composite = tmp_path / "composite.png"
            object_path = tmp_path / "object.png"
            point_map_path = tmp_path / "point.npy"
            output_dir = tmp_path / "receiver"
            config_path = tmp_path / "receiver_config.yaml"
            save_image(composite)
            save_mask(object_path, object_mask())
            np.save(point_map_path, fake_point_map())
            write_receiver_config(config_path)

            output_path = generate_receiver_mask(
                sample_id="sample",
                composite_path=composite,
                object_mask_path=object_path,
                point_map_path=point_map_path,
                output_dir=output_dir,
                config_path=config_path,
            )

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as image:
                values = set(np.unique(np.asarray(image.convert("L"))).tolist())
            self.assertTrue(values.issubset({0, 255}))

    def test_load_sam_masks_from_png_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "sam"
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[8:, :] = 255
            save_mask(directory / "mask_000.png", mask)

            loaded = load_sam_masks(directory, (32, 32))

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].shape, (32, 32))
            self.assertEqual(loaded[0].dtype, bool)


if __name__ == "__main__":
    unittest.main()
