import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.module_a_geometry import ensure_dir, load_point_map, normalize_depth_for_png, save_depth_png, save_point_map
from scripts.run_moge_inference import build_output_paths


class GeometryIOTestCase(unittest.TestCase):
    def test_ensure_dir_creates_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "a" / "b" / "c"

            result = ensure_dir(directory)

            self.assertTrue(directory.exists())
            self.assertTrue(directory.is_dir())
            self.assertEqual(result, directory)

    def test_save_and_load_point_map_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            point_map = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
            path = Path(tmp) / "point_maps" / "sample.npy"

            save_point_map(path, point_map)
            loaded = load_point_map(path)

            self.assertTrue(path.exists())
            self.assertEqual(loaded.shape, (3, 4, 3))
            self.assertTrue(np.allclose(loaded, point_map))

    def test_save_point_map_rejects_invalid_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_point_map(Path(tmp) / "bad.npy", np.zeros((4, 4), dtype=np.float32))

    def test_normalize_depth_for_png_range_and_dtype(self):
        depth = np.array(
            [
                [1.0, 2.0, np.nan],
                [3.0, np.inf, 5.0],
            ],
            dtype=np.float32,
        )

        normalized = normalize_depth_for_png(depth)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertGreaterEqual(int(normalized.min()), 0)
        self.assertLessEqual(int(normalized.max()), 255)
        self.assertEqual(int(normalized[0, 2]), 0)
        self.assertEqual(int(normalized[1, 1]), 0)
        self.assertEqual(int(normalized[0, 0]), 0)
        self.assertEqual(int(normalized[1, 2]), 255)

    def test_save_depth_png_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth" / "sample.png"

            save_depth_png(path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

            self.assertTrue(path.exists())

    def test_output_filename_behavior(self):
        paths = build_output_paths(
            "sample_001",
            Path("data/intermediate/point_maps"),
            Path("data/intermediate/depth_maps"),
            Path("data/intermediate/normal_maps"),
        )

        self.assertEqual(paths["point_map"], Path("data/intermediate/point_maps/sample_001.npy"))
        self.assertEqual(paths["depth_png"], Path("data/intermediate/depth_maps/sample_001.png"))
        self.assertEqual(paths["normal_png"], Path("data/intermediate/normal_maps/sample_001.png"))


if __name__ == "__main__":
    unittest.main()
