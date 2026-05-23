import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from modules.module_c_shadow_sketch import (
    compute_centroid,
    generate_shadow_sketch_from_direction,
    save_shadow_sketch,
)


class ShadowSketchTestCase(unittest.TestCase):
    def test_centroid_computation(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2, 4] = 255
        mask[4, 6] = 255

        self.assertEqual(compute_centroid(mask), (5.0, 3.0))

    def test_empty_mask_raises_value_error(self):
        mask = np.zeros((10, 10), dtype=np.uint8)

        with self.assertRaises(ValueError):
            compute_centroid(mask)

    def test_shadow_sketch_shape_and_value_range(self):
        object_mask = np.zeros((16, 16), dtype=np.uint8)
        object_mask[4:8, 4:8] = 255
        config = {
            "shadow_sketch": {
                "minimum_mask_area": 4,
                "blur_kernel_size": 3,
                "projection_length_scale": 2.0,
                "num_projection_steps": 8,
            }
        }

        sketch = generate_shadow_sketch_from_direction(object_mask, [-1.0, 0.0], (20, 24), config)

        self.assertEqual(sketch.shape, (20, 24))
        self.assertEqual(sketch.dtype, np.uint8)
        self.assertGreaterEqual(int(sketch.min()), 0)
        self.assertLessEqual(int(sketch.max()), 255)

    def test_save_shadow_sketch_writes_grayscale_png(self):
        sketch = np.zeros((8, 8), dtype=np.uint8)
        sketch[2:4, 2:4] = 255
        with tempfile.TemporaryDirectory() as tmp:
            output = save_shadow_sketch(sketch, Path(tmp) / "sketch.png")
            with Image.open(output) as loaded:
                self.assertEqual(loaded.mode, "L")
                self.assertEqual(loaded.size, (8, 8))


if __name__ == "__main__":
    unittest.main()
