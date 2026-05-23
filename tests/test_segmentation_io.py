import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from modules.module_a_segmentation import filter_masks, load_mask_png, save_mask_overlay, save_mask_png
from scripts.run_sam_segmentation import build_sample_output_paths, save_candidate_masks


class SegmentationIOTestCase(unittest.TestCase):
    def test_save_and_load_mask_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = np.zeros((12, 10), dtype=bool)
            mask[2:6, 3:8] = True
            path = Path(tmp) / "masks" / "mask_000.png"

            save_mask_png(path, mask)
            loaded = load_mask_png(path)

            self.assertTrue(path.exists())
            self.assertEqual(loaded.shape, (12, 10))
            self.assertTrue(np.array_equal(loaded, mask))

    def test_overlay_generation_with_fake_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "image.png"
            output_path = tmp_path / "overlay" / "sample_overlay.png"
            Image.new("RGB", (24, 16), (120, 120, 120)).save(image_path)
            mask = np.zeros((16, 24), dtype=bool)
            mask[3:10, 4:12] = True

            save_mask_overlay(
                image_path,
                [{"mask": mask, "area": int(mask.sum()), "bbox": [4, 3, 8, 7], "score": 0.9}],
                output_path,
            )

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as overlay:
                self.assertEqual(overlay.mode, "RGB")
                self.assertEqual(overlay.size, (24, 16))

    def test_filtering_by_min_mask_area(self):
        small = np.zeros((10, 10), dtype=bool)
        small[0:2, 0:2] = True
        large = np.zeros((10, 10), dtype=bool)
        large[0:5, 0:5] = True

        filtered = filter_masks(
            [
                {"mask": small, "area": int(small.sum()), "bbox": [0, 0, 2, 2], "score": 0.8},
                {"mask": large, "area": int(large.sum()), "bbox": [0, 0, 5, 5], "score": 0.7},
            ],
            min_mask_area=10,
            max_masks=10,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["area"], 25)

    def test_output_directory_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = build_sample_output_paths("sample_001", tmp_path / "sam_masks", tmp_path / "vis")
            mask = np.zeros((8, 8), dtype=bool)
            mask[2:6, 2:6] = True

            written = save_candidate_masks(
                paths["mask_dir"],
                [{"mask": mask, "area": int(mask.sum()), "bbox": [2, 2, 4, 4], "score": None}],
            )

            self.assertEqual(paths["mask_dir"], tmp_path / "sam_masks" / "sample_001")
            self.assertEqual(paths["overlay"], tmp_path / "vis" / "sample_001_sam_overlay.png")
            self.assertEqual(written, [tmp_path / "sam_masks" / "sample_001" / "mask_000.png"])
            self.assertTrue(written[0].exists())


if __name__ == "__main__":
    unittest.main()
