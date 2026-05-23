import unittest

import numpy as np

from modules.module_b_sg_light import (
    flatten_sg_lobes,
    initialize_direct_lobe,
    initialize_sg_from_light_direction,
)


class SGLightTestCase(unittest.TestCase):
    def test_direct_lobe_direction_is_unit_vector(self):
        lobe = initialize_direct_lobe([3.0, 4.0, 0.0])
        self.assertTrue(np.isclose(np.linalg.norm(lobe.mu), 1.0))

    def test_flatten_length_is_k_times_five(self):
        lobes = initialize_sg_from_light_direction([1.0, 0.0, 1.0])
        flattened = flatten_sg_lobes(lobes)
        self.assertEqual(len(flattened), 2 * 5)

    def test_lambda_and_amplitude_ranges(self):
        lobes = initialize_sg_from_light_direction([1.0, 0.0, 1.0])
        for lobe in lobes:
            self.assertGreater(lobe.lambda_, 0.0)
            self.assertGreaterEqual(lobe.amplitude, 0.0)
            self.assertLessEqual(lobe.amplitude, 1.0)


if __name__ == "__main__":
    unittest.main()
