import unittest

import numpy as np

from modules.module_c_sg_shadow_prior import (
    compute_sg_shadow_prior,
    resize_mask,
    resize_point_map,
)


TEST_CONFIG = {
    "sg_shadow_prior": {
        "coarse_size": 64,
        "angular_tolerance_deg": 8.0,
        "min_object_area": 4,
        "max_object_points": 64,
        "max_receiver_points": 2048,
        "hard_blur_kernel": 3,
        "soft_blur_kernel": 31,
        "lambda_hard_threshold": 80.0,
        "lambda_soft_threshold": 5.0,
        "min_mu_z": 0.1,
        "max_forward_distance_base": 3.0,
        "normalize_output": True,
        "save_per_lobe_debug": True,
    }
}


def simple_point_map(size: int = 32) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = np.zeros_like(grid_x)
    return np.stack([grid_x, grid_y, grid_z], axis=-1)


def object_mask(size: int = 32) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[13:18, 8:13] = 255
    return mask


def sg_lobes(direct_amplitude=1.0, diffuse_amplitude=0.2, direct_lambda=100.0, diffuse_lambda=5.0):
    return [
        {
            "type": "direct",
            "mu": [-1.0, 0.0, 0.05],
            "lambda": direct_lambda,
            "amplitude": direct_amplitude,
        },
        {
            "type": "diffuse",
            "mu": [0.0, 0.0, 1.0],
            "lambda": diffuse_lambda,
            "amplitude": diffuse_amplitude,
        },
    ]


def gradient_energy(prior: np.ndarray) -> float:
    gy, gx = np.gradient(prior.astype(np.float32))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


class SGShadowPriorTestCase(unittest.TestCase):
    def test_valid_inputs_return_combined_64x64(self):
        result = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(), TEST_CONFIG, return_debug=True)

        self.assertEqual(result["combined"].shape, (64, 64))
        self.assertEqual(result["direct"].shape, (64, 64))
        self.assertEqual(result["diffuse_modulated"].shape, (64, 64))

    def test_output_range_is_zero_to_one(self):
        result = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(), TEST_CONFIG)

        self.assertGreaterEqual(float(result["combined"].min()), 0.0)
        self.assertLessEqual(float(result["combined"].max()), 1.0)

    def test_empty_object_mask_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_sg_shadow_prior(np.zeros((32, 32), dtype=np.uint8), simple_point_map(), sg_lobes(), TEST_CONFIG)

    def test_invalid_point_map_shape_raises_value_error(self):
        with self.assertRaises(ValueError):
            resize_point_map(np.zeros((32, 32), dtype=np.float32), 64)

    def test_direct_sg_amplitude_affects_direct_prior_intensity(self):
        config = {**TEST_CONFIG, "sg_shadow_prior": {**TEST_CONFIG["sg_shadow_prior"], "normalize_output": False}}
        high = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(direct_amplitude=1.0), config)
        low = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(direct_amplitude=0.25), config)

        self.assertGreater(float(high["direct"].sum()), float(low["direct"].sum()))

    def test_diffuse_amplitude_attenuates_direct_confidence(self):
        config = {**TEST_CONFIG, "sg_shadow_prior": {**TEST_CONFIG["sg_shadow_prior"], "normalize_output": False}}
        weak = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(diffuse_amplitude=0.1), config)
        strong = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(diffuse_amplitude=1.0), config)

        self.assertLess(float(strong["combined"].max()), float(weak["combined"].max()))

    def test_lower_direct_lambda_produces_blurrier_prior(self):
        hard = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(direct_lambda=100.0), TEST_CONFIG)
        soft = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(direct_lambda=5.0), TEST_CONFIG)

        self.assertLess(gradient_energy(soft["direct"]), gradient_energy(hard["direct"]))

    def test_receiver_mask_restricts_output(self):
        receiver = np.zeros((32, 32), dtype=np.uint8)
        receiver[:, 18:] = 255
        result = compute_sg_shadow_prior(object_mask(), simple_point_map(), sg_lobes(), TEST_CONFIG, receiver_mask=receiver)
        coarse_receiver = resize_mask(receiver, 64)

        self.assertEqual(float(result["combined"][~coarse_receiver].sum()), 0.0)

    def test_deterministic_subsampling_stable_shape_and_no_crash(self):
        config = {**TEST_CONFIG, "sg_shadow_prior": {**TEST_CONFIG["sg_shadow_prior"], "max_object_points": 5, "max_receiver_points": 50}}

        first = compute_sg_shadow_prior(object_mask(96), simple_point_map(96), sg_lobes(), config)
        second = compute_sg_shadow_prior(object_mask(96), simple_point_map(96), sg_lobes(), config)

        self.assertEqual(first["combined"].shape, (64, 64))
        self.assertTrue(np.allclose(first["combined"], second["combined"]))


if __name__ == "__main__":
    unittest.main()
