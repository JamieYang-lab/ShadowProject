import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.module_b_weather import build_weather_features, classify_weather_mode
from modules.module_b_weather_to_sg import (
    SOURCE_NAME,
    save_sg_params_json,
    save_weather_features_json,
    weather_to_sg_params,
)


TEST_CONFIG = {
    "default_cloudiness": 20,
    "default_visibility": 10000,
    "mode_thresholds": {
        "sunny_cloud_max": 30,
        "cloudy_cloud_max": 75,
    },
    "visibility_thresholds": {
        "high": 8000,
        "medium": 3000,
    },
    "sg_rules": {
        "sunny": {
            "direct_lambda": 100.0,
            "direct_amplitude": 1.0,
            "diffuse_lambda": 5.0,
            "diffuse_amplitude": 0.2,
        },
        "cloudy": {
            "direct_lambda": 50.0,
            "direct_amplitude": 0.6,
            "diffuse_lambda": 5.0,
            "diffuse_amplitude": 0.5,
        },
        "overcast": {
            "direct_lambda": 10.0,
            "direct_amplitude": 0.2,
            "diffuse_lambda": 3.0,
            "diffuse_amplitude": 0.9,
        },
    },
    "visibility_adjustment": {
        "low_visibility_direct_scale": 0.5,
        "low_visibility_diffuse_boost": 0.2,
        "medium_visibility_direct_scale": 0.8,
        "medium_visibility_diffuse_boost": 0.1,
    },
}


def features(sample_id: str, cloudiness: float, visibility: float):
    return build_weather_features(
        sample_id=sample_id,
        cloudiness=cloudiness,
        visibility=visibility,
        weather_main="Clear",
        weather_description="test",
        config=TEST_CONFIG,
    )


class WeatherToSGTestCase(unittest.TestCase):
    def test_sunny_high_visibility_keeps_high_direct_ratio(self):
        high_vis = features("sunny_high", 20, 10000)

        self.assertGreater(high_vis["direct_ratio"], high_vis["diffuse_ratio"])
        self.assertAlmostEqual(high_vis["direct_ratio"] + high_vis["diffuse_ratio"], 1.0)

    def test_sunny_low_visibility_has_lower_direct_ratio_than_high_visibility(self):
        high_vis = features("sunny_high", 20, 10000)
        low_vis = features("sunny_low", 20, 1000)

        self.assertLess(low_vis["direct_ratio"], high_vis["direct_ratio"])
        self.assertGreater(low_vis["diffuse_ratio"], high_vis["diffuse_ratio"])
        self.assertAlmostEqual(low_vis["direct_ratio"] + low_vis["diffuse_ratio"], 1.0)

    def test_overcast_low_visibility_keeps_diffuse_ratio_dominant(self):
        low_vis = features("overcast_low", 90, 1000)

        self.assertGreater(low_vis["diffuse_ratio"], low_vis["direct_ratio"])
        self.assertAlmostEqual(low_vis["direct_ratio"] + low_vis["diffuse_ratio"], 1.0)

    def test_sunny_weather_uses_high_direct_and_low_diffuse(self):
        sg = weather_to_sg_params("sunny", [0.0, 1.0, 1.0], features("sunny", 10, 10000), TEST_CONFIG)
        direct, diffuse = sg["sg_lobes"]

        self.assertEqual(direct["lambda"], 100.0)
        self.assertEqual(direct["amplitude"], 1.0)
        self.assertEqual(diffuse["amplitude"], 0.2)

    def test_overcast_weather_uses_low_direct_and_high_diffuse(self):
        sg = weather_to_sg_params("overcast", [0.0, 1.0, 1.0], features("overcast", 90, 10000), TEST_CONFIG)
        direct, diffuse = sg["sg_lobes"]

        self.assertEqual(direct["lambda"], 10.0)
        self.assertEqual(direct["amplitude"], 0.2)
        self.assertEqual(diffuse["lambda"], 3.0)
        self.assertEqual(diffuse["amplitude"], 0.9)

    def test_low_visibility_reduces_direct_and_increases_diffuse(self):
        high_vis = weather_to_sg_params("cloudy", [0.0, 1.0, 1.0], features("cloudy", 40, 10000), TEST_CONFIG)
        low_vis = weather_to_sg_params("cloudy", [0.0, 1.0, 1.0], features("cloudy", 40, 1000), TEST_CONFIG)

        self.assertLess(low_vis["sg_lobes"][0]["amplitude"], high_vis["sg_lobes"][0]["amplitude"])
        self.assertGreater(low_vis["sg_lobes"][1]["amplitude"], high_vis["sg_lobes"][1]["amplitude"])

    def test_direct_mu_is_normalized_unit_vector(self):
        sg = weather_to_sg_params("sample", [0.0, 3.0, 4.0], features("sample", 10, 10000), TEST_CONFIG)
        mu = np.asarray(sg["sg_lobes"][0]["mu"], dtype=np.float64)

        self.assertTrue(np.isclose(np.linalg.norm(mu), 1.0))

    def test_output_has_exactly_two_lobes_and_diffuse_mu(self):
        sg = weather_to_sg_params("sample", [1.0, 0.0, 1.0], features("sample", 40, 10000), TEST_CONFIG)

        self.assertEqual(len(sg["sg_lobes"]), 2)
        self.assertEqual(sg["sg_lobes"][1]["type"], "diffuse")
        self.assertEqual(sg["sg_lobes"][1]["mu"], [0.0, 0.0, 1.0])

    def test_source_is_rule_based(self):
        sg = weather_to_sg_params("sample", [1.0, 0.0, 1.0], features("sample", 40, 10000), TEST_CONFIG)

        self.assertEqual(sg["source"], SOURCE_NAME)

    def test_json_save_and_load_works(self):
        weather = features("sample", 20, 10000)
        sg = weather_to_sg_params("sample", [1.0, 0.0, 1.0], weather, TEST_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            weather_path = Path(tmp) / "weather" / "sample.json"
            sg_path = Path(tmp) / "sg" / "sample.json"

            save_weather_features_json(weather_path, weather)
            save_sg_params_json(sg_path, sg)

            with open(weather_path, "r", encoding="utf-8") as handle:
                loaded_weather = json.load(handle)
            with open(sg_path, "r", encoding="utf-8") as handle:
                loaded_sg = json.load(handle)

            self.assertEqual(loaded_weather["sample_id"], "sample")
            self.assertEqual(loaded_sg["source"], SOURCE_NAME)
            self.assertEqual(len(loaded_sg["sg_lobes"]), 2)

    def test_classify_weather_mode_thresholds(self):
        self.assertEqual(classify_weather_mode(29, 10000, TEST_CONFIG), "sunny")
        self.assertEqual(classify_weather_mode(30, 10000, TEST_CONFIG), "cloudy")
        self.assertEqual(classify_weather_mode(75, 10000, TEST_CONFIG), "overcast")


if __name__ == "__main__":
    unittest.main()
