import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from scripts.generate_noaa_weather_sg_from_metadata import (
    compute_solar_vectors_from_metadata,
    generate_noaa_weather_sg_from_metadata,
)


def write_weather_config(path: Path) -> None:
    config = {
        "api_key_env": "OPENWEATHER_API_KEY",
        "api_base_url": "https://api.openweathermap.org/data/2.5/weather",
        "units": "metric",
        "output_weather_dir": "data/intermediate/weather",
        "output_sg_dir": "data/intermediate/sg_params",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def write_metadata_sg_config(path: Path) -> None:
    config = {
        "metadata_sg": {
            "weather_config": "configs/weather_config.yaml",
            "output_sg_dir": "data/intermediate/sg_params",
            "coordinate_frame": "camera",
            "source": "noaa_openweather_rule_based",
            "offline_defaults": {
                "cloudiness": 20,
                "visibility": 10000,
                "weather_main": "Clear",
                "weather_description": "manual/offline",
            },
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def metadata(sample_id: str = "demo_001", heading: float = 90.0) -> dict:
    return {
        "sample_id": sample_id,
        "latitude": 25.033,
        "longitude": 121.565,
        "timestamp": "2026-05-19 14:30:00",
        "timezone": "Asia/Taipei",
        "camera_heading": heading,
        "camera_pitch": 0.0,
        "camera_roll": 0.0,
    }


def write_metadata(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


class MetadataToSGTestCase(unittest.TestCase):
    def generate(self, tmp_path: Path, data: dict):
        weather_config = tmp_path / "configs" / "weather_config.yaml"
        metadata_sg_config = tmp_path / "configs" / "metadata_sg_config.yaml"
        metadata_path = tmp_path / "metadata.json"
        output_dir = tmp_path / "sg_params"
        write_weather_config(weather_config)
        write_metadata_sg_config(metadata_sg_config)
        write_metadata(metadata_path, data)
        return generate_noaa_weather_sg_from_metadata(
            metadata_path=metadata_path,
            weather_config_path=weather_config,
            offline=True,
            cloudiness=20,
            visibility=10000,
            output_dir=output_dir,
            metadata_sg_config_path=metadata_sg_config,
        )

    def test_output_json_exists_and_coordinate_frame_is_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            sg_params, output_path = self.generate(Path(tmp), metadata())

            self.assertTrue(output_path.exists())
            self.assertEqual(sg_params["coordinate_frame"], "camera")
            with open(output_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertEqual(loaded["coordinate_frame"], "camera")

    def test_direct_sg_mu_equals_sun_vector_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            sg_params, _ = self.generate(Path(tmp), metadata())

            self.assertTrue(np.allclose(sg_params["sg_lobes"][0]["mu"], sg_params["solar"]["sun_vector_camera"]))

    def test_direct_sg_mu_is_unit_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            sg_params, _ = self.generate(Path(tmp), metadata())
            mu = np.asarray(sg_params["sg_lobes"][0]["mu"], dtype=np.float64)

            self.assertTrue(np.isclose(np.linalg.norm(mu), 1.0))

    def test_changing_camera_heading_changes_sun_vector_camera(self):
        solar_a = compute_solar_vectors_from_metadata(metadata(heading=0.0))
        solar_b = compute_solar_vectors_from_metadata(metadata(heading=90.0))

        self.assertFalse(np.allclose(solar_a["sun_vector_camera"], solar_b["sun_vector_camera"]))

    def test_k_two_lobes_are_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            sg_params, _ = self.generate(Path(tmp), metadata())

            self.assertEqual(len(sg_params["sg_lobes"]), 2)
            self.assertEqual(sg_params["sg_lobes"][0]["type"], "direct")
            self.assertEqual(sg_params["sg_lobes"][1]["type"], "diffuse")

    def test_offline_mode_works_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            sg_params, _ = self.generate(Path(tmp), metadata("offline_demo"))

            self.assertEqual(sg_params["weather"]["sample_id"], "offline_demo")
            self.assertEqual(sg_params["source"], "noaa_openweather_rule_based")


if __name__ == "__main__":
    unittest.main()
