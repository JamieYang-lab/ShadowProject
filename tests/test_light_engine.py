import sys
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_b_light_engine import (
    get_solar_angles,
    load_config,
    run_light_engine,
    solar_angles_to_world_vector,
    sun_world_to_light_world,
    validate_config_contract,
    world_to_camera_light_vector,
)
from utils.math_utils import is_unit_vector


class LightEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.config_path = PROJECT_ROOT / "configs" / "config.yaml"
        self.config = load_config(str(self.config_path))
        self.validated_config = validate_config_contract(self.config)

    def assert_unit_vectors(self, result: dict):
        self.assertTrue(is_unit_vector(result["sun_vec_world"]))
        self.assertTrue(is_unit_vector(result["light_vec_world"]))
        self.assertTrue(is_unit_vector(result["light_vec_camera"]))

    def write_temp_config(self, config: dict) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        with handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_current_config_returns_solar_metadata(self):
        result = run_light_engine(str(self.config_path))

        self.assert_unit_vectors(result)
        self.assertEqual(result["light_source_type"], "solar")
        self.assertEqual(result["solar_provider"], "pvlib")

    def test_multiple_timestamps_follow_expected_solar_trend(self):
        longitude = self.validated_config["location"]["longitude"]
        latitude = self.validated_config["location"]["latitude"]
        timezone = self.validated_config["capture"]["timezone"]

        test_times = [
            "2024-07-01 09:00:00",
            "2024-07-01 12:00:00",
            "2024-07-01 15:00:00",
            "2024-07-01 17:00:00",
        ]

        elevations = []
        azimuths = []

        for timestamp in test_times:
            elevation_deg, azimuth_deg = get_solar_angles(
                longitude=longitude,
                latitude=latitude,
                timestamp=timestamp,
                timezone=timezone,
            )
            elevations.append(elevation_deg)
            azimuths.append(azimuth_deg)

        self.assertGreater(elevations[1], elevations[0])
        self.assertGreater(elevations[1], elevations[2])
        self.assertGreater(elevations[2], elevations[3])
        self.assertEqual(azimuths, sorted(azimuths))

    def test_heading_changes_only_camera_space_light_vector(self):
        longitude = self.validated_config["location"]["longitude"]
        latitude = self.validated_config["location"]["latitude"]
        timestamp = self.validated_config["capture"]["timestamp"]
        timezone = self.validated_config["capture"]["timezone"]

        elevation_deg, azimuth_deg = get_solar_angles(
            longitude=longitude,
            latitude=latitude,
            timestamp=timestamp,
            timezone=timezone,
        )
        sun_vec_world = solar_angles_to_world_vector(elevation_deg, azimuth_deg)
        base_light_vec_world = sun_world_to_light_world(sun_vec_world)

        headings = [0.0, 90.0, 180.0, 270.0]
        camera_vectors = []

        for heading in headings:
            light_vec_camera = world_to_camera_light_vector(
                light_vec_world=base_light_vec_world,
                heading_deg=heading,
                pitch_deg=0.0,
                roll_deg=0.0,
            )
            self.assert_unit_vectors(
                {
                    "sun_vec_world": sun_vec_world,
                    "light_vec_world": base_light_vec_world,
                    "light_vec_camera": light_vec_camera,
                }
            )
            camera_vectors.append(light_vec_camera)

        for _heading in headings:
            self.assertTrue(is_unit_vector(base_light_vec_world))

        self.assertFalse(np.allclose(camera_vectors[0], camera_vectors[1]))
        self.assertFalse(np.allclose(camera_vectors[1], camera_vectors[2]))

    def test_missing_required_field_raises_readable_error(self):
        invalid_config = {
            "location": {"longitude": 121.5654},
            "capture": {
                "timestamp": "2024-07-01 15:00:00",
                "timezone": "Asia/Taipei",
            },
            "camera": {"heading": 90.0},
        }
        temp_config_path = self.write_temp_config(invalid_config)

        with self.assertRaisesRegex(KeyError, "location.latitude"):
            run_light_engine(temp_config_path)

    def test_invalid_timestamp_format_raises_error(self):
        invalid_config = {
            "location": {"longitude": 121.5654, "latitude": 25.0330},
            "capture": {
                "timestamp": "2024-07-01T15:00:00+08:00",
                "timezone": "Asia/Taipei",
            },
            "camera": {"heading": 90.0, "pitch": 0.0, "roll": 0.0},
        }
        temp_config_path = self.write_temp_config(invalid_config)

        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD HH:MM:SS"):
            run_light_engine(temp_config_path)

    def test_sun_below_horizon_raises_error(self):
        invalid_config = {
            "location": {"longitude": 121.5654, "latitude": 25.0330},
            "capture": {
                "timestamp": "2024-07-01 00:00:00",
                "timezone": "Asia/Taipei",
            },
            "camera": {"heading": 90.0, "pitch": 0.0, "roll": 0.0},
        }
        temp_config_path = self.write_temp_config(invalid_config)

        with self.assertRaisesRegex(ValueError, "below the horizon"):
            run_light_engine(temp_config_path)


if __name__ == "__main__":
    unittest.main()
