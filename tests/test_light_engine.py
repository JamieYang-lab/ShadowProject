import sys
from pathlib import Path

# 讓 Python 能找到專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.module_b_light_engine import (
    load_config,
    get_solar_angles,
    solar_angles_to_world_vector,
    sun_world_to_light_world,
    world_to_camera_light_vector,
    run_light_engine
)


def print_result(title: str, result: dict):
    print(title)
    print(f"太陽高度角 elevation (deg): {result['elevation_deg']:.4f}")
    print(f"太陽方位角 azimuth (deg):   {result['azimuth_deg']:.4f}")
    print(f"世界座標下的太陽方向向量:   {result['sun_vec_world']}")
    print(f"世界座標下的光線方向向量:   {result['light_vec_world']}")
    print(f"相機座標下的光源方向向量:   {result['light_vec_camera']}")
    print("-" * 70)


def test_current_config():
    """
    測試目前 config.yaml 的設定
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    result = run_light_engine(str(config_path))
    print_result("=== Test 1: Current Config ===", result)


def test_multiple_timestamps():
    """
    固定地點與 heading，測試不同時間的太陽角度與光向量
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))

    longitude = config["location"]["longitude"]
    latitude = config["location"]["latitude"]
    timezone = config["capture"]["timezone"]
    heading = config["camera"]["heading"]

    test_times = [
        "2024-07-01 09:00:00",
        "2024-07-01 12:00:00",
        "2024-07-01 15:00:00",
        "2024-07-01 17:00:00",
    ]

    print("=== Test 2: Multiple Timestamps ===")
    for timestamp in test_times:
        elevation_deg, azimuth_deg = get_solar_angles(
            longitude=longitude,
            latitude=latitude,
            timestamp=timestamp,
            timezone=timezone
        )

        sun_vec_world = solar_angles_to_world_vector(
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg
        )

        light_vec_world = sun_world_to_light_world(sun_vec_world)

        light_vec_camera = world_to_camera_light_vector(
            light_vec_world=light_vec_world,
            heading_deg=heading
        )

        result = {
            "elevation_deg": elevation_deg,
            "azimuth_deg": azimuth_deg,
            "sun_vec_world": sun_vec_world,
            "light_vec_world": light_vec_world,
            "light_vec_camera": light_vec_camera
        }

        print_result(f"[timestamp = {timestamp}, heading = {heading}°]", result)


def test_multiple_headings():
    """
    固定地點與時間，測試不同 heading 下相機座標光向量如何改變
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))

    longitude = config["location"]["longitude"]
    latitude = config["location"]["latitude"]
    timestamp = config["capture"]["timestamp"]
    timezone = config["capture"]["timezone"]

    test_headings = [0.0, 90.0, 180.0, 270.0]

    print("=== Test 3: Multiple Headings ===")
    for heading in test_headings:
        elevation_deg, azimuth_deg = get_solar_angles(
            longitude=longitude,
            latitude=latitude,
            timestamp=timestamp,
            timezone=timezone
        )

        sun_vec_world = solar_angles_to_world_vector(
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg
        )

        light_vec_world = sun_world_to_light_world(sun_vec_world)

        light_vec_camera = world_to_camera_light_vector(
            light_vec_world=light_vec_world,
            heading_deg=heading
        )

        result = {
            "elevation_deg": elevation_deg,
            "azimuth_deg": azimuth_deg,
            "sun_vec_world": sun_vec_world,
            "light_vec_world": light_vec_world,
            "light_vec_camera": light_vec_camera
        }

        print_result(f"[timestamp = {timestamp}, heading = {heading}°]", result)


def main():
    test_current_config()
    test_multiple_timestamps()
    test_multiple_headings()


if __name__ == "__main__":
    main()