import numpy as np
import pandas as pd
import pvlib
import yaml


def load_config(config_path: str) -> dict:
    """
    讀取 YAML 設定檔
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_solar_angles(longitude: float,
                     latitude: float,
                     timestamp: str,
                     timezone: str):
    """
    根據經緯度與拍攝時間，計算太陽方位角與高度角

    輸入:
        longitude : 經度（東經為正）
        latitude  : 緯度（北緯為正）
        timestamp : 拍攝時間字串，例如 "2024-07-01 15:00:00"
        timezone  : 時區，例如 "Asia/Taipei"

    輸出:
        elevation_deg : 太陽高度角（度）
        azimuth_deg   : 太陽方位角（度）
    """
    time = pd.DatetimeIndex([pd.Timestamp(timestamp, tz=timezone)])
    solar_pos = pvlib.solarposition.get_solarposition(
        time,
        latitude=latitude,
        longitude=longitude
    )

    azimuth_deg = float(solar_pos["azimuth"].iloc[0])
    elevation_deg = float(solar_pos["apparent_elevation"].iloc[0])

    return elevation_deg, azimuth_deg


def solar_angles_to_world_vector(elevation_deg: float,
                                 azimuth_deg: float) -> np.ndarray:
    """
    將太陽高度角 / 方位角轉成世界座標下的太陽方向向量

    世界座標採 ENU:
        x = East
        y = North
        z = Up

    NOAA / pvlib 的方位角定義:
        0°   = 北
        90°  = 東
        180° = 南
        270° = 西

    輸出:
        sun_vec_world : 單位向量，方向為「由場景指向太陽」
    """
    a = np.deg2rad(azimuth_deg)
    e = np.deg2rad(elevation_deg)

    sun_vec_world = np.array([
        np.cos(e) * np.sin(a),  # x = East
        np.cos(e) * np.cos(a),  # y = North
        np.sin(e)               # z = Up
    ], dtype=np.float64)

    norm = np.linalg.norm(sun_vec_world)
    if norm == 0:
        raise ValueError("太陽方向向量長度為 0，請檢查輸入參數。")

    return sun_vec_world / norm


def sun_world_to_light_world(sun_vec_world: np.ndarray) -> np.ndarray:
    """
    將世界座標下的太陽方向向量，轉成世界座標下的光線方向向量

    sun_vec_world:
        場景 -> 太陽

    light_vec_world:
        太陽 -> 場景
    """
    light_vec_world = -sun_vec_world
    norm = np.linalg.norm(light_vec_world)
    if norm == 0:
        raise ValueError("世界座標下的光線方向向量長度為 0。")
    return light_vec_world / norm


def world_to_camera_light_vector(light_vec_world: np.ndarray,
                                 heading_deg: float) -> np.ndarray:
    """
    將世界座標下的光線方向向量轉成相機座標下的光源方向向量

    目前先只考慮 heading，不考慮 pitch / roll。

    世界座標:
        x = East
        y = North
        z = Up

    相機座標:
        x_cam = 右
        y_cam = 上
        z_cam = 前

    heading 定義:
        0°   = 朝北
        90°  = 朝東
        180° = 朝南
        270° = 朝西
    """
    h = np.deg2rad(heading_deg)

    # 相機在世界座標中的三個基底方向
    forward = np.array([
        np.sin(h),   # x = East
        np.cos(h),   # y = North
        0.0
    ], dtype=np.float64)

    right = np.array([
        np.cos(h),
        -np.sin(h),
        0.0
    ], dtype=np.float64)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_cam = np.dot(light_vec_world, right)
    y_cam = np.dot(light_vec_world, up)
    z_cam = np.dot(light_vec_world, forward)

    light_vec_camera = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

    norm = np.linalg.norm(light_vec_camera)
    if norm == 0:
        raise ValueError("相機座標下的光源方向向量長度為 0。")

    return light_vec_camera / norm


def run_light_engine(config_path: str) -> dict:
    """
    主功能：
    讀取 config，輸出：
        - elevation
        - azimuth
        - 世界座標下的太陽方向向量
        - 世界座標下的光線方向向量
        - 相機座標下的光源方向向量
    """
    config = load_config(config_path)

    longitude = config["location"]["longitude"]
    latitude = config["location"]["latitude"]
    timestamp = config["capture"]["timestamp"]
    timezone = config["capture"]["timezone"]
    heading = config["camera"]["heading"]

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

    return {
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
        "sun_vec_world": sun_vec_world,
        "light_vec_world": light_vec_world,
        "light_vec_camera": light_vec_camera
    }