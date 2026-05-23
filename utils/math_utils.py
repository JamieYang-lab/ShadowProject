import numpy as np


# =========================
# 基本向量工具
# =========================

def safe_norm(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    norm = safe_norm(v)

    if norm == 0:
        raise ValueError("向量長度為 0，無法正規化。")

    return v / norm


def is_unit_vector(v: np.ndarray, atol: float = 1e-6) -> bool:
    return np.isclose(safe_norm(v), 1.0, atol=atol)


# =========================
# 角度工具
# =========================

def deg2rad(deg: float) -> float:
    return float(np.deg2rad(deg))


def rad2deg(rad: float) -> float:
    return float(np.rad2deg(rad))


def clip_cosine(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return float(np.dot(v1_u, v2_u))


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_val = clip_cosine(cosine_similarity(v1, v2))
    return rad2deg(np.arccos(cos_val))


# =========================
# Rotation Matrix（核心）
# =========================

def rotation_matrix_yaw_deg(yaw_deg: float) -> np.ndarray:
    """
    yaw = heading（繞 z 軸旋轉）
    """
    h = deg2rad(yaw_deg)

    return np.array([
        [ np.cos(h), -np.sin(h), 0],
        [ np.sin(h),  np.cos(h), 0],
        [ 0,          0,         1]
    ], dtype=np.float64)


def rotation_matrix_pitch_deg(pitch_deg: float) -> np.ndarray:
    """
    pitch（繞 x 軸旋轉）
    """
    p = deg2rad(pitch_deg)

    return np.array([
        [1, 0,           0          ],
        [0, np.cos(p),  -np.sin(p)],
        [0, np.sin(p),   np.cos(p)]
    ], dtype=np.float64)


def rotation_matrix_roll_deg(roll_deg: float) -> np.ndarray:
    """
    roll（繞 y 軸旋轉）
    """
    r = deg2rad(roll_deg)

    return np.array([
        [ np.cos(r), 0, np.sin(r)],
        [ 0,         1, 0        ],
        [-np.sin(r), 0, np.cos(r)]
    ], dtype=np.float64)


# =========================
# World → Camera Rotation
# =========================

def build_world_to_camera_rotation(heading_deg: float,
                                   pitch_deg: float,
                                   roll_deg: float) -> np.ndarray:
    """
    建立 world → camera rotation matrix

    順序（非常重要）:
        先 yaw → 再 pitch → 再 roll

    R = R_roll @ R_pitch @ R_yaw
    """
    R_yaw = rotation_matrix_yaw_deg(heading_deg)
    R_pitch = rotation_matrix_pitch_deg(pitch_deg)
    R_roll = rotation_matrix_roll_deg(roll_deg)

    R = R_roll @ R_pitch @ R_yaw

    return R


def world_to_camera_vector(vec_world: np.ndarray,
                           heading_deg: float,
                           pitch_deg: float,
                           roll_deg: float) -> np.ndarray:
    """
    將世界座標向量轉成相機座標向量
    """
    vec_world = normalize(vec_world)

    R = build_world_to_camera_rotation(
        heading_deg,
        pitch_deg,
        roll_deg
    )

    vec_cam = R @ vec_world

    return normalize(vec_cam)


# =========================
# 平面投影（之後 ray casting 用）
# =========================

def project_vector_onto_plane(v: np.ndarray,
                              plane_normal: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = normalize(plane_normal)

    projected = v - np.dot(v, n) * n

    if safe_norm(projected) == 0:
        raise ValueError("投影結果為 0 向量")

    return normalize(projected)