import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from modules.module_b_light_engine import run_light_engine


def run_main_pipeline(config_path: str = "configs/config.yaml") -> dict:
    light_result = run_light_engine(config_path)

    return {
        "light_vec_camera": light_result["light_vec_camera"],
        "light_engine_result": light_result,
    }


if __name__ == "__main__":
    result = run_main_pipeline()
    print("=== Main Pipeline Result ===")
    print(f"light_vec_camera: {result['light_vec_camera']}")
