from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def load_metrics(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["dice"] = float(row["dice"])
    return rows


def copy_cases(
    metrics_summary: str | Path,
    predictions_dir: str | Path,
    best_dir: str | Path,
    worst_dir: str | Path,
    top_k: int = 10,
) -> tuple[list[Path], list[Path]]:
    rows = load_metrics(metrics_summary)
    best = sorted(rows, key=lambda row: row["dice"], reverse=True)[:top_k]
    worst = sorted(rows, key=lambda row: row["dice"])[:top_k]
    predictions = Path(predictions_dir)
    best_output = Path(best_dir)
    worst_output = Path(worst_dir)
    best_output.mkdir(parents=True, exist_ok=True)
    worst_output.mkdir(parents=True, exist_ok=True)

    copied_best: list[Path] = []
    copied_worst: list[Path] = []
    for group, output_dir, copied in ((best, best_output, copied_best), (worst, worst_output, copied_worst)):
        for row in group:
            source = predictions / f"{row['sample_id']}_prediction.png"
            if not source.exists():
                continue
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied_best, copied_worst


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best and worst U-Net prediction panels by Dice score.")
    parser.add_argument("--metrics", default="data/outputs/visualizations/unet_shadow_refiner/report/metrics_summary.csv")
    parser.add_argument("--predictions-dir", default="data/outputs/visualizations/unet_shadow_refiner/predictions")
    parser.add_argument("--best-dir", default="data/outputs/visualizations/unet_shadow_refiner/best_cases")
    parser.add_argument("--worst-dir", default="data/outputs/visualizations/unet_shadow_refiner/worst_cases")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    best, worst = copy_cases(args.metrics, args.predictions_dir, args.best_dir, args.worst_dir, args.top_k)
    print(f"Copied {len(best)} best cases and {len(worst)} worst cases.")


if __name__ == "__main__":
    main()
