from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _to_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def load_history(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        value = _to_float(row.get(key))
        if value is not None:
            xs.append(int(row["epoch"]))
            ys.append(value)
    return xs, ys


def _plot_metric(rows: list[dict], keys: list[tuple[str, str]], title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for key, label in keys:
        xs, ys = _series(rows, key)
        if xs:
            plt.plot(xs, ys, marker="o", label=label)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_training_curves(history: str | Path, output_dir: str | Path) -> list[Path]:
    rows = load_history(history)
    output = Path(output_dir)
    paths = [
        output / "loss_curve.png",
        output / "dice_curve.png",
        output / "iou_curve.png",
    ]
    _plot_metric(rows, [("train_total_loss", "train total"), ("val_total_loss", "val total")], "Loss", "loss", paths[0])
    _plot_metric(rows, [("train_dice_score", "train dice"), ("val_dice", "val dice")], "Dice", "dice", paths[1])
    _plot_metric(rows, [("val_iou", "val IoU")], "IoU", "iou", paths[2])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot U-Net shadow refiner training curves.")
    parser.add_argument("--history", default="checkpoints/unet_shadow_refiner/history.csv")
    parser.add_argument("--output-dir", default="data/outputs/visualizations/unet_shadow_refiner/curves")
    args = parser.parse_args()

    paths = plot_training_curves(args.history, args.output_dir)
    for path in paths:
        print(f"Saved curve: {path}")


if __name__ == "__main__":
    main()
