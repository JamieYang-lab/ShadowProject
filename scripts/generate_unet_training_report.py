from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _float_or_none(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _read_csv(path: str | Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_epoch(history: list[dict]) -> dict | None:
    val_rows = [row for row in history if _float_or_none(row.get("val_dice")) is not None]
    if val_rows:
        return max(val_rows, key=lambda row: _float_or_none(row.get("val_dice")) or -1.0)
    if history:
        return min(history, key=lambda row: _float_or_none(row.get("train_total_loss")) or float("inf"))
    return None


def generate_report(
    history_path: str | Path,
    metrics_path: str | Path,
    curves_dir: str | Path,
    best_dir: str | Path,
    worst_dir: str | Path,
    output_path: str | Path,
) -> Path:
    history = _read_csv(history_path)
    metrics = _read_csv(metrics_path)
    best_epoch = _best_epoch(history)
    final_train_loss = _float_or_none(history[-1].get("train_total_loss")) if history else None
    best_val_dice = _float_or_none(best_epoch.get("val_dice")) if best_epoch else None
    best_val_iou = _float_or_none(best_epoch.get("val_iou")) if best_epoch else None
    mean_dice = None
    mean_iou = None
    if metrics:
        mean_dice = sum(float(row["dice"]) for row in metrics) / len(metrics)
        mean_iou = sum(float(row["iou"]) for row in metrics) / len(metrics)

    best_files = sorted(Path(best_dir).glob("*.png"))
    worst_files = sorted(Path(worst_dir).glob("*.png"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# U-Net Shadow Refiner Training Report",
        "",
        "## Summary",
        f"- Best epoch: {best_epoch.get('epoch') if best_epoch else 'N/A'}",
        f"- Best val Dice: {best_val_dice if best_val_dice is not None else 'N/A'}",
        f"- Best val IoU: {best_val_iou if best_val_iou is not None else 'N/A'}",
        f"- Final train loss: {final_train_loss if final_train_loss is not None else 'N/A'}",
        f"- Mean eval Dice: {mean_dice if mean_dice is not None else 'N/A'}",
        f"- Mean eval IoU: {mean_iou if mean_iou is not None else 'N/A'}",
        "",
        "## Curves",
        f"- loss_curve.png: {Path(curves_dir) / 'loss_curve.png'}",
        f"- dice_curve.png: {Path(curves_dir) / 'dice_curve.png'}",
        f"- iou_curve.png: {Path(curves_dir) / 'iou_curve.png'}",
        "",
        "## Best Cases",
    ]
    lines.extend(f"- {path.name}" for path in best_files[:10])
    lines.extend(["", "## Worst Cases"])
    lines.extend(f"- {path.name}" for path in worst_files[:10])
    lines.extend(
        [
            "",
            "## Notes",
            "This report summarizes mask-refinement training only. It does not include diffusion, ControlNet, SGDiffusion, or final image generation.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown report for U-Net shadow refiner training.")
    parser.add_argument("--history", default="checkpoints/unet_shadow_refiner/history.csv")
    parser.add_argument("--metrics", default="data/outputs/visualizations/unet_shadow_refiner/report/metrics_summary.csv")
    parser.add_argument("--curves-dir", default="data/outputs/visualizations/unet_shadow_refiner/curves")
    parser.add_argument("--best-dir", default="data/outputs/visualizations/unet_shadow_refiner/best_cases")
    parser.add_argument("--worst-dir", default="data/outputs/visualizations/unet_shadow_refiner/worst_cases")
    parser.add_argument("--output", default="data/outputs/visualizations/unet_shadow_refiner/report/training_report.md")
    args = parser.parse_args()

    output = generate_report(args.history, args.metrics, args.curves_dir, args.best_dir, args.worst_dir, args.output)
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
