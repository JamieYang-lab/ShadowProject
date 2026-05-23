from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_unet_shadow_refiner import evaluate
from scripts.generate_unet_training_report import generate_report
from scripts.plot_unet_training_curves import plot_training_curves
from scripts.select_best_worst_predictions import copy_cases
from scripts.visualize_unet_predictions import visualize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all U-Net shadow refiner result visualizations and report.")
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/unet_shadow_refiner/best.pt")
    parser.add_argument("--history", default="checkpoints/unet_shadow_refiner/history.csv")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = Path("data/outputs/visualizations/unet_shadow_refiner")
    curves_dir = root / "curves"
    report_dir = root / "report"
    predictions_dir = root / "predictions"
    best_dir = root / "best_cases"
    worst_dir = root / "worst_cases"

    plot_training_curves(args.history, curves_dir)
    eval_args = argparse.Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        size=args.size,
        batch_size=2,
        device=args.device,
        limit=args.max_samples,
        report_dir=str(report_dir),
    )
    evaluate(eval_args)
    visualize_predictions(args.config, args.checkpoint, args.max_samples, args.size, args.device, predictions_dir)
    metrics_path = report_dir / "metrics_summary.csv"
    copy_cases(metrics_path, predictions_dir, best_dir, worst_dir, args.top_k)
    generate_report(args.history, metrics_path, curves_dir, best_dir, worst_dir, report_dir / "training_report.md")
    print(f"Saved complete U-Net result package under: {root}")


if __name__ == "__main__":
    main()
