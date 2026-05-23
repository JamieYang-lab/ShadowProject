from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.unet_shadow_dataset import UNetShadowDataset
from losses.mask_losses import bce_dice_loss
from scripts.train_unet_shadow_refiner import dice_score_from_logits, resolve_device
from scripts.visualize_unet_predictions import load_checkpoint_model


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(torch.clamp(preds + targets, 0, 1), dim=dims)
    return ((intersection + eps) / (union + eps)).mean()


def _per_sample_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def _per_sample_iou(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(torch.clamp(preds + targets, 0, 1), dim=dims)
    return (intersection + eps) / (union + eps)


def save_metrics_summary(records: list[dict], report_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_summary.csv"
    json_path = output_dir / "metrics_summary.json"
    fieldnames = ["sample_id", "bce", "dice", "iou", "abs_error_mean", "pred_foreground_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return csv_path, json_path


def evaluate(args) -> None:
    device = resolve_device(args.device)
    dataset = UNetShadowDataset(args.config, size=args.size, strict=False, limit=args.limit)
    if len(dataset) == 0:
        print("No valid samples found for U-Net shadow refiner evaluation.")
        return
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_checkpoint_model(args.checkpoint, device)

    total_bce = total_dice = total_iou = total_ratio = 0.0
    records: list[dict] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["shadow_mask"].to(device)
            logits = model(inputs)
            _, bce, _ = bce_dice_loss(logits, targets)
            batch_size = inputs.shape[0]
            bce_per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(1, 2, 3))
            dice_per_sample = _per_sample_dice(logits, targets)
            iou_per_sample = _per_sample_iou(logits, targets)
            probs = torch.sigmoid(logits)
            pred_binary = (probs >= 0.5).float()
            abs_error = torch.abs(probs - targets).mean(dim=(1, 2, 3))
            foreground_ratio = pred_binary.mean(dim=(1, 2, 3))
            sample_ids = batch["sample_id"]
            for index in range(batch_size):
                records.append(
                    {
                        "sample_id": sample_ids[index],
                        "bce": float(bce_per_sample[index].item()),
                        "dice": float(dice_per_sample[index].item()),
                        "iou": float(iou_per_sample[index].item()),
                        "abs_error_mean": float(abs_error[index].item()),
                        "pred_foreground_ratio": float(foreground_ratio[index].item()),
                    }
                )
            seen += batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice += float(dice_score_from_logits(logits, targets).item()) * batch_size
            total_iou += float(iou_from_logits(logits, targets).item()) * batch_size
            total_ratio += float((torch.sigmoid(logits) >= 0.5).float().mean().item()) * batch_size

    csv_path, json_path = save_metrics_summary(records, args.report_dir)
    print(
        "Evaluation summary: "
        f"samples={seen} "
        f"bce={total_bce / max(seen, 1):.6f} "
        f"dice={total_dice / max(seen, 1):.6f} "
        f"iou={total_iou / max(seen, 1):.6f} "
        f"pred_foreground_ratio={total_ratio / max(seen, 1):.6f}"
    )
    print(f"Saved metrics summary: {csv_path}")
    print(f"Saved metrics summary JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net shadow mask refiner.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset_config.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "unet_shadow_refiner" / "best.pt"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "data" / "outputs" / "visualizations" / "unet_shadow_refiner" / "report"))
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
