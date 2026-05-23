from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from datasets.unet_shadow_dataset import UNetShadowDataset
from losses.mask_losses import bce_dice_loss
from models.unet_shadow_refiner import UNetShadowRefiner
from scripts.visualize_unet_predictions import save_prediction_panel


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def save_checkpoint(path: Path, model: UNetShadowRefiner, optimizer: torch.optim.Optimizer, epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "input_channels": 18,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_history(output_dir: Path, history: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "history.csv"
    json_path = output_dir / "history.json"
    fieldnames = [
        "epoch",
        "train_total_loss",
        "train_bce_loss",
        "train_dice_loss",
        "train_dice_score",
        "val_total_loss",
        "val_dice",
        "val_iou",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in fieldnames})
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def train(args) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = UNetShadowDataset(args.config, size=args.size, strict=args.strict, limit=args.limit)
    if len(dataset) == 0:
        print("No valid samples found for U-Net shadow refiner training.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = UNetShadowRefiner(input_channels=18).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    vis_dir = PROJECT_ROOT / "data" / "outputs" / "visualizations" / "unet_shadow_refiner" / "predictions"
    best_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_bce = total_dice_loss = total_dice_score = 0.0
        seen = 0
        first_batch = None
        first_logits = None
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["shadow_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss, bce, dice = bce_dice_loss(logits, targets, args.dice_weight)
            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            seen += batch_size
            total_loss += float(loss.item()) * batch_size
            total_bce += float(bce.item()) * batch_size
            total_dice_loss += float(dice.item()) * batch_size
            total_dice_score += float(dice_score_from_logits(logits.detach(), targets).item()) * batch_size
            if first_batch is None:
                first_batch = batch
                first_logits = logits.detach().cpu()

        mean_loss = total_loss / max(seen, 1)
        mean_bce = total_bce / max(seen, 1)
        mean_dice_loss = total_dice_loss / max(seen, 1)
        mean_dice_score = total_dice_score / max(seen, 1)
        history_row = {
            "epoch": epoch,
            "train_total_loss": mean_loss,
            "train_bce_loss": mean_bce,
            "train_dice_loss": mean_dice_loss,
            "train_dice_score": mean_dice_score,
            "val_total_loss": None,
            "val_dice": None,
            "val_iou": None,
        }
        history.append(history_row)
        save_history(output_dir, history)
        print(
            f"epoch={epoch} train_loss={mean_loss:.6f} "
            f"bce={mean_bce:.6f} "
            f"dice_loss={mean_dice_loss:.6f} "
            f"dice_score={mean_dice_score:.6f}"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, mean_loss)
        if first_batch is not None and first_logits is not None:
            sample = {key: value[0] if torch.is_tensor(value) else value[0] for key, value in first_batch.items()}
            save_prediction_panel(sample, first_logits[0], vis_dir / f"epoch_{epoch:03d}_{sample['sample_id']}_prediction.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight U-Net shadow mask refiner.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "dataset_config.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints" / "unet_shadow_refiner"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
