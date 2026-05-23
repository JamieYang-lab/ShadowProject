from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    dice = dice_loss(logits, targets)
    total = bce + float(dice_weight) * dice
    return total, bce, dice
