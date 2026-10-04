"""손실 함수."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """FL = -alpha_t (1-p_t)^gamma log(p_t). alpha는 양성(label=1) 가중치."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
