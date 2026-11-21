"""MRI 단독 평가용 분류기: MRIEncoder + 선형 probe head.

fusion 이전에 MRI encoder 단독 성능을 재기 위한 래퍼. 이진 로짓 1개를 출력한다.
(label: CN=0, MCI/AD=1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mri_encoder import MRIEncoder


class MRIClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, proj_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.encoder = MRIEncoder(in_channels=in_channels, proj_dim=proj_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(proj_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,D,H,W) -> (B,) 로짓."""
        return self.head(self.dropout(self.encoder(x))).squeeze(1)
