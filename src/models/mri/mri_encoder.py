"""MRI encoder: backbone + projection."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .resnet3d import ResNet3DBackbone


class MRIEncoder(nn.Module):
    """backbone -> global average pooling -> linear projection으로 고정 길이 임베딩 생성."""

    def __init__(self, in_channels: int = 1, proj_dim: int = 256):
        super().__init__()
        self.backbone = ResNet3DBackbone(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(self.backbone.out_channels, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,D,H,W) -> (B, proj_dim) 임베딩."""
        feat_map = self.backbone(x)
        pooled = self.avgpool(feat_map).flatten(1)
        return self.projection(pooled)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """pooling 이전 feature map (CAM 계산용)."""
        return self.backbone(x)


def load_med3d_pretrained(model: MRIEncoder, checkpoint_path: str | Path,
                           strict: bool = False) -> tuple[list[str], list[str]]:
    """Med3D 포맷 체크포인트({'state_dict': {'module.<key>': tensor}})를 backbone에 로드.

    반환된 missing/unexpected가 둘 다 비어있는지 항상 확인할 것 --
    strict=False라 키가 안 맞아도 조용히 넘어간다.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    result = model.backbone.load_state_dict(cleaned, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
