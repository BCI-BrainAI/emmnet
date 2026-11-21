"""CAM(Class Activation Mapping) 계산 -- GAP 이후 단일 Linear 분류층 구조를 가정.

분류기 자체엔 의존하지 않음: feature map과 class weight 벡터만 받아 계산하므로,
어떤 분류 head(probe든 fusion 이후 joint head든)에도 재사용 가능.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_cam(feature_map: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """feature_map: (B, C, D, H, W), class_weights: (C,) -> (B, D, H, W), 정규화 전 raw CAM."""
    return torch.einsum("bcdhw,c->bdhw", feature_map, class_weights)


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """샘플별 min-max 정규화 -> [0, 1]."""
    b = cam.shape[0]
    flat = cam.view(b, -1)
    min_v = flat.min(dim=1, keepdim=True).values
    max_v = flat.max(dim=1, keepdim=True).values
    norm = (flat - min_v) / (max_v - min_v).clamp(min=1e-8)
    return norm.view_as(cam)


def upsample_cam(cam: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """(B, D, H, W) -> (B, *size), trilinear upsample (입력 해상도에 맞춰 오버레이할 때 사용)."""
    cam = cam.unsqueeze(1)
    up = F.interpolate(cam, size=size, mode="trilinear", align_corners=False)
    return up.squeeze(1)


def generate_cam(
    encoder,
    x: torch.Tensor,
    class_weights: torch.Tensor,
    output_size: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """encoder.forward_feature_map(x) -> CAM 계산 -> 정규화 -> (선택) 입력 해상도로 upsample."""
    feature_map = encoder.forward_feature_map(x)
    cam = compute_cam(feature_map, class_weights)
    cam = normalize_cam(cam)
    if output_size is not None:
        cam = upsample_cam(cam, output_size)
    return cam
