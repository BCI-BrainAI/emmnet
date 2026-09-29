"""MRI encoder의 3D ResNet-18 backbone."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    """3x3x3 conv+BN 두 개와 residual 덧셈으로 구성된 기본 블록."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ShortcutA(nn.Module):
    """학습 파라미터 없는 skip 경로 -- avgpool로 공간 축소, zero-pad로 채널 확장."""

    def __init__(self, out_channels: int, stride: int):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        pad_channels = self.out_channels - out.shape[1]
        if pad_channels > 0:
            zero_pad = out.new_zeros(out.shape[0], pad_channels, *out.shape[2:])
            out = torch.cat([out, zero_pad], dim=1)
        return out


def make_stem(in_channels: int) -> tuple[nn.Conv3d, nn.BatchNorm3d, nn.ReLU, nn.MaxPool3d]:
    """입력 -> 64ch, 공간 1/4 축소하는 stem 구성요소 (conv1, bn1, relu, maxpool).

    모듈 하나로 묶지 않고 튜플로 반환한다: backbone에 conv1/bn1으로 바로 붙어야
    state_dict key가 Med3D 체크포인트(conv1.weight, bn1.*)와 일치한다.
    """
    conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    bn1 = nn.BatchNorm3d(64)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
    return conv1, bn1, relu, maxpool


def make_layer(in_channels: int, out_channels: int, blocks: int,
               stride: int = 1, dilation: int = 1,
               shortcut_type: str = "A") -> nn.Sequential:
    """BasicBlock3D를 blocks개 쌓아 ResNet stage 하나를 구성."""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        if shortcut_type == "A":
            downsample = ShortcutA(out_channels, stride)
        else:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
    layers = [BasicBlock3D(in_channels, out_channels, stride, dilation, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock3D(out_channels, out_channels, 1, dilation))
    return nn.Sequential(*layers)


class ResNet3DBackbone(nn.Module):
    """stem + layer1~4. 입력 (B,C,D,H,W) -> 출력 (B,512,D',H',W').

    layer3/4는 stride=1 대신 dilation(2,4)으로 해상도를 유지한다.
    shortcut_type='A' 고정 -- 공개 pretrained 체크포인트와 구조를 맞추기 위함.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = make_stem(in_channels)

        self.layer1 = make_layer(64, 64, blocks=2, stride=1, dilation=1)
        self.layer2 = make_layer(64, 128, blocks=2, stride=2, dilation=1)
        self.layer3 = make_layer(128, 256, blocks=2, stride=1, dilation=2)
        self.layer4 = make_layer(256, 512, blocks=2, stride=1, dilation=4)
        self.out_channels = 512

    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        """stem 통과: (B,C,D,H,W) -> (B,64,D/4,H/4,W/4)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.maxpool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
