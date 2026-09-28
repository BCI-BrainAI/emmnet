"""
3D ResNet 공통 빌딩 블록.

Med3D(Chen et al., 2019, arXiv:1904.00625)가 2D ResNet 계열을 3D 의료영상에
맞게 변형한 방식을 따른다:
  1) stem conv 입력 채널 3 -> 1 (단일 채널 볼륨)               [Med3D Sec 3.2, p.6]
  2) 모든 2D conv 커널 -> 3D 버전으로 교체                     [Med3D Sec 3.2, p.6]
  3) (segmentation 사전학습 전용) block3/block4 stride=1 +
     dilation=2 로 다운샘플 방지, receptive field는 유지        [Med3D Sec 3.2, p.6]

주의(Teardown): 3)은 논문에서 "encoder-decoder segmentation 학습" 맥락에서만
명시됨. classification transfer(예: EMMNet의 MRI encoder)에 이 변형을 그대로
쓰는지는 Med3D/EMMNet 둘 다 미기재 [Not mentioned in context]. 실제 배포된
pretrained 체크포인트의 구조를 먼저 확인하고 `seg_style` 값을 정할 것.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    """표준 3D ResNet BasicBlock (2x Conv3d+BN3d, residual add)."""

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


def make_layer(in_channels: int, out_channels: int, blocks: int,
               stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """BasicBlock3D `blocks`개를 쌓아 하나의 ResNet stage를 만든다."""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels),
        )
    layers = [BasicBlock3D(in_channels, out_channels, stride, dilation, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock3D(out_channels, out_channels, 1, dilation))
    return nn.Sequential(*layers)


class ResNet3DBackbone(nn.Module):
    """
    3D-ResNet-18 backbone (stem + layer1-4), stem/layer1/2는 항상 표준(stride 2),
    layer3/4는 `seg_style`에 따라 stride 2(표준) 또는 stride1+dilation2(Med3D seg) 선택.

    forward 출력: (B, 512, D', H', W') — pooling/projection은 상위 encoder에서 처리.
    """

    def __init__(self, in_channels: int = 1, seg_style: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # [Med3D p.6] ch 3->1
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        s34, d34 = (1, 2) if seg_style else (2, 1)   # [Med3D Sec 3.2, p.6] segmentation 전용 변형
        self.layer1 = make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = make_layer(128, 256, blocks=2, stride=s34, dilation=d34)
        self.layer4 = make_layer(256, 512, blocks=2, stride=s34, dilation=d34)
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
