"""
3D ResNet 공통 빌딩 블록 -- MedicalNet(Med3D, Chen et al. 2019) 공개 구현 기준.

검증 출처: github.com/Tencent/MedicalNet models/resnet.py (2026-09-28 클론 확인)

Med3D가 2D ResNet을 3D 의료영상용으로 변형한 지점 [MedicalNet models/resnet.py]:
  1) stem conv 입력 채널 3 -> 1                                    [L126-132]
  2) 모든 2D conv -> 3D                                            [전역]
  3) layer3/layer4는 항상 stride=1 + dilation(각각 2, 4) 고정      [L140-143]
     -> "segmentation 전용 옵션"이 아니라 공식 구현에 분기 자체가 없음.
        classification 전이 시에도 이 구조 그대로 써야 Med3D pretrained
        weight와 아키텍처가 일치함.
  4) resnet-18/34 공개 체크포인트는 shortcut_type='A'
     (파라미터 없는 avg_pool3d+zero-pad shortcut)                  [README L26-27]
     -> resnet-10/50/101/... 은 'B'(학습 가능 1x1 conv) 사용, 우리는 18만 다룸.

Teardown(정정 이력, 2026-09-28): 이전 버전은 `seg_style` bool로 표준 stride-2
다운샘플과 Med3D 변형을 토글하도록 설계했으나 이는 잘못된 가정이었음. 실제로는
토글 대상 자체가 없고, resnet-18 pretrained weight를 쓰려면 항상 위 고정 구조를
써야 함. 이번 리비전에서 `seg_style` 제거.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    """표준 3D ResNet BasicBlock (2x Conv3d+BN3d, residual add).

    conv1/bn1/conv2/bn2/downsample 네이밍은 MedicalNet 체크포인트의 state_dict
    키(`layer{1..4}.{i}.conv1.weight` 등)와 그대로 맞춰 pretrained 로딩을 보장.
    """

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
    """MedicalNet shortcut_type='A': 학습 파라미터 없는 다운샘플.

    avg_pool3d(kernel=1, stride=stride)로 공간 해상도만 줄이고, 채널 증가분은
    0으로 zero-pad. resnet-18/34 공개 체크포인트가 이 방식으로 학습됨
    [MedicalNet models/resnet.py L26-37, README L26-27] -- 즉 이 경로엔 애초에
    대응되는 pretrained 파라미터가 존재하지 않음(로드 대상에서 항상 제외됨).
    """

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


def make_layer(in_channels: int, out_channels: int, blocks: int,
               stride: int = 1, dilation: int = 1,
               shortcut_type: str = "A") -> nn.Sequential:
    """BasicBlock3D `blocks`개를 쌓아 하나의 ResNet stage 구성.

    shortcut_type: 'A'(파라미터 없음, resnet-18/34 pretrained 기준) | 'B'(학습
    가능 1x1 conv+BN, resnet-10/50+ 기준) [MedicalNet README L25-28].
    """
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
    """
    3D-ResNet-18 backbone, MedicalNet(Med3D) resnet-18 공개 구현과 구조 동일
    [MedicalNet models/resnet.py L112-176].

    고정 구성 (분기 없음):
      stem  : Conv3d(k7,s2) -> BN -> ReLU -> MaxPool3d(k3,s2)
      layer1: stride1, dilation1               (64ch)
      layer2: stride2, dilation1               (128ch)
      layer3: stride1, dilation2               (256ch)  <- Med3D 고정값
      layer4: stride1, dilation4               (512ch)  <- Med3D 고정값
      shortcut_type='A' (파라미터 없는 다운샘플, resnet-18 pretrained 기준)

    forward 출력: (B, 512, D', H', W'). 256^3 입력 기준 D'=H'=W'=32
    (stem/layer1/layer2에서만 실제 다운샘플, layer3/4는 dilation으로 해상도 유지).

    속성 이름(conv1/bn1/maxpool/layer1-4)은 MedicalNet state_dict 키와 정확히
    일치시켜 `load_med3d_pretrained()`에서 `module.` prefix만 벗기면 바로
    매칭되게 함(이전 버전의 `self.stem` Sequential 래핑을 제거한 이유).
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_layer(64, 64, blocks=2, stride=1, dilation=1)
        self.layer2 = make_layer(64, 128, blocks=2, stride=2, dilation=1)
        self.layer3 = make_layer(128, 256, blocks=2, stride=1, dilation=2)  # [MedicalNet L140]
        self.layer4 = make_layer(256, 512, blocks=2, stride=1, dilation=4)  # [MedicalNet L142]
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
