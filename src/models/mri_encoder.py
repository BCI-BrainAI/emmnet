"""
EMMNet MRI Encoder.

논문 근거
---------
- 구조: "The MRI encoder employs 3D convolutions to extract volumetric
  structural features, followed by a projection layer that aligns feature
  dimensionality for multimodal fusion." [EMMNet Sec 3.2, p.284]
- 백본 선정: "3D-ResNet architecture for MRI... we select the ResNet-18-based
  configuration" [EMMNet Sec 4.2, p.286]
- 초기화: "We initialize the MRI backbone using pretrained weights from [9]"
  (=Med3D, Chen et al. 2019) [EMMNet Sec 4.2, p.286] -- 아키텍처/가중치
  정합성 검증은 resnet3d.py, `load_med3d_pretrained()` 참고.
- 출력 차원: "The flattened feature dimension n produced by each modality
  encoder is set to 256." [EMMNet Sec 4.2, p.286]
- 입력 전처리: zero-padding 후 256^3 resize, min-max [0,1] 정규화
  [EMMNet Sec 3.1, p.283]
- MRI 브랜치엔 age 정보 미주입 (age는 EEG 채널에만 결합) [EMMNet Sec 3.2, p.284]

미해결 사항
-----------
1. projection layer의 activation/dropout 여부: 논문 미기재
   [Not mentioned in context] -> 기본값 ReLU 없이 Linear만 적용, 실험으로 검증.
2. Med3D 원 사전학습 정규화는 z-score (Eq.2, Med3D p.4)인데 EMMNet은
   min-max 사용 [EMMNet p.283] -> pretrained weight의 입력 분포 가정과
   어긋날 수 있음. fine-tuning 초반 loss spike 가능성 염두.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .resnet3d import ResNet3DBackbone


class MRIEncoder(nn.Module):
    """3D-ResNet-18(Med3D 구조) backbone + global average pooling + projection.

    Parameters
    ----------
    in_channels : int
        입력 채널 수. T1 단일 채널이므로 기본 1 [EMMNet p.283].
    proj_dim : int
        projection 출력 차원. 논문 값 256 [EMMNet p.286].
    """

    def __init__(self, in_channels: int = 1, proj_dim: int = 256):
        super().__init__()
        self.backbone = ResNet3DBackbone(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(self.backbone.out_channels, proj_dim)  # n=256 [EMMNet p.286]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, D, H, W), 학습 시 D=H=W=256 [EMMNet p.283].
        Med3D pretrained weight가 256^3 기준으로 학습됐으므로 실 파이프라인
        에서는 반드시 이 해상도를 맞출 것 (forward 자체는 다른 크기도 동작함).

        반환: (B, proj_dim) -- fusion(concat) 직전 feature.
        """
        feat_map = self.backbone(x)                 # (B, 512, D', H', W') -- 256^3 입력 시 32^3
        pooled = self.avgpool(feat_map).flatten(1)   # (B, 512)
        return self.projection(pooled)               # (B, proj_dim)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """CAM(Fig.2, p.288-289) 재현용 -- pooling 전 마지막 conv feature map 반환."""
        return self.backbone(x)


def load_med3d_pretrained(model: MRIEncoder, checkpoint_path: str | Path,
                           strict: bool = False) -> tuple[list[str], list[str]]:
    """Med3D 공개 체크포인트를 backbone에 로드.

    체크포인트 형태: {'state_dict': {'module.<key>': tensor, ...}}
    ('module.'는 nn.DataParallel 학습으로 붙은 prefix). 이를 제거하면
    conv1/bn1/layer1-4.* 키가 ResNet3DBackbone과 1:1 매칭됨 -- 실측 검증
    (`pretrain/resnet_18.pth`): missing_keys=0, unexpected_keys=0, 102/102
    키 shape까지 일치.

    반환: (missing_keys, unexpected_keys). strict=False라 키가 안 맞아도
    조용히 넘어가므로, 호출부에서 반드시 이 두 리스트 길이를 확인할 것 --
    확인 없이는 "로드 성공"처럼 보여도 사실상 random init일 수 있음.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Med3D checkpoint 없음: {checkpoint_path}. "
            "pretrained/ 폴더에 체크포인트 배치 후 재시도."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    result = model.backbone.load_state_dict(cleaned, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
