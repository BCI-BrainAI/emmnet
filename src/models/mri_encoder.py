"""
EMMNet MRI Encoder 초안.

논문 근거
---------
- 구조: "The MRI encoder employs 3D convolutions to extract volumetric
  structural features, followed by a projection layer that aligns feature
  dimensionality for multimodal fusion." [EMMNet Sec 3.2, p.284]
- 백본 선정: VGG/ResNet/Transformer 후보 중 "3D-ResNet architecture for MRI...
  we select the ResNet-18-based configuration" [EMMNet Sec 4.2, p.286]
- 초기화: "We initialize the MRI backbone using pretrained weights from [9]"
  (=Med3D, Chen et al. 2019) [EMMNet Sec 4.2, p.286]
- 출력 차원: "The flattened feature dimension n produced by each modality
  encoder is set to 256." [EMMNet Sec 4.2, p.286]
- 입력 전처리: zero-padding 후 256^3 resize, min-max [0,1] 정규화
  [EMMNet Sec 3.1, p.283]
- MRI 브랜치엔 age 정보 미주입 (age는 EEG 채널에만 결합) [EMMNet Sec 3.2, p.284]

Teardown 미해결 사항 (구현 시 반드시 확인)
------------------------------------------
1. projection layer의 activation/dropout 여부: 논문 미기재
   [Not mentioned in context] -> 기본값 ReLU 없이 Linear만 적용, 실험으로 검증.
2. Med3D block3/4 stride-dilation 변형(seg_style)을 classification 전이에도
   유지했는지 불명 -> `ResNet3DBackbone(seg_style=...)`, 실제 체크포인트
   확인 후 결정. resnet3d.py 참고.
3. Med3D 원 사전학습 정규화는 z-score (Eq.2, Med3D p.4)인데 EMMNet은
   min-max 사용 [EMMNet p.283] -> pretrained weight의 입력 분포 가정과
   어긋날 수 있음. fine-tuning 초반 loss spike 가능성 염두.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .resnet3d import ResNet3DBackbone


class MRIEncoder(nn.Module):
    """
    3D-ResNet-18 backbone + global average pooling + projection(256-d).

    Parameters
    ----------
    in_channels : int
        입력 채널 수. T1 단일 채널이므로 기본 1 [EMMNet p.283].
    proj_dim : int
        projection 출력 차원. 논문 값 256 [EMMNet p.286].
    seg_style : bool
        True면 Med3D의 segmentation 전용 stride/dilation 변형 적용.
        기본 False(표준 ResNet 다운샘플) — 체크포인트 확인 전 임시값.
    """

    def __init__(self, in_channels: int = 1, proj_dim: int = 256, seg_style: bool = False):
        super().__init__()
        self.backbone = ResNet3DBackbone(in_channels=in_channels, seg_style=seg_style)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(self.backbone.out_channels, proj_dim)  # n=256 [EMMNet p.286]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, D, H, W) — 학습 시 D=H=W=256 [EMMNet p.283].
           AdaptiveAvgPool3d를 쓰므로 해상도가 달라도 forward는 동작하지만,
           Med3D pretrained weight는 256^3 기준으로 학습되지 않았을 수 있어
           실제 파이프라인에서는 반드시 256^3로 맞출 것.
        반환: (B, proj_dim) — fusion(concat) 직전 feature.
        """
        feat_map = self.backbone(x)          # (B, 512, D', H', W')
        pooled = self.avgpool(feat_map).flatten(1)  # (B, 512)
        return self.projection(pooled)        # (B, proj_dim)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """CAM(Fig.2, p.288-289) 재현용 — pooling 전 마지막 conv feature map 반환."""
        return self.backbone(x)


def load_med3d_pretrained(model: MRIEncoder, checkpoint_path: str | Path,
                           strict: bool = False) -> tuple[list[str], list[str]]:
    """
    Med3D(MedicalNet) 공개 체크포인트를 backbone에 로드.

    주의: 공식 체크포인트의 state_dict 키 이름(prefix, block 명명 규칙)은
    이 저장소의 `ResNet3DBackbone` 모듈 이름과 다를 수 있음 — 아직 실제
    체크포인트를 확보하지 못해 키 매핑을 검증하지 못함 [Not mentioned in context].
    체크포인트 입수 후 아래 TODO를 채울 것.

    반환: (missing_keys, unexpected_keys) — strict=False일 때 로딩 결과 점검용.
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

    # TODO: 공식 체크포인트 키 접두어(예: "module.", "conv_seg.") 제거/매핑.
    # 아직 실제 파일로 검증 전이므로 우선 그대로 시도한다.
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    result = model.backbone.load_state_dict(cleaned, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
