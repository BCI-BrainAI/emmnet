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

아키텍처 정정 이력 (2026-09-28)
-------------------------------
이전 버전은 백본에 `seg_style` bool 플래그를 두고 표준 ResNet 다운샘플과 Med3D
변형 중 선택하도록 설계했으나, 이는 잘못된 가정이었음. 공식 MedicalNet
(github.com/Tencent/MedicalNet) 구현 확인 결과 resnet-18은 분기 없이 항상
layer3=stride1/dilation2, layer4=stride1/dilation4 고정이며, 공개 pretrained
체크포인트(resnet_18.pth)는 shortcut_type='A'(파라미터 없는 다운샘플)로 학습됨.
이번 리비전에서 `ResNet3DBackbone`을 이 고정 구조에 맞춰 재작성하고 `seg_style`
파라미터를 제거함. 상세: resnet3d.py 모듈 docstring 참고.

Teardown 미해결 사항
--------------------
1. projection layer의 activation/dropout 여부: 논문 미기재
   [Not mentioned in context] -> 기본값 ReLU 없이 Linear만 적용, 실험으로 검증.
2. Med3D 원 사전학습 정규화는 z-score (Eq.2, Med3D p.4)인데 EMMNet은
   min-max 사용 [EMMNet p.283] -> pretrained weight의 입력 분포 가정과
   어긋날 수 있음. fine-tuning 초반 loss spike 가능성 염두.
3. 실제 `resnet_18.pth`(23-dataset 버전) 바이너리를 아직 확보하지 못해
   `load_med3d_pretrained()`의 키 매핑을 실 데이터로 검증하지 못함 -- 구조
   비교는 공개 소스코드 기준으로 완료. 체크포인트 입수 후 missing/unexpected
   keys 로그로 최종 확인 필요.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .resnet3d import ResNet3DBackbone


class MRIEncoder(nn.Module):
    """
    3D-ResNet-18(Med3D 구조 고정) backbone + global average pooling + projection(256-d).

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
        """
        x: (B, in_channels, D, H, W) -- 학습 시 D=H=W=256 [EMMNet p.283].
           AdaptiveAvgPool3d를 쓰므로 해상도가 달라도 forward는 동작하지만,
           Med3D pretrained weight는 256^3 기준으로 학습되지 않았을 수 있어
           실제 파이프라인에서는 반드시 256^3로 맞출 것.
        반환: (B, proj_dim) -- fusion(concat) 직전 feature.
        """
        feat_map = self.backbone(x)          # (B, 512, D', H', W') -- 256^3 입력 시 32^3
        pooled = self.avgpool(feat_map).flatten(1)  # (B, 512)
        return self.projection(pooled)        # (B, proj_dim)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """CAM(Fig.2, p.288-289) 재현용 -- pooling 전 마지막 conv feature map 반환."""
        return self.backbone(x)


def load_med3d_pretrained(model: MRIEncoder, checkpoint_path: str | Path,
                           strict: bool = False) -> tuple[list[str], list[str]]:
    """
    Med3D(MedicalNet) 공개 체크포인트를 backbone에 로드.

    기대 키 매핑 (공식 레포 models/resnet.py 기준, 구조 검증 완료 / 실 바이너리
    미확인):
      - 체크포인트: {'state_dict': {'module.conv1.weight': ..., 'module.layer1.0.conv1.weight': ...,
        ..., 'module.conv_seg.*': ...}} ('module.'는 nn.DataParallel 학습 흔적)
      - 'module.' 제거 후 conv1/bn1/layer1-4.*는 우리 ResNet3DBackbone과 1:1 매치.
      - conv_seg.*(segmentation head)는 우리 backbone에 대응 키가 없어 자동으로
        unexpected_keys 처리 -- classification 전이 목적상 정상.
      - shortcut_type='A' 구간(ShortcutA)은 원래 파라미터가 없으므로 로드 대상에서
        애초에 빠짐 -- missing_keys에 나타나지 않아야 정상.

    반환: (missing_keys, unexpected_keys) -- strict=False일 때 로딩 결과 점검용.
    호출부에서 반드시 missing_keys 길이를 확인할 것 (stem 등 핵심 레이어가 통째로
    빠지면 "로드 성공"처럼 보여도 실질적으로 random init과 동일할 수 있음).
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
