"""MRI Encoder 단위 테스트 (shape/forward 검증용, 실제 데이터 불필요).

경로 설정: 저장소 루트에서 `PYTHONPATH=src pytest` 등으로 src/를 sys.path에
추가해야 함 (pyproject.toml/conftest.py 없음 -- 2026-09 리비전에서 제거됨).
"""
import pytest
import torch

from models.mri_encoder import MRIEncoder
from models.resnet3d import ResNet3DBackbone


def test_mri_encoder_forward_shape():
    """작은 볼륨(64^3)으로 forward 후 (B, proj_dim) shape 확인. 256^3은 CI에 과함."""
    model = MRIEncoder(in_channels=1, proj_dim=256)
    model.eval()
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 256)


def test_mri_encoder_feature_map_for_cam():
    """CAM 재현(Fig.2, p.288-289)용 forward_feature_map이 pooling 전 map을 반환하는지 확인."""
    model = MRIEncoder(in_channels=1, proj_dim=256)
    model.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        feat_map = model.forward_feature_map(x)
    assert feat_map.dim() == 5  # (B, C, D, H, W)
    assert feat_map.shape[1] == 512


def test_backbone_fixed_downsample_matches_med3d():
    """MedicalNet resnet-18 고정 구조 회귀 테스트: stem+layer1/2에서만 다운샘플,
    layer3/4는 dilation으로 해상도 유지 -> 64^3 입력이 8^3으로 줄어드는지 확인
    (256^3 기준으로는 32^3). 값이 바뀌면 Med3D 아키텍처와 어긋난 것.
    """
    backbone = ResNet3DBackbone(in_channels=1)
    backbone.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (1, 512, 8, 8, 8)


def test_backbone_shortcut_a_has_no_learnable_params():
    """resnet-18 pretrained가 shortcut_type='A'로 학습됐으므로, 다운샘플 경로에
    학습 파라미터가 없어야 pretrained 키 매핑과 어긋나지 않음."""
    backbone = ResNet3DBackbone(in_channels=1)
    downsample_params = [
        p for name, p in backbone.named_parameters() if "downsample" in name
    ]
    assert downsample_params == []


# --- 실제 Med3D 체크포인트 회귀 테스트 -------------------------------------
# pretrained/resnet_18.pth는 용량(126MB) 때문에 .gitignore 대상 -- 로컬에
# 파일이 없는 환경(다른 팀원 PC, CI)에서는 자동 skip. 2026-09-28 최초 실측
# 검증: missing_keys=0, unexpected_keys=0 (Med3D 8-dataset 원본 기준).
import os
from pathlib import Path

from models.mri_encoder import load_med3d_pretrained

_CKPT = Path(__file__).resolve().parent.parent / "pretrained" / "resnet_18.pth"


@pytest.mark.skipif(not _CKPT.exists(), reason="pretrained/resnet_18.pth 없음 (.gitignore 대상, 로컬에 직접 배치 필요)")
def test_load_real_med3d_checkpoint_has_no_missing_keys():
    """실제 Med3D resnet_18.pth 로드 시 missing/unexpected keys가 0이어야 함
    -- resnet3d.py 아키텍처가 공식 배포 체크포인트와 어긋나면 이 테스트가 깨짐."""
    model = MRIEncoder(in_channels=1, proj_dim=256)
    missing, unexpected = load_med3d_pretrained(model, _CKPT, strict=False)
    assert missing == [], f"missing_keys가 있음: {missing}"
    assert unexpected == [], f"unexpected_keys가 있음: {unexpected}"
