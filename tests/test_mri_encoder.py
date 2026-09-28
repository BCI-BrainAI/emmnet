"""MRI Encoder 초안 단위 테스트 (shape/forward 검증용, 실제 데이터 불필요).

경로 설정은 conftest.py에서 일괄 처리 (pip install -e . 했다면 그것도 불필요).
"""
import pytest
import torch

from models.mri_encoder import MRIEncoder
from models.resnet3d import ResNet3DBackbone


@pytest.mark.parametrize("seg_style", [False, True])
def test_mri_encoder_forward_shape(seg_style):
    """작은 볼륨(64^3)으로 forward 후 (B, proj_dim) shape 확인. 256^3은 CI에 과함."""
    model = MRIEncoder(in_channels=1, proj_dim=256, seg_style=seg_style)
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


def test_backbone_seg_style_changes_output_resolution():
    """seg_style=True면 block3/4가 다운샘플하지 않아 표준 대비 출력 해상도가 커야 함."""
    x = torch.randn(1, 1, 64, 64, 64)
    standard = ResNet3DBackbone(in_channels=1, seg_style=False)
    seg = ResNet3DBackbone(in_channels=1, seg_style=True)
    standard.eval(); seg.eval()
    with torch.no_grad():
        out_std = standard(x)
        out_seg = seg(x)
    assert out_seg.shape[-1] > out_std.shape[-1]
