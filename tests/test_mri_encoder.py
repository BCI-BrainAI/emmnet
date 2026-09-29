"""MRI encoder 단위 테스트."""
import pytest
import torch

from models.mri.mri_encoder import MRIEncoder, load_med3d_pretrained
from models.mri.resnet3d import ResNet3DBackbone


def test_mri_encoder_forward_shape():
    """64^3 입력 forward 후 (B, proj_dim) shape 확인."""
    model = MRIEncoder(in_channels=1, proj_dim=256)
    model.eval()
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 256)


def test_mri_encoder_feature_map_for_cam():
    """forward_feature_map이 pooling 이전 feature map을 반환하는지 확인."""
    model = MRIEncoder(in_channels=1, proj_dim=256)
    model.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        feat_map = model.forward_feature_map(x)
    assert feat_map.dim() == 5
    assert feat_map.shape[1] == 512


def test_backbone_fixed_downsample_matches_med3d():
    """64^3 입력이 8^3으로 줄어드는지 확인 (stem+layer1/2에서만 다운샘플)."""
    backbone = ResNet3DBackbone(in_channels=1)
    backbone.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (1, 512, 8, 8, 8)


def test_backbone_shortcut_a_has_no_learnable_params():
    """downsample 경로에 학습 파라미터가 없는지 확인."""
    backbone = ResNet3DBackbone(in_channels=1)
    downsample_params = [
        p for name, p in backbone.named_parameters() if "downsample" in name
    ]
    assert downsample_params == []


from pathlib import Path

_CKPT = Path(__file__).resolve().parent.parent / "pretrained" / "resnet_18.pth"


@pytest.mark.skipif(not _CKPT.exists(), reason="pretrained/resnet_18.pth 없음")
def test_load_real_med3d_checkpoint_has_no_missing_keys():
    """실제 체크포인트 로드 시 missing/unexpected keys가 0이어야 함."""
    model = MRIEncoder(in_channels=1, proj_dim=256)
    missing, unexpected = load_med3d_pretrained(model, _CKPT, strict=False)
    assert missing == [], f"missing_keys가 있음: {missing}"
    assert unexpected == [], f"unexpected_keys가 있음: {unexpected}"
