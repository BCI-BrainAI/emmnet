"""CAM 계산 단위 테스트."""
import torch

from evaluation.cam import compute_cam, generate_cam, normalize_cam, upsample_cam
from models.mri.mri_encoder import MRIEncoder


def test_compute_cam_shape():
    feature_map = torch.randn(2, 8, 4, 4, 4)
    weights = torch.randn(8)
    cam = compute_cam(feature_map, weights)
    assert cam.shape == (2, 4, 4, 4)


def test_normalize_cam_range():
    cam = torch.randn(2, 4, 4, 4)
    norm = normalize_cam(cam)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0 + 1e-6


def test_upsample_cam_matches_target_size():
    cam = torch.randn(1, 4, 4, 4)
    up = upsample_cam(cam, (8, 8, 8))
    assert up.shape == (1, 8, 8, 8)


def test_generate_cam_end_to_end():
    encoder = MRIEncoder(in_channels=1, proj_dim=256)
    encoder.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    class_weights = torch.randn(encoder.backbone.out_channels)
    with torch.no_grad():
        cam = generate_cam(encoder, x, class_weights, output_size=(64, 64, 64))
    assert cam.shape == (1, 64, 64, 64)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6
