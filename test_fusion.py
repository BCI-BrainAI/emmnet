import torch
import sys
import os

sys.path.append(os.path.abspath("."))

from src.models.eeg_encoder import EEGEncoder
from src.models.mri.mri_encoder import MRIEncoder
from src.models.fusion import EMMNetFusion

def test_full_fusion():
    print("=" * 50)
    print("EMMNet 멀티모달 퓨전 파이프라인 통합 테스트 시작")
    print("=" * 50)

    # 1. 인코더 2개 생성
    eeg_enc = EEGEncoder(eeg_channels=19, seq_length=2000)
    mri_enc = MRIEncoder() # 기본 3D ResNet-18 기반

    # 2. 통합 융합 모델 생성
    model = EMMNetFusion(eeg_enc, mri_enc, num_classes=2, freeze_backbones=True)
    model.eval()

    # 3. 가상 환자 2명의 입력 생성
    # EEG: [2, 19, 2000], Age: [2], MRI: [2, 1, 256, 256, 256] (테스트를 위해 작은 크기로 축소 가능)
    dummy_eeg = torch.randn(2, 19, 2000)
    dummy_age = torch.tensor([65.0, 72.0])
    # 3D 볼륨 연산 메모리를 고려해 스모크 테스트용 작은 볼륨(32^3 또는 원본 256^3) 투입
    dummy_mri = torch.randn(2, 1, 64, 64, 64) 

    print("모델에 가상 EEG, Age, MRI 데이터 통과 중...")
    with torch.no_grad():
        out = model(dummy_eeg, dummy_age, dummy_mri)

    print("최종 출력 텐서 형태 (Shape):", out.shape)
    if out.shape == (2, 2):
        print("\n🎉 축하합니다! EEG + MRI 퓨전 모델 결합이 완벽히 성공했습니다!")
    else:
        print(f"\n⚠️ 예상과 다른 출력 형태: {out.shape}")

if __name__ == "__main__":
    test_fusion_output = test_full_fusion
    test_fusion_output()
