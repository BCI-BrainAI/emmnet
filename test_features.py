import torch
# eeg_encoder.py가 있는 위치에 맞게 import 경로를 설정합니다.
# src/models/eeg_encoder.py 라면 아래와 같이 작성:
from src.models.eeg_encoder import EEGEncoder

def test_encoder_output():
    # 1. 모델 인스턴스 생성 (19개 EEG 채널, 2000 타임스텝)
    model = EEGEncoder(eeg_channels=19, seq_length=2000)
    model.eval()  # 평가 모드로 전환 (BatchNorm 고정)

    # 2. 가상의 입력 텐서 생성 (Batch 크기 2명 가정)
    # [배치 크기, 채널 수, 시퀀스 길이] -> [2, 19, 2000]
    dummy_eeg = torch.randn(2, 19, 2000)
    # [배치 크기] -> [2] (환자 나이)
    dummy_age = torch.tensor([65.0, 72.0])

    # 3. 모델에 입력 통과 (추론 모드)
    with torch.no_grad():
        output = model(dummy_eeg, dummy_age)

    # 4. 결과 출력
    print("=" * 40)
    print("EEG 인코더 출력 텐서 형태 (Shape):", output.shape)
    print("출력 벡터 미리보기 (첫 번째 환자 앞 5개 값):", output[0, :5].tolist())
    print("=" * 40)

    # 5. EMMNet 규격(256차원) 일치 여부 판별
    expected_shape = (2, 256)
    if output.shape == expected_shape:
        print("✅ 검증 성공: EMMNet 규격인 [Batch, 256] 차원으로 정확히 출력됩니다!")
    else:
        print(f"❌ 검증 실패: 예상 차원은 {expected_shape}이지만 {output.shape}가 출력되었습니다.")

if __name__ == "__main__":
    test_encoder_output()