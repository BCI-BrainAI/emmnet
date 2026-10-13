# 실행 검증 결과

검증일: 2026-10-13 (한국 시간)

- Python 3.12, torch 2.14.1, numpy 2.5.4, scipy 1.18.1, mne 1.13.2,
  pytest 9.1.1, CPU 환경에서 실행.
- `python -m pytest -q`: **8 passed in 5.85s**.
- 필터링/200 Hz 리샘플링 및 채널 매핑, 누락 채널 거부 확인.
- 학습 환자만 사용한 신호/나이 통계와 deterministic 평가 크롭 확인.
- 정확히 crop 길이인 신호 지원 및 짧은 기록 거부 확인.
- split 간 환자 중복 거부, 누락 나이 거부, annotation 나이 연결 확인.
- focal loss의 수식과 gradient 확인.
- 나이 채널 gradient, 256차원 특징, EEG/MRI concat 출력, checkpoint 복원 동등성 확인.
- 원본 ResNet1D에 동일 state_dict를 넣어 adapter 출력 동등성 확인.
- MNE EDF reader 경로는 mocked reader로 확인; 실제 EDF 파일 검증은 미실시.
- ECG 포함 20채널 입력 / 3클래스 출력 `[1,3]` 별도 순전파 확인.

## 전체 실행 경로

총 8명의 합성 기록(학습4/검증2/테스트2, 각각 12초, 19채널)을 만들어
batch2, epoch1, eval crops2로 실제 `run_eeg.py` 명령을 실행했습니다.

1. train: focal loss 역전파와 AdamW 업데이트, validation 선택, best.pt 저장.
2. evaluate: best.pt 복원, 환자별 확률/혼동행렬 JSON 생성.
3. features: test_features.npz 생성, `[2,256]`, 환자 ID `[7,8]`, 유한 값 확인.

train loss=0.0415876938495785, validation/test accuracy=.5였으며,
이는 임의 합성 데이터의 실행 결과입니다. 실제 CAUEMM 성능과 관계없습니다.
합성 데이터와 가중치는 검증용 임시 파일로만 사용했고 배포물에 넣지 않았습니다.

## 모델 크기 및 미검증 범위

기본 19채널 EEG binary classifier: **11,421,954 parameters**.
논문 표 4 EEG-only의 25.7M과 일치하지 않습니다. 상세 백본 설정을 공개 자료만으로
확정할 수 없어 원본 브랜치 ResNet-18 기본 구성을 사용했습니다.

실제 EDF/환자 데이터 학습, GPU 실행, 장기간 학습, 논문 수치 재현,
MRI 인코더와 실제 환자별 pairing은 미검증입니다.

## 서버 CAUEEG 실행 결과 (사용자 제공 화면)

사용자가 제공한 서버 실행 화면에서 다음을 추가 확인했습니다.
Python 3.9.25, PyTorch 2.5.1+cu121, MNE 1.8.0 환경이며,
GPU 1번을 지정하는 명령으로 학습64/검증32/테스트32명, batch16,
5 epoch, 평가2-crop 실험을 진행했습니다. GPU 프로세스 자체는 별도 확인하지 않았습니다.

- 학습 loss: 약 .0687 -> .0362.
- 최고 검증 정확도: epoch4, .625.
- 저장된 최선 모델의 테스트 정확도: .65625 (21/32).
- 클래스별 recall: [.375, .9375].
- confusion matrix (행 정답, 열 예측): [[6,10],[1,15]].

이 결과는 실제 EDF 전처리, 학습, 체크포인트 저장·복원, 테스트 평가가
실행되었다는 근거입니다. 위 미검증 범위 중 실제 EDF와 GPU 명령 실행에 대한
후속 확인이며, 독립적으로 서버를 검사한 결과는 아닙니다.
실제 데이터의 256차원 특징 export 결과는 아직 전달받지 않았습니다.
소규모 단일 실행 결과로 논문 성능 재현이나 일반화 성능을 주장하지 않습니다.
