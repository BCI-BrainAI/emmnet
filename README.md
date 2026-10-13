# EMMNet EEG 구현

`BCI-BrainAI/emmnet`의 **feature/eeg-preprocessing** 브랜치
커밋 `2550cb46f7a0dbfd2403661217d41fe8520d591e`를 가져와,
제공된 EMMNet 논문의 EEG 전처리 → 인코더 → 학습 → 평가 → 특징 추출을 연결했습니다.
기존 로컬 `emmnet_eeg.py`의 인터페이스를 유지하며 빠졌던 원본 백본도 포함했습니다.
새 실행 진입점은 **run_eeg.py**입니다. 기존 run_train.py, notebook, test_*.py는
브랜치의 비교용 원본이며 새 경로와 설정을 공유하지 않습니다. 원본 설명은 README.upstream.md에 있습니다.

## 설치 및 검증

Python 3.10 이상에서 이 폴더를 터미널로 열고 실행합니다.

```powershell
python -m pip install -r requirements-eeg.txt
python -m pytest
```

CPU에서도 실행할 수 있습니다. GPU 학습에는 해당 GPU와 호환되는 PyTorch를 설치하세요.
실제 데이터는 이 배포물에 포함하지 않았습니다. 검증 결과는 VALIDATION.md를 확인하세요.

## 입력 데이터

CAUEEG 형식 JSON의 `train_split`, `validation_split`, `test_split`을 사용합니다.
각 항목은 `serial`, `class_label`, `age`가 필요합니다. 나이가 split JSON에 없으면
`--annotation annotation.json`으로 `data` 안의 `serial`, `age`를 연결합니다.
나이를 임의로 0으로 채우지 않습니다. 환자 중복이나 split 간 겹침, 누락된 파일,
잘못된 라벨, 누락된 채널은 오류로 보고합니다. 여러 recording을 가진 동일 환자는
사전에 환자 단위로 정리해야 합니다. `serial`은 환자를 식별하는 키여야 합니다.

```json
{
  "train_split": [
    {"serial": "00001", "class_label": 0, "age": 55},
    {"serial": "00002", "class_label": 1, "age": 75}
  ],
  "validation_split": [{"serial": "00003", "class_label": 0, "age": 60}],
  "test_split": [{"serial": "00004", "class_label": 1, "age": 80}]
}
```

기본 EDF 경로는 `--eeg-dir` 아래 `00001.edf`입니다. `eeg_file` 필드로 파일명을
명시할 수 있습니다. NPZ도 지원하며 `signal` [채널, 시간] **volt 단위**,
`sfreq` 원본 샘플링 주파수, `channel_names` 문자열 배열을 저장해야 합니다.
EDF는 MNE가 volt로 읽습니다. 원본 샘플링 주파수는 140 Hz 초과여야 합니다.
10초 미만 기록은 거부하며 자동 zero padding으로 실제 기록처럼 학습시키지 않습니다.

기본 EEG 채널 순서는 논문에 적힌 다음 19개입니다.
`Fp1 F3 C3 P3 O1 Fp2 F4 C4 P4 O2 F7 T3 T5 F8 T4 T6 Fz Cz Pz`
T7/P7/T8/P8 별칭을 T3/T5/T4/T6으로 처리합니다. `EEG Fp1-REF` 등의 이름도
지원합니다. 입력은 이미 linked-ear reference로 기록된 신호를 가정합니다.
다른 reference를 자동으로 재구성하지 않습니다. 추가 ECG 입력은 `--include-ecg`로
선택하고 광자극 채널은 제외합니다. 이 선택은 학습 checkpoint에 저장됩니다.

## 학습

이진 abnormal 작업은 원본 abnormal.json의 라벨 의미를 확인하고 사용하세요.
논문의 sensitivity 비교에서는 abnormal이 라벨 1이어야 합니다.
Normal/MCI/Dementia 3분류 데이터는 `--num-classes 3`을 지정합니다.
라벨을 자동으로 이진/3분류로 바꾸지 않습니다.

```powershell
python run_eeg.py train --split-json "D:/CAUEEG/abnormal.json" --annotation "D:/CAUEEG/annotation.json" --eeg-dir "D:/CAUEEG/signal/edf" --output "local/eeg-run" --epochs 100 --batch-size 16
```

학습 환자 전체 신호에서 채널별 평균/표준편차, 학습 환자의 나이 평균/표준편차를
계산합니다. 검증·테스트는 해당 통계를 재사용합니다. 전처리는 전체 기록을 필터링하고
리샘플링한 다음 학습에서는 매번 무작위 2000점 크롭을 뽑습니다. 마지막 가능한 크롭과
정확히 2000점인 입력도 지원합니다. 정규화 뒤 학습 신호에만 Gaussian noise를 추가합니다.
ICA, blink 제거 등의 명시적 artifact 제거는 하지 않습니다.

AdamW, beta=(0.9,0.999), focal loss alpha=.25/gamma=2, warmup + cosine decay,
기본 batch=16을 사용합니다. FC BatchNorm을 위해 학습의 마지막 미완성 배치는 버립니다.
학습 환자 수가 batch보다 적으면 batch 크기를 2 이상으로 줄이세요.
검증 정확도로 선택한 `best.pt`에 모델, 통계, 옵션, 환자 분할을 저장합니다.
`history.json`과 최선 checkpoint의 `test_metrics.json`도 생성합니다.
이 실행 경로는 단일 GPU/CPU용이며 학습 재개와 DDP는 지원하지 않습니다.
통계 계산과 EDF 전처리는 매번 기록을 읽으므로 대규모 데이터에서는 느릴 수 있습니다.

## 평가 및 MRI 결합용 특징 추출

```powershell
python run_eeg.py evaluate --split-json "D:/CAUEEG/abnormal.json" --annotation "D:/CAUEEG/annotation.json" --eeg-dir "D:/CAUEEG/signal/edf" --checkpoint "local/eeg-run/best.pt" --output "local/eeg-eval"
python run_eeg.py features --split-json "D:/CAUEEG/abnormal.json" --annotation "D:/CAUEEG/annotation.json" --eeg-dir "D:/CAUEEG/signal/edf" --checkpoint "local/eeg-run/best.pt" --output "local/eeg-features" --split test
```

검증·테스트는 균등 위치 8개 크롭의 softmax 확률 평균으로 환자별 예측을 만듭니다.
평가 결과에는 confusion matrix, 클래스별 sensitivity, 환자별 확률이 저장됩니다.
특징 파일은 `test_features.npz`이며 `features` [N,256], `subject_ids`, `labels`를 포함합니다.
특징은 8개 크롭의 특징 평균입니다. MRI 특징은 반드시 같은 환자 ID로 정렬해서 결합하세요.
EMMNetFeatureFusion은 EEG/MRI 각각 256차원의 concat과 1개/2개 FC를 지원합니다.
MRI 인코더 자체는 이번 EEG 작업에 포함되지 않습니다. 특징을 미리 추출해서 학습하는 것은
end-to-end fusion과 다릅니다. 공동 학습에서는 EEGEncoder를 MRI 인코더와 함께 호출하세요.

## 논문에 명시된 사항과 구현 선택

| 항목 | 적용 내용 | 근거/한계 |
|---|---|---|
| 필터/샘플링 | 0.5–70 Hz / 200 Hz | 논문 3.1절. 4차 Butterworth zero-phase 및 polyphase resample은 구현 선택 |
| crop/artifact | 학습 random crop / 명시적 artifact 제거 없음 | 논문 3.1절 |
| 시간 길이 | 10초, 2000점 | 표 2 집계 및 브랜치 기본 설정. 정확한 최종 T는 본문 미명시 |
| 백본 | 원본 ResNet1D BasicBlock [2,2,2,2] | 논문 4.2절의 CEEDNet ResNet-18 |
| 나이/출력 | age conv 채널 / 256차원 | 논문 3.2, 4.2절 |
| 채널 | 기본 EEG 19개, ECG 선택 | 논문은 기록 채널 21개를 명시하나 실제 모델 채널 선택은 불명확 |
| 활성화/FC | Mish / fc_stages=3 | CEEDNet 브랜치 기본 설정 유지; EMMNet 세부값 미명시 |
| augmentation | 정규화 신호에 Gaussian std=.001 | 그림 1 random noise 및 브랜치 awgn 설정. 나이 noise/multiplicative noise는 미사용 |
| 학습 | AdamW / focal .25,2 / batch16 | 논문 4.2절 |
| epoch/LR/warmup | 100 / 1e-4 / 5%, weight decay=.01 | 실행 가능한 기본값이며 논문의 확정 세부값이 아님 |
| focal alpha | 모든 클래스에 scalar .25 | 논문은 alpha_t의 정확한 클래스별 매핑 미명시; eeg_loss.py는 벡터도 지원 |
| 평가 | deterministic 8-crop | 브랜치의 8-crop 기본에 기반한 구현 선택; 논문의 100회 반복 평가와 다름 |

논문 표 4의 EEG-only 파라미터 25.7M과 현재 CEEDNet 기본 백본 파라미터 수는 다릅니다.
현재 기본 19채널 이진 분류기는 11,421,954개 파라미터입니다.
숫자를 맞추기 위해 근거 없이 네트워크를 키우지 않았습니다. 저자의 최종 상세 설정과
CAUEMM 데이터/가중치가 없어 **정확한 모델·성능 재현을 주장하지 않습니다**.
VALIDATION.md의 결과는 코드 동작 검증이며 진단 정확도 평가가 아닙니다.

## 출처

- 요청 브랜치: https://github.com/BCI-BrainAI/emmnet/tree/feature/eeg-preprocessing
- 가져온 커밋: https://github.com/BCI-BrainAI/emmnet/commit/2550cb46f7a0dbfd2403661217d41fe8520d591e
- 논문: 사용자 제공 emmnet.pdf, 3.1/3.2/4.2절, 그림 1, 표 2/4.
- 백본 원본: 브랜치 models/resnet_1d.py, activation.py, utils.py를 바이트 그대로
  ceednet_original/에 보관했습니다. SOURCE.json에 SHA-256을 기록합니다.
- 원본 파일의 출처 주석을 보존했습니다. 원격 GitHub 저장소에는 변경을 push하지 않았습니다.
