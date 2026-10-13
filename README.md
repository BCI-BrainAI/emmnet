# EMMNet EEG 파트

EEG 신호와 환자 나이를 입력받아 **256차원 특징을 추출하고 정상·비정상을 분류하는 코드**입니다.
`feature/eeg-preprocessing` 브랜치를 바탕으로 EMMNet 논문의 EEG 전처리, 모델, 학습, 평가를 연결했습니다.

## 어디까지 구현했나요?

| 부분 | 현재 상태 |
|---|---|
| 데이터 읽기 | CAUEEG EDF 파일, 환자 나이, 정답 라벨 연결 |
| EEG 전처리 | 채널 정렬, 필터링, 200 Hz 리샘플링, 정규화, 10초 구간 선택 |
| EEG 인코더 | ResNet-18 기반 모델, 나이 추가 채널, 256차원 특징 출력 |
| EEG 단독 분류 | 정상·비정상 분류 및 학습·평가. 3분류도 선택 가능 |
| MRI 연결 준비 | EEG 특징 저장, 외부 MRI 특징과 연결하는 모듈 |
| MRI 인코더·전체 공동 학습 | 아직 구현하지 않음 |

```text
EEG 기록 + 환자 나이
    ↓
전처리 및 10초 구간 선택
    ↓
나이를 추가 채널로 입력하는 1D ResNet-18
    ↓
256차원 EEG 특징
    ├─ 분류 헤드 → 정상·비정상 예측
    └─ 특징 저장 → 이후 같은 환자의 MRI 특징과 연결
```

전체 EMMNet의 완성본이나 논문과 정확히 동일한 재현 모델은 아닙니다.
공개된 핵심 설정을 반영했지만 공개되지 않은 세부값은 원본 브랜치 설정 또는 구현 선택을 사용했습니다.
현재 EEG 이진 분류기는 약 11.42M 파라미터로, 논문 표의 EEG-only 25.7M과 차이가 있습니다.

## 실제 데이터로 확인한 결과

CAUEEG에서 학습 64명, 검증 32명, 테스트 32명을 선택해 5 epoch 실행했습니다.
각 분할의 정상·비정상 인원은 같고, 환자는 분할 사이에 겹치지 않도록 선택했습니다.

| 항목 | 결과 |
|---|---|
| 학습 loss | 약 0.0687 → 0.0362 |
| 최고 검증 정확도 | 62.5% (4번째 epoch) |
| 테스트 정확도 | 65.625% (32명 중 21명 정답) |
| 클래스 0 recall | 37.5% |
| 클래스 1 recall | 93.75% |

사용자가 제공한 서버 실행 화면을 기준으로 기록했습니다.
실제 EDF 전처리, 학습, 모델 저장·복원, 평가가 실행된 소규모 동작 검사입니다.
클래스 1 쪽으로 예측이 치우쳤으며, 이 결과로 논문 성능 재현을 판단할 수는 없습니다.
실제 데이터의 256차원 특징 저장 결과는 아직 확인하지 않았습니다.
로컬 자동 검사 8개와 합성 데이터의 전체 실행 검사도 통과했습니다.
자세한 기록은 [VALIDATION.md](VALIDATION.md)에 있습니다.

## 어떤 파일을 실행하나요?

**`run_eeg.py`를 실행하면 됩니다.**

| 파일 | 역할 |
|---|---|
| `run_eeg.py` | 학습, 평가, 특징 저장 명령 |
| `eeg_data.py` | 데이터 읽기, 환자 중복 확인, EEG 전처리 |
| `emmnet_eeg.py` | EEG 인코더, 분류기, 외부 MRI 특징 연결 모듈 |
| `eeg_loss.py` | focal loss 계산 |
| `ceednet_original/` | 수정하지 않은 원본 ResNet1D 백본 |
| `tests/test_eeg.py` | 전처리·모델·손실·체크포인트 자동 검사 |

기존 `run_train.py`, `notebook/`, 루트의 `test_*.py`는 이전 브랜치의 코드입니다.
이번 구현에는 `run_eeg.py`를 사용하세요. 이전 설명은 [README.upstream.md](README.upstream.md)에 있습니다.

## 설치

`run_eeg.py`가 있는 폴더에서 사용할 Python 환경을 활성화한 뒤 실행합니다.
서버에서는 Python 3.9.25, PyTorch 2.5.1+cu121, MNE 1.8.0으로 학습을 확인했습니다.

```bash
python -m pip install -r requirements-eeg.txt
python -m pytest
```

GPU용 PyTorch가 이미 설치돼 있다면 해당 환경을 사용하세요.
필요한 패키지 버전은 Python·CUDA 환경에 따라 달라질 수 있습니다.

## 데이터 준비

데이터는 저장소에 포함하지 않습니다. 다음 구조의 데이터 폴더가 필요합니다.

```text
caueeg-dataset/
├── abnormal.json       # 학습·검증·테스트 환자와 라벨
├── annotation.json     # 환자 나이
└── signal/edf/
    ├── 00001.edf
    └── ...
```

분할 JSON에는 `train_split`, `validation_split`, `test_split`이 있어야 합니다.
각 항목의 `serial`로 EDF를 연결하고 `class_label`을 정답으로 사용합니다.
나이는 분할 JSON의 `age` 또는 `annotation.json`의 `data` 항목에서 가져옵니다.
환자 중복, 누락된 나이·파일·채널, 10초 미만 신호는 오류로 보고합니다.
3분류는 3분류용 JSON과 `--num-classes 3`을 함께 지정하세요.

## 학습 실행

아래는 **Linux 서버에서 GPU 1번을 사용하는 예시**입니다. 경로를 자신의 데이터 위치로 바꾸세요.
전체 `abnormal.json`을 사용합니다. 짧은 검사에서는 환자를 선택한 JSON을 `--split-json`에 지정하세요.

```bash
CUDA_VISIBLE_DEVICES=1 python -u run_eeg.py train \
  --split-json /path/to/caueeg-dataset/abnormal.json \
  --annotation /path/to/caueeg-dataset/annotation.json \
  --eeg-dir /path/to/caueeg-dataset/signal/edf \
  --output local/eeg-run \
  --epochs 5 \
  --batch-size 16 \
  --eval-crops 2 \
  --device cuda
```

CPU에서는 `CUDA_VISIBLE_DEVICES=1`을 빼고 `--device cpu`로 바꿉니다.
Windows PowerShell에서는 `$env:CUDA_VISIBLE_DEVICES='1'`로 설정하고 명령을 한 줄로 입력하세요.
위 예시의 역슬래시는 Linux용 줄 연결 문자입니다.

| 옵션 | 의미 |
|---|---|
| `--epochs 5` | 학습 데이터를 5번 반복 |
| `--batch-size 16` | 한 번에 환자 16명 처리 |
| `--eval-crops 2` | 평가 시 환자별 2개 구간의 예측 확률을 평균 |
| `--output` | 결과 저장 폴더 |
| `--include-ecg` | 기본 EEG 19채널에 ECG 추가 |

처음에는 학습 환자의 신호·나이 정규화 통계를 계산하므로 출력 없이 시간이 걸릴 수 있습니다.
검증·테스트에는 같은 통계를 사용합니다. 매번 EDF 전체를 읽고 전처리하므로 환자가 많으면 느립니다.
학습의 마지막 배치가 batch 크기보다 작으면 그 배치를 제외합니다. 학습 환자는 최소 batch 크기 이상이어야 합니다.

## 결과 확인

학습이 끝나면 `--output` 폴더에 다음 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `best.pt` | 검증 정확도가 가장 높은 모델, 정규화 통계, 설정 |
| `history.json` | epoch별 학습 loss와 검증 정확도 |
| `test_metrics.json` | 최선 모델의 테스트 정확도, 클래스별 recall, 혼동행렬, 환자별 예측 |

혼동행렬은 **행이 실제 정답, 열이 예측**입니다.
0=정상, 1=비정상이면 클래스 1 recall은 비정상 민감도, 클래스 0 recall은 정상 특이도입니다.
라벨 의미는 사용한 JSON에서 확인하세요.

## 저장된 모델로 평가 또는 특징 추출

학습에 사용한 분할 JSON과 체크포인트를 지정합니다.

```bash
python run_eeg.py evaluate \
  --split-json /path/to/split.json \
  --annotation /path/to/caueeg-dataset/annotation.json \
  --eeg-dir /path/to/caueeg-dataset/signal/edf \
  --checkpoint local/eeg-run/best.pt \
  --output local/eeg-eval \
  --device cpu
```

위 명령의 `evaluate`를 `features`로 바꾸면 테스트 환자의 특징을 저장합니다.
`test_features.npz`에는 `features` [환자 수, 256], `subject_ids`, `labels`가 들어갑니다.
평가 구간 수는 체크포인트의 설정을 사용하며 특징은 구간별 특징의 평균입니다.
MRI와 연결할 때는 반드시 `subject_ids`로 같은 환자를 맞춰야 합니다.

## 남은 작업과 자세한 설정

- MRI 인코더 연결과 EEG–MRI 공동 학습이 남아 있습니다.
- 현재 실행 코드는 단일 GPU/CPU용이며 학습 재개·여러 GPU 분산 학습은 지원하지 않습니다.
- 특징을 미리 저장하는 방식과 EEG·MRI 인코더를 함께 학습하는 방식은 다릅니다.
- 필터 종류, 활성화 함수, 논문과 설정의 차이는 [EEG_DETAILS.md](EEG_DETAILS.md)에 정리했습니다.

출발 브랜치: [feature/eeg-preprocessing](https://github.com/BCI-BrainAI/emmnet/tree/feature/eeg-preprocessing).
원본 커밋과 백본 파일 해시는 [SOURCE.json](SOURCE.json)에 있습니다.
