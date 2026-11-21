# EMMNet 재현 — Multimodal Fusion (뇌질환팀 캡스톤)

EEG-MRI Multimodal neural network (EMMNet, Lee et al., ICPR 2026)의 **멀티모달 융합(EEG + MRI) 파트** 재현.
담당 범위: 사전 학습·검증된 **1D ResNet-18 EEG Encoder**와 **3D ResNet-18(Med3D) MRI Encoder**에서 256차원 잠재 특징(Latent Feature)을 추출하고, 경량 어댑터(Multimodal Adapter) 기반의 **Mid-level Concatenation (Decoupled Late Fusion)**으로 정상·비정상을 분류한다.

- 데이터:
  - EEG: CAUEEG (`abnormal.json`, Normal=0, Abnormal=1)
  - MRI: ADNI Screening/Scaled T1 (CN=0, MCI·AD=1)
- 작업 브랜치: `feature/multimodal-adapter` (`main`에는 직접 반영하지 않음)

---

## 현재 상태 (2026-11-06)

- **MRI 파트**: 3D ResNet-18 Encoder, 전처리, 학습·평가 파이프라인 및 256차원 특징 추출(`mri_features_{split}.npz`) 완료.
- **EEG 파트**: 1D ResNet-18(Age-conditioned) Encoder 구현 및 CAUEEG 전체 코호트(1,379건) 대상 256차원 특징 추출(`eeg_features_{split}.npz`) 완료.
- **멀티모달 융합 파트**:
  - `src/models/eeg_encoder.py` 및 `src/models/fusion.py` 통합 완료(더미 텐서 Forward Pass 검증 통과).
  - Unpaired 코호트(CAUEEG + ADNI) 환경에서 Class-Conditional Pairing 기반 경량 어댑터 학습 파이프라인(`train_adapter.py`, `train_adapter_v2.py`) 구현 및 평가 완료.
  - v2 모델(Residual Adapter + $L_2$ 정규화 + MRI Oversampling + 가중치 손실) 기준 Test Sensitivity **79.55%**, F1-Score **76.92%** 달성.

---

## 시스템 구조 (Decoupled Late Fusion)

CAUEEG(EEG)와 ADNI(MRI)가 서로 다른 피험자 코호트로 구성된 Unpaired 환경에서 End-to-End 학습 시 발생하는 과적합 및 기울기 불안정을 방지하기 위해, **각 단일 모달리티 인코더의 가중치를 고정(Freeze)**한 뒤 후반부 경량 어댑터만 학습하는 구조를 채택한다.

```text
[EEG Pipeline] ──► 1D-ResNet-18 (Age-conditioned, Frozen) ──► 256-D Feature (L2 Norm) ┐
                                                                                      ├─► 512-D Concat ──► Residual Adapter ──► 정상/비정상 분류
[MRI Pipeline] ──► 3D-ResNet-18 (Med3D Backbone, Frozen)  ──► 256-D Feature (L2 Norm) ┘                    (Linear-BN-Skip)
```

---

## 설치

```bash
# torch는 실행 환경(CUDA 또는 Mac MPS/CPU)에 맞는 휠을 먼저 설치
pip install -e ".[dev]"
pytest -q
```

- 로컬 환경에서 추출된 `.npz` 특징 벡터만으로 멀티모달 어댑터를 학습할 경우 `torch`, `numpy`, `scikit-learn`만 있으면 실행 가능하다.
- `models`/`data`/`training`/`evaluation`/`utils`는 top-level 패키지이므로 전용 가상환경 사용을 권장한다.

---

## 사용 방법

### 1. 특징 벡터 파일 준비 (`data/`)

EEG와 MRI 인코더에서 각각 추출한 256차원 특징 파일을 동일한 명명 규칙과 키 구조(`features`, `labels`, `subject_ids`)로 `data/` 폴더에 배치한다.

- EEG: `data/eeg_features_{train,val,test}.npz`
  - Train `(1107, 256)` / Val `(136, 256)` / Test `(136, 256)`
- MRI: `data/mri_features_{train,val,test}.npz`
  - Train `(509, 256)` / Val `(63~64, 256)` / Test `(63~64, 256)`

#### (참고) 서버에서 EEG 특징 벡터 추출 명령어
```bash
# 1) EEG 단일 인코더 학습 (best.pt 생성)
CUDA_VISIBLE_DEVICES=0 python -u run_eeg.py train \
  --split-json /path/to/caueeg-dataset/abnormal.json \
  --annotation /path/to/caueeg-dataset/annotation.json \
  --eeg-dir /path/to/caueeg-dataset/signal/edf \
  --output local/eeg-run \
  --epochs 5 --batch-size 16 --eval-crops 2 --device cuda

# 2) 분할별 256차원 특징 추출 (--split: train / val / test)
CUDA_VISIBLE_DEVICES=0 python run_eeg.py features \
  --split-json /path/to/caueeg-dataset/abnormal.json \
  --annotation /path/to/caueeg-dataset/annotation.json \
  --eeg-dir /path/to/caueeg-dataset/signal/edf \
  --checkpoint local/eeg-run/best.pt \
  --output local/eeg-run \
  --split test --device cuda
```

#### (참고) MRI 단독 파이프라인 실행 (`docs/SERVER_RUNBOOK.md` 참조)
```bash
python scripts/emmnet.py info
python scripts/emmnet.py train base2
python scripts/emmnet.py status
```

### 2. 멀티모달 어댑터 학습 및 평가

```bash
# v1: 기본 어댑터 (512 -> 128 -> 2, Focal Loss)
python3 train_adapter.py

# v2: 개선 어댑터 (L2 Norm + MRI Oversampling + Residual Skip-Connection + Weighted CE)
python3 train_adapter_v2.py
```

학습이 완료되면 검증 성능 기준 최적 체크포인트(`adapter_best.pt` 또는 `adapter_v2_best.pt`)가 저장되며, Test 세트 기준 Accuracy, Sensitivity(Abnormal 검출률), Precision, F1-Score가 출력된다.

---

## 폴더 구조

```text
emmnet/
├── pyproject.toml, conftest.py
├── train_adapter.py                # 멀티모달 어댑터 학습 v1 (Focal Loss, 2-Layer MLP)
├── train_adapter_v2.py             # 멀티모달 어댑터 학습 v2 (L2 Norm, Oversampling, Residual Adapter)
├── test_fusion.py                  # EEG + MRI 퓨전 모델 더미 텐서 통합 테스트
├── configs/                        # mri_dataset.yaml, mri_encoder.yaml
├── data/                           # 추출된 256-D 특징 파일 (eeg_features_*.npz, mri_features_*.npz) — git 미추적
├── scripts/                        # MRI 서버 자동화 CLI 및 전처리/학습/평가 스크립트
├── src/
│   ├── data/
│   │   ├── mri_dataset.py          # ADNI 전처리 + MRIDataset (subject 누수 검사)
│   │   └── eeg_dataset.py          # CAUEEG 데이터셋
│   ├── models/
│   │   ├── mri/                    # resnet3d.py, mri_encoder.py (+256D proj, Med3D 로더), classifier.py
│   │   ├── eeg_encoder.py          # 1D ResNet-18 기반 Age-conditioned EEG Encoder (256-D)
│   │   └── fusion.py               # EMMNetFusion (Mid-level 512-D Concat + Adapter)
│   ├── training/                   # trainer.py, losses.py
│   ├── evaluation/                 # evaluate.py, metrics.py, cam.py
│   └── utils/                      # config.py, runinfo.py
├── tests/                          # 단위 테스트 (encoder / training / fusion)
├── docs/SERVER_RUNBOOK.md
└── pretrained/                     # Med3D 가중치 (resnet_18.pth) — git 미추적
```

`checkpoints/`, `data/`, `outputs/`, `pretrained/*.pth`, `*.pt` 가중치 파일은 git 미추적.

---

## 초기 실험 결과 (Unpaired Multimodal Adapter)

| 모델 버전 | 주요 설정 | Test Sensitivity (Abnormal) | Test Precision | Test F1-Score |
| --- | --- | --- | --- | --- |
| **Adapter v1** (`train_adapter.py`) | 2-Layer MLP, Focal Loss ($\alpha=0.25, \gamma=2.0$), MRI 수 기준 매칭 | 43.18% | 90.48% | - |
| **Adapter v2** (`train_adapter_v2.py`) | Residual Adapter, 모달리티별 $L_2$ Norm, Train MRI Oversampling, Weighted CE (`[1.5, 1.0]`), CosineAnnealingLR | **79.55%** | - | **76.92%** |

- **v1 한계**: Precision은 높으나 실제 비정상 환자를 정상으로 놓치는 위음성(False Negative)이 많아 민감도가 낮음.
- **v2 개선**: 모달리티 간 스케일 불균형을 $L_2$ 정규화로 완화하고, 학습 시 EEG 전체 샘플을 활용하도록 MRI Oversampling 및 클래스 가중치를 적용하여 민감도(79.55%)와 F1-Score(76.92%)를 EMMNet 논문 벤치마크(Table 4: Sensitivity 77.37%) 수준으로 끌어올림.

---

## 논문 채택 / 이탈 요약

### 채택 [EMMNet]
- **MRI**: 전처리 최소화(skull strip/bias/공간정합 생략), zero-padding 후 $256^3$ resize, 3D ResNet-18 + 256-D projection (Med3D 사전학습) — Sec 3.1~3.2, Sec 4.2
- **EEG**: 200 Hz 리샘플링, 10초 구간(2,000 타임스텝) 크로핑, 환자 나이(Age) 추가 채널 결합, 1D ResNet-18 + 256-D 출력 — Sec 3.1~3.2
- **융합 방식**: Mid-level Feature Concatenation (`256 + 256 = 512`) — Table 5, p.288
- **평가 기준**: Abnormal(MCI/AD)을 Positive로 지정, Subject 단위 분할(Data Leakage 방지) — Table 3, Sec 4.1

### 이탈 (의도적 / 환경 제약)
- **Unpaired 코호트 및 Decoupled Late Fusion**: 동일 환자의 Paired EEG-MRI 데이터(CAUEMM) 부재로 인해 CAUEEG와 ADNI를 클래스 라벨(Normal=0, Abnormal=1) 기준으로 매칭(Class-Conditional Pairing)하고, 사전 학습된 인코더를 고정한 채 경량 어댑터만 학습.
- **v2 손실 함수 및 정규화**: Focal Loss($\alpha=0.25, \gamma=2.0$) 대신 클래스 불균형과 모달리티 스케일 차이를 완화하기 위해 $L_2$ 정규화 및 가중치 CrossEntropyLoss(`weight=[1.5, 1.0]`) 적용.
- **MRI 입력 정규화**: Med3D 사전학습 환경에 맞춰 percentile 0.5~99.5 truncation 후 z-score 적용(`--set data.normalization=minmax`로 대조 가능).

---

## 미해결 / TODO

- [ ] Paired 데이터셋(또는 동일 코호트) 확보 시 End-to-End 공동 학습(Joint Training) 및 성능 비교
- [ ] Adapter v2 기준 Test Accuracy / Precision / ROC-AUC 및 다중 시드 평균±표준편차 기록 보완
- [ ] 이진 분류(Normal vs Abnormal)에서 3분류(CN / MCI / Dementia)로의 어댑터 확장
- [ ] EEG 인코더 5 에포크 이상 풀 학습 체크포인트 기반 특징 재추출 및 어댑터 재평가
- [ ] MRI 라벨 검증(`label_verified`), 시각 QC, `percentile_zscore` vs `minmax` 비교
- [ ] 원 저자 저장소 접근 권한 확보 시 아키텍처 및 가중치 재검증