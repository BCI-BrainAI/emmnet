# EMMNet 재현 — MRI Encoder (뇌질환팀 캡스톤)

EEG-MRI Multimodal neural network (EMMNet, Lee et al., ICPR 2026)의 **MRI 파트** 재현.
담당 범위: 3D ResNet-18(Med3D 사전학습) MRI Encoder를 완성하고 **MRI 단독 성능**을 측정한다(fusion 이전 단계).
데이터: ADNI Screening/Scaled T1 (CN=0, MCI·AD=1). 작업 브랜치: `integration/mri-train` (main에는 직접 반영하지 않음).

## 현재 상태 (2026-10-06)

- 구현 완료: MRI Encoder, 데이터셋/전처리, 학습·평가·분석 파이프라인, CAM 계산 함수. pytest 28개 통과(CPU, 합성 데이터).
- 베이스라인 학습(단일 GPU) 종료, 결과 분석 전. 실데이터·GPU 검증은 서버에서 진행.
- 미구현(스텁, `NotImplementedError`): `models/eeg_encoder.py`, `models/fusion.py`, `data/eeg_dataset.py` — EEG 담당자 배정 후 작성.
- 원 저자 저장소(`github.com/BCI-BrainAI/emmnet`)는 인증 문제로 접근 불가 — 논문(`emmnet.pdf`)과 Med3D 논문 기반 자체 재현.
- 라벨은 `researchGroup`을 그대로 쓰며 **진단 시점 검증 전**(`label_verified=False`; 학습 로그/`meta.json`에 경고).

## 설치

```bash
# torch는 서버 CUDA/Python에 맞는 휠을 먼저 설치 (서버 드라이버 CUDA 12.4: cu126 휠 또는 Python 3.11/3.12)
pip install -e ".[dev]"
pytest -q
```
설치 없이 테스트만 돌리면 `conftest.py`가 `src/`를 path에 추가한다. `models`/`data`/`training`/`evaluation`/`utils`는 top-level 패키지라 전용 가상환경을 권장.

## 사용 (전체 절차는 `docs/SERVER_RUNBOOK.md`)

```bash
python scripts/emmnet.py init --data <전처리 out_dir> --gpus 0 1   # 1회
python scripts/emmnet.py setup --install-torch                     # 환경 설치/검증
python scripts/emmnet.py check                                     # 데이터 점검
python scripts/emmnet.py train base2                               # 학습(백그라운드) + train/val 평가
python scripts/emmnet.py status                                    # 진행/요약
python scripts/emmnet.py sweep                                     # lr sweep (GPU 자동 분배)
python scripts/emmnet.py final --lr <선택값>                        # 시드 3개 최종 학습
python scripts/emmnet.py test final_s0 final_s1 final_s2           # test 1회 + 평균±표준편차
```
개별 스크립트(`run_experiment.sh`, `evaluate_mri_encoder.py`, `analyze_results.py` 등)와 `--set key=value` override는 런북 참조.

## 폴더 구조

```
emmnet/
├── pyproject.toml, conftest.py
├── configs/            mri_dataset.yaml (전처리), mri_encoder.yaml (학습)
├── scripts/
│   ├── emmnet.py               서버 작업 자동화 CLI(init/setup/check/train/sweep/final/test/status)
│   ├── preprocess_mri.py       ADNI NIfTI -> 256^3 .npy + manifest
│   ├── check_mri_dataset.py    분할/누수/제외사유/QC 미리보기
│   ├── cache_norm_stats.py     percentile_zscore 통계 캐시, fg_frac 경고
│   ├── make_subset.py          스모크용 subset(파일 복사)
│   ├── profile_memory.py       GPU 메모리 측정 -> micro_batch_size 결정
│   ├── train_mri_encoder.py    학습 CLI
│   ├── run_experiment.sh       학습 + train/val 평가 래퍼(로그/설정/git sha 저장)
│   ├── evaluate_mri_encoder.py 평가 CLI (학습 당시 config, val 기반 임계값, test 1회 잠금)
│   └── analyze_results.py      결과 분석(analysis.json, curves.png)
├── src/
│   ├── data/mri_dataset.py     전처리 + MRIDataset(subject 누수 검사, Med3D 정규화)
│   ├── models/mri/             resnet3d.py (backbone), mri_encoder.py (+projection, Med3D 로더), classifier.py (linear probe)
│   ├── training/               trainer.py (단일 GPU, AMP, grad accum), losses.py (binary focal)
│   ├── evaluation/             evaluate.py, metrics.py, cam.py
│   ├── utils/                  config.py (--set/EMMNET_DATA), runinfo.py (git 메타, test 잠금)
│   └── models/{eeg_encoder,fusion}.py, data/eeg_dataset.py   스텁
├── tests/              encoder / training pipeline / tooling / CAM
├── docs/SERVER_RUNBOOK.md
└── pretrained/         Med3D 가중치(git 추적). 사용: resnet_18.pth (resnet_18_23dataset.pth는 해시가 달라 미사용, 보관)
```
`checkpoints/`, `data/processed/`, `outputs/`는 git 미추적.

## 논문 채택/이탈 요약

채택 [EMMNet]:
- 전처리 최소화(skull strip/bias/공간정합 생략) 및 zero-padding 후 256³ resize, min-max [0,1] 저장 — Sec 3.1, p.283
- 3D ResNet-18 + projection(n=256), Med3D 사전학습 — Sec 3.2 p.284, Sec 4.2 p.286
- AdamW(0.9, 0.999), cosine decay + warmup, batch 16(effective), focal α=0.25 γ=2 — Sec 4.2 p.286, Eq.(1) p.285
- abnormal(MCI/AD)을 positive로 지표 계산, subject 단위 분할 — Table 3 p.287, Sec 4.1 p.285

이탈(의도적):
- **학습 입력 정규화**: percentile 0.5~99.5 truncation 후 z-score(Med3D 사전학습과 동일, 전경 `volume>0` 기준·배경 0). 논문은 min-max. `--set data.normalization=minmax`로 대조 실험.
- 학습 샘플 4,000개 [Sec 4.2]와 balanced test set [Sec 4.1]은 따르지 않음(ADNI 규모/층화 분할 사용).
- 단일 GPU 학습(논문 dual RTX 4090).

논문에 없는 팀 기본값: lr 1e-4, weight decay 1e-2, epochs 50, warmup 2, head lr ×10, grad clip 1.0, early stopping 10, bf16 AMP, 모델 선택=val AUC 최대, 임계값=val Youden J (출처 표는 런북 6절).

참고 수치: MRI-only 33.3M params / Acc 34.86% [Table 4, p.287]는 joint 학습에서 분리한 값이라 목표로 삼지 않음. 구현 파라미터는 encoder(proj 256) 33.12M으로 0.18M 차이가 있으며 원인은 확인되지 않음(논문의 projection 상세는 미기재).

## 미해결 / TODO

- [ ] 베이스라인 결과 분석(곡선 진단, test AUC CI, age-only AUC 등) → val AUC가 0.5를 의미 있게 넘으면 lr sweep, 이후 시드 3개 최종 평가
- [ ] 라벨 검증(진단 변경자, Screening 진단과 researchGroup 일치), 시각 QC, 전경비율(`fg_frac`) 확인
- [ ] percentile_zscore vs minmax 비교
- [ ] 서버 torch 설치 방법 런북에 기록, 서버 pytest 재확인
- [ ] EEG encoder, fusion(mid-level concat이 논문 채택안, Table 5 p.288) — 담당자 배정 후
- [ ] 원 저장소 접근 권한 확보 시 구조/가중치 재검증
