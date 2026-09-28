# EMMNet 재현 (뇌질환팀 캡스톤)

EEG-MRI Multimodal neural network (EMMNet, Lee et al., ICPR 2026) 재현 작업.
윤성현 담당 파트: **MRI Encoder** (3D ResNet-18, Med3D 사전학습 전이).

## 설치

```
pip install -e ".[dev]"
```

`pyproject.toml` 기준 src-layout 설치. 설치 없이 바로 테스트만 돌리려면
`conftest.py`가 `src/`를 자동으로 path에 추가함.

## 현재 상태 (2026-09-28)

- 원 저자 저장소(`github.com/BCI-BrainAI/emmnet`) `origin`으로 연결은
  해뒀으나 `git fetch` 여전히 인증 실패, API도 404 — 실제 접근은 미해결.
- 논문 2편(`emmnet.pdf`, Med3D) 기반 자체 재현 진행 중.
- `models/mri_encoder.py`: MRI Encoder 구현 완료, 단위테스트 4/4 통과
  (`tests/test_mri_encoder.py`). 파라미터 수 33.29M ≈ 논문 Table 4
  "MRI only" 33.3M — 채널 구조 정합성 검증 완료.
- EEG encoder / fusion / dataset / trainer / evaluate: 함수·클래스
  시그니처만 정의된 TODO 스텁 (import는 되지만 호출 시 `NotImplementedError`).

## 폴더 구조

```
emmnet/
├── pyproject.toml              # 설치 가능한 패키지 정의 (src-layout)
├── conftest.py                 # pytest용 sys.path 설정 (설치 없이 테스트 시)
├── configs/mri_encoder.yaml
├── scripts/                    # CLI 진입점 (설치된 패키지를 얇게 호출)
│   ├── preprocess_mri.py
│   ├── train_mri_encoder.py
│   └── evaluate_mri_encoder.py
├── data/                       # raw/interim/processed/metadata (대용량 원본은 04_데이터셋 참조)
├── pretrained/                 # 외부 공개 가중치 (Med3D 등) — git 미추적
├── checkpoints/                # 우리가 학습한 모델 저장 위치 — git 미추적
├── docs/                       # (현재 비어있음 — 필요 시 설계 노트 추가)
├── src/
│   ├── models/
│   │   ├── resnet3d.py         # 3D ResNet BasicBlock/backbone 공통 모듈
│   │   ├── mri_encoder.py      # MRI Encoder (구현 완료)
│   │   ├── eeg_encoder.py      # TODO
│   │   └── fusion.py           # TODO (mid-level concat 이 논문 최종 채택안)
│   ├── data/
│   │   ├── mri_dataset.py      # TODO
│   │   └── eeg_dataset.py      # TODO
│   ├── training/trainer.py     # TODO
│   └── evaluation/evaluate.py  # TODO
├── tests/test_mri_encoder.py   # MRI Encoder forward pass 테스트
└── outputs/                    # 로그/그림/예측 등 결과물 — git 미추적
```

`models`/`data`/`training`/`evaluation`은 프로젝트 전용 네임스페이스
(예전 `emmnet.*`) 없이 top-level 패키지로 설치됨 — 재사용 시 이름 변경을
최소화하려는 의도. 다만 `models`/`data`처럼 흔한 이름이라 다른 설치된
패키지와 충돌할 수 있음(이 프로젝트 전용 가상환경에서 쓰는 걸 권장).

## 미해결 / TODO

- [ ] Med3D 공식 pretrained 체크포인트 입수 후 `seg_style` 플래그(stride/dilation) 검증
- [ ] EEG encoder, fusion(mid-level concat) 구현
- [ ] MRI 데이터셋 subject-level split 설계 (leakage 방지, CNN12 재현 때와 동일 원칙 적용)
- [ ] 원 저장소 접근 권한 확보 시 구조/가중치 재검증
