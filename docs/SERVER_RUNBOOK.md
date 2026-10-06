# MRI 단독 학습 서버 실행 절차 (integration/mri-train)

main에는 반영하지 않는다. 이 브랜치에서만 작업한다. 모든 명령은 repo 루트에서 실행한다.
직접 코드를 타이핑하는 단계는 없다(스크립트/`--set`으로 대체).

## 0. 환경
```bash
git fetch origin && git checkout integration/mri-train
# 서버 로컬 YAML 수정이 남아 있으면: git stash (micro_batch_size=4는 이제 커밋된 YAML 기본값)
python -m venv .venv && source .venv/bin/activate        # 서버 Python 3.14면 3.11/3.12 환경 권장
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 드라이버 CUDA 12.4 -> cu126 휠 (실제 설치 방법은 아래 기록란에 남길 것)
pip install -e ".[dev]" matplotlib      # pyproject.toml 기준. matplotlib 포함
pytest -q                               # 로컬 기준 24개(신규 tooling 7개 포함) 통과 확인
export EMMNET_DATA=<전처리 out_dir>      # 이후 모든 스크립트가 데이터 경로로 사용(--processed-dir 생략 가능)
```
설치 기록(서버에서 실제로 쓴 torch 설치 명령/버전): `__________`

## 1. 사전학습 가중치 (확정: resnet_18.pth)
```bash
sha256sum pretrained/resnet_18.pth   # 로컬 값: 38b3a174...f61da3
```
`model.pretrained_path`는 `resnet_18.pth`로 고정. (`resnet_18_23dataset.pth`는 해시가 달라 사용하지 않음.)
학습 스크립트는 backbone `missing/unexpected` 키가 하나라도 있으면 중단한다.

## 2. 데이터 전처리 (ADNI)
`configs/mri_dataset.yaml`의 `raw_dir`, `metadata_dir`, `out_dir`를 서버 경로로 확인.
```bash
python scripts/preprocess_mri.py --config configs/mri_dataset.yaml --dry-run
python scripts/preprocess_mri.py --config configs/mri_dataset.yaml            # out_dir는 새 폴더 (이미 파일이 있으면 중단됨; run.json으로 확인)
```
산출: `manifest.csv`, `selection.csv`, `excluded.csv`, `run.json`, `volumes/*.npy`, `provenance/*.json`

## 3. 산출물 점검 (학습 전 필수)
```bash
python scripts/check_mri_dataset.py --processed-dir $EMMNET_DATA --n-preview 8
python scripts/cache_norm_stats.py  --processed-dir $EMMNET_DATA     # 정규화 통계 캐시 + 전경비율(fg_frac) 경고. 학습 속도 개선(결과 동일)
```
확인:
- split별 CN/(MCI+AD) 모두 존재, subject 중복 0
- `excluded.csv` 사유 분포(`multiple_screening_scaled_candidates` 비율 = 표본 손실/선택 편향)
- `qc_preview.png` 육안 확인 후 qc_status 갱신
- `cache_norm_stats.py`의 fg_frac 경고(>0.9): 전경 마스크 `volume>0`이 깨졌다는 신호 -> z-score 통계 오염

## 4. GPU 메모리 사전 측정
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_memory.py --sizes 1 2 4 8
```
권장 `micro_batch_size`를 `--set train.micro_batch_size=<값>`으로 적용. 측정 기록(256^3, bf16): bs 1/2/4/8 = 1.8/3.6/6.6/12.8 GiB (bs8은 OOM 직전이라 미사용).

## 5. 학습 / 평가
### 5-0. 점검 학습 (본 학습 전)
```bash
# subset 스모크
python scripts/make_subset.py --processed-dir $EMMNET_DATA --out-dir <subset_dir> --per-split 18 4 4
# overfit 확인: train_auc -> ~1.0 이 되어야 파이프라인/라벨/정규화가 정상
bash scripts/run_experiment.sh overfit --gpu 0 -- --set data.overfit_n=16 train.lr=1e-3 train.epochs=40 train.early_stopping_patience=999
python scripts/analyze_results.py --run-dir checkpoints/overfit
```

### 5-1. 단일 GPU 본 학습 (권장)
```bash
bash scripts/run_experiment.sh base --gpu 0 --bg                  # 백그라운드(nohup), 로그: checkpoints/base/train.log
bash scripts/run_experiment.sh lr3e-4 --gpu 0 --bg -- --set train.lr=3e-4 train.seed=1
```
- 자동 저장: `config.yaml`(해석된 설정), `meta.json`(git sha/dirty/환경/label_verified), `history.json`(매 epoch),
  `best.pth`(val AUC 최대, val 기반 Youden threshold 포함), `last.pth`, `train_metrics.json`/`val_metrics.json`, `predictions_{train,val}.csv`
- test는 자동 실행하지 않는다. 같은 이름의 run 폴더가 있으면 거부(덮어쓰기 방지). 폴더 위치는 `CKPT_ROOT`로 변경 가능.
- `--set`은 YAML에 있는 키만 허용(오타 방지). 예: `--set train.lr=3e-4 train.weight_decay=0.05`

### 5-2. DDP (사용하지 않기로 결정했으나 구현/CPU 테스트됨)
```bash
torchrun --standalone --nproc_per_node=2 scripts/train_mri_encoder.py --checkpoint-dir checkpoints/run_ddp
```
effective batch = `micro_batch_size` x GPU수 x accumulation = `batch_size`(16). 배수가 아니면 에러. val은 rank0만 평가.

### 5-3. 분석 (test 전)
```bash
python scripts/analyze_results.py --run-dir checkpoints/base
```
출력: `analysis.json`, `curves.png`, 콘솔 요약
- history 진단(best==last, 미학습, 과적합, val 변동 큼), NaN 검사
- split별 AUC + 부트스트랩 95% CI, sens/spec(체크포인트 임계값), 클래스별 확률 분위수
- MCI/AD 등 research_group별 mean prob / positive rate, age-only AUC, Spearman(prob, age)
- 주의: `train_auc`는 학습 중 BN train-mode 확률로 계산한 값(추가 forward 없음). eval-mode train AUC는 `predictions_train.csv` 기준(`analysis.json`의 train).
  val 지표 + val 기반 임계값 조합은 낙관적이다.

### 5-4. test 평가 (run당 1회)
```bash
python scripts/evaluate_mri_encoder.py --checkpoint checkpoints/base/best.pth --splits test
python scripts/analyze_results.py --run-dir checkpoints/base        # test CI 포함 재생성
```
- 평가는 체크포인트에 저장된 **학습 당시 config**(정규화/모델 크기)를 사용한다. YAML을 나중에 바꿔도 영향 없음.
- `.test_used.json` 잠금: 같은 run에서 test 재실행은 거부. 의도한 재평가만 `--force`(기록은 남음).
- `last.pth`(threshold 없음)는 평가 거부.

## 6. 하이퍼파라미터 출처
| 항목 | 값 | 출처 |
|---|---|---|
| optimizer / betas | AdamW / (0.9, 0.999) | EMMNet Sec 4.2, p.286 |
| batch size | 16 (effective) | EMMNet Sec 4.2, p.286 |
| LR schedule | warmup + cosine decay | EMMNet Sec 4.2, p.286 (warmup 길이 미기재) |
| focal loss | alpha 0.25, gamma 2 | EMMNet Sec 4.2, p.286 |
| feature dim | 256 | EMMNet Sec 4.2, p.286 |
| lr, weight decay, epochs, warmup 길이, head lr 배수, grad clip | 1e-4, 1e-2, 50, 2, x10, 1.0 | **논문에 없음 (팀 기본값)** |

참고: Med3D Sec 4.2 (p.8) fine-tuning Adam lr 0.001(별개 과제). Med3D 사전학습은 SGD lr 0.1, momentum 0.9, wd 0.001 (Sec 4.1, p.6).
lr sweep {1e-3, 3e-4, 1e-4, 3e-5}는 val AUC로만 선택(test는 최종 1회):
```bash
for lr in 1e-3 3e-4 1e-4 3e-5; do bash scripts/run_experiment.sh lr$lr --gpu 0 -- --set train.lr=$lr; done
```
복수 시드: `-- --set train.seed=1` (평균±표준편차 보고).

## 7. 라벨 / 미해결
- 라벨 확정: CN=0, MCI=1, AD=1. 진단 변경자 제외 여부, Screening 진단과 researchGroup 일치는 미검증(`label_verified=False`; `meta.json`과 학습/분석 로그에 경고).
- 모델 선택(val AUC)과 임계값(Youden)이 같은 val에서 정해지므로 val이 작으면 노이즈에 민감 -> `split_counts`와 CI를 함께 볼 것.
- 최종 보고는 단일 시드가 아닌 복수 시드 평균/표준편차 권장.
