# MRI 단독 학습 서버 실행 절차 (integration/mri-train)

main에는 반영하지 않는다. 이 브랜치에서만 작업한다.

## 0. 환경
```bash
git fetch origin && git checkout integration/mri-train
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pyyaml nibabel pytest matplotlib   # torch는 서버 CUDA 버전에 맞는 휠
pytest -q                                                  # 합성 데이터 스모크 포함 15개 통과 확인
```

## 1. 사전학습 가중치
```bash
ls -l pretrained/            # resnet_18.pth 필요 (yaml 기본)
sha256sum pretrained/*.pth
```
- 로컬에는 `resnet_18.pth`와 `resnet_18_23dataset.pth`가 있고 **해시가 서로 다르다**(다른 가중치).
- yaml은 `resnet_18.pth`를 쓴다. 어느 쪽을 쓸지(3DSeg-8 vs 23 datasets) 결정 후 `model.pretrained_path` 고정.

## 2. 데이터 전처리 (ADNI)
`configs/mri_dataset.yaml`의 `raw_dir`, `metadata_dir`, `out_dir`를 서버 경로로 확인.
```bash
python scripts/preprocess_mri.py --config configs/mri_dataset.yaml --dry-run   # 경로/설정만 검사
python scripts/preprocess_mri.py --config configs/mri_dataset.yaml            # 실제 실행 (out_dir는 새 폴더)
du -sh <out_dir>      # 256^3 float32 = 샘플당 ~64MB
```
산출: `manifest.csv`, `selection.csv`, `excluded.csv`, `run.json`, `volumes/*.npy`, `provenance/*.json`

## 3. 산출물 점검 (학습 전 필수)
```bash
python scripts/check_mri_dataset.py --processed-dir <out_dir> --n-preview 8
```
확인:
- split별 CN/(MCI+AD) 모두 존재 (한 클래스만 있으면 AUC nan)
- subject 중복 0
- `excluded.csv` 사유 분포 (`multiple_screening_scaled_candidates`, `image_missing` 비율)
- `qc_preview.png`로 방향/크롭/배경 육안 확인 -> OK이면 qc_status 갱신
- 전경 비율 경고(>0.9)가 뜨면 배경이 0이 아니므로 정규화 전 영상 확인

## 4. GPU 메모리 사전 측정 (256^3, 미검증)
```bash
python - <<'PY'
import sys, torch; sys.path.insert(0,'src')
from models.mri.classifier import MRIClassifier
m = MRIClassifier().cuda(); opt = torch.optim.AdamW(m.parameters())
for bs in (1, 2, 4):
    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs,1,256,256,256, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16): out = m(x).sum()
        out.backward(); opt.step(); opt.zero_grad()
        print(bs, f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    except torch.cuda.OutOfMemoryError:
        print(bs, "OOM"); break
PY
```
`train.micro_batch_size`를 OOM 나지 않는 최대값으로 설정. effective batch는 `batch_size`(16) 유지(accumulation 자동).

## 5. 학습 / 평가
```bash
# 한 GPU에 한 실험. 4080 2장이면 시드/설정이 다른 두 실험을 병렬로 실행
CUDA_VISIBLE_DEVICES=0 python scripts/train_mri_encoder.py --config configs/mri_encoder.yaml \
    --processed-dir <out_dir> --checkpoint-dir checkpoints/run1 2>&1 | tee checkpoints/run1.log
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_mri_encoder.py --checkpoint checkpoints/run1/best.pth \
    --processed-dir <out_dir> --split test --out-json checkpoints/run1/test_metrics.json
```
- 모델 선택: val AUC 최대 epoch (`best.pth`), 임계값은 best epoch의 val에서 Youden J로 결정해 test에 고정.
- 정규화: `data.normalization: percentile_zscore`(Med3D와 동일). `minmax`와 비교하려면 yaml만 변경.
- BN 불안정 시 `train.freeze_bn: true` 비교.

## 6. 미해결 / 합의 필요
- 라벨: `researchGroup`(CN=0, MCI=1, AD=1) 미검증. 진단 변경자 제외 여부.
- 하이퍼파라미터 `lr`, `head_lr_mult`, `weight_decay`, epochs는 논문 값이 아닌 팀 기본값.
- 최종 보고는 단일 시드가 아닌 복수 시드 평균/표준편차 권장.
