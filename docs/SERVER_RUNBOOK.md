# MRI 단독 학습 서버 실행 절차 (integration/mri-train)

main에는 반영하지 않는다. 이 브랜치에서만 작업한다.

## 0. 환경
```bash
git fetch origin && git checkout integration/mri-train
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pyyaml nibabel pytest matplotlib   # torch는 서버 CUDA 버전에 맞는 휠
pytest -q                                                  # 합성 데이터 스모크 포함 15개 통과 확인
```

## 1. 사전학습 가중치 (확정: resnet_18.pth)
```bash
ls -l pretrained/resnet_18.pth
sha256sum pretrained/resnet_18.pth   # 로컬 값: 38b3a174...f61da3
```
`configs/mri_encoder.yaml`의 `model.pretrained_path`는 `resnet_18.pth`로 고정. (`resnet_18_23dataset.pth`는 해시가 달라 사용하지 않음.)
학습 스크립트는 backbone `missing/unexpected` 키가 하나라도 있으면 중단한다.

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
### 단일 GPU
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_mri_encoder.py --config configs/mri_encoder.yaml \
    --processed-dir <out_dir> --checkpoint-dir checkpoints/run1 2>&1 | tee checkpoints/run1.log
```

### GPU 2장 (DDP, 한 실험을 두 장으로)
```bash
torchrun --standalone --nproc_per_node=2 scripts/train_mri_encoder.py --config configs/mri_encoder.yaml \
    --processed-dir <out_dir> --checkpoint-dir checkpoints/run_ddp 2>&1 | tee checkpoints/run_ddp.log
```
- effective batch = `micro_batch_size` x GPU수 x accumulation = `batch_size`(16). 예: micro 2, GPU 2 -> accum 4. 배수가 아니면 에러.
- BN은 SyncBatchNorm(`train.sync_bn: true`)으로 GPU 간 통계 공유. GPU당 micro batch가 작아도 BN이 덜 불안정.
- val은 rank0만 평가하고 early stop 판단을 전 rank에 broadcast. 체크포인트도 rank0만 저장(`best.pth`는 단일 GPU와 동일 포맷이라 evaluate 스크립트 그대로 사용).
- 로그의 train_loss는 rank0 샤드 기준.
- 대안: GPU마다 시드/lr이 다른 실험을 각각 `CUDA_VISIBLE_DEVICES=0`, `=1`로 동시에 실행(lr sweep, 복수 시드에 효율적).
- 로컬 샌드박스처럼 호스트명 조회가 안 되는 환경에서는 `--rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:<port> --local-addr=127.0.0.1`과 `GLOO_SOCKET_IFNAME=lo`가 필요(서버는 보통 불필요).
- NCCL은 4080에서 P2P가 막혀 있으면 느릴 수 있음: 문제 시 `NCCL_P2P_DISABLE=1`.

### 평가
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_mri_encoder.py --checkpoint checkpoints/run1/best.pth \
    --processed-dir <out_dir> --split test --out-json checkpoints/run1/test_metrics.json
```
- 모델 선택: val AUC 최대 epoch (`best.pth`), 임계값은 best epoch의 val에서 Youden J로 결정해 test에 고정.
- 정규화: `data.normalization: percentile_zscore`(Med3D와 동일). `minmax`와 비교하려면 yaml만 변경.
- BN 불안정 시 `train.freeze_bn: true` 비교(DDP+sync_bn과 함께 쓰는 의미는 작음).

## 6. 하이퍼파라미터 출처
| 항목 | 값 | 출처 |
|---|---|---|
| optimizer / betas | AdamW / (0.9, 0.999) | EMMNet Sec 4.2, p.286 |
| batch size | 16 (effective) | EMMNet Sec 4.2, p.286 |
| LR schedule | warmup + cosine decay | EMMNet Sec 4.2, p.286 (warmup 길이 미기재) |
| focal loss | alpha 0.25, gamma 2 | EMMNet Sec 4.2, p.286 |
| feature dim | 256 | EMMNet Sec 4.2, p.286 |
| lr, weight decay, epochs, warmup 길이, head lr 배수, grad clip | 1e-4, 1e-2, 50, 2, x10, 1.0 | **논문에 없음 (팀 기본값)** |

참고: Med3D Sec 4.2 (p.8)는 사전학습 모델 fine-tuning에 Adam lr 0.001을 사용(분류 head를 쓰는 별개 과제). 사전학습 자체는 SGD lr 0.1, momentum 0.9, weight decay 0.001 (Sec 4.1, p.6).
권장: lr sweep {1e-3, 3e-4, 1e-4, 3e-5}를 val AUC로 비교(필요 시 test는 마지막 1회만).

## 7. 라벨 / 미해결
- 라벨 확정: CN=0, MCI=1, AD=1 (`LABELS` in mri_dataset.py와 일치).
- 진단 변경자 제외 여부, Screening 진단과 researchGroup 일치 여부는 미검증(`label_verified=False`).
- 최종 보고는 단일 시드가 아닌 복수 시드 평균/표준편차 권장.
