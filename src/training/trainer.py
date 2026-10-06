"""MRI 단독 학습 루프 (AMP, gradient accumulation, 선택적 multi-GPU DDP).

- 단일 GPU: `python scripts/train_mri_encoder.py ...`
- 다중 GPU : `torchrun --nproc_per_node=2 scripts/train_mri_encoder.py ...` (DDP + SyncBN)
- 모델 선택: val AUC 최대 epoch의 체크포인트(best.pth). test는 학습 중 사용하지 않는다.
- 임계값: best epoch의 val 예측에서 Youden J로 결정해 체크포인트에 저장한다.
- LR 스케줄: linear warmup 후 cosine decay (EMMNet Sec 4.2, p.286).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.mri_dataset import MRIDataset
from evaluation.metrics import summarize, youden_threshold
from training.losses import BinaryFocalLoss, focal_from_probs
from utils.runinfo import run_meta


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:  # 속도 저하 가능. 3D conv/pool 일부 연산은 완전 결정적이지 않을 수 있다.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    """DataLoader worker별 numpy/random 시드 고정(torch 시드는 base_seed로 자동 설정됨)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def setup_distributed(device_name: str) -> tuple[bool, int, int, torch.device]:
    """torchrun 환경변수(WORLD_SIZE>1)가 있으면 process group 초기화. (is_ddp, rank, world, device)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return False, 0, 1, resolve_device(device_name)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and device_name in ("auto", "cuda")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=60))
    return True, rank, world, torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")


def overfit_indices(dataset: MRIDataset, n: int) -> list[int]:
    """클래스 균형(앞에서부터)으로 n개 인덱스. overfit 점검 전용."""
    by_class = {0: [], 1: []}
    for i, r in enumerate(dataset.records):
        by_class[r["label"]].append(i)
    half = max(1, n // 2)
    return sorted(by_class[0][:half] + by_class[1][:n - half])


def dataset_records(loader: DataLoader) -> list[dict[str, Any]]:
    """loader가 (Subset일 수 있는) MRIDataset에서 내보내는 순서 그대로의 manifest 행."""
    ds = loader.dataset
    if isinstance(ds, Subset):
        return [ds.dataset.records[i] for i in ds.indices]
    return ds.records


def make_loader(config: dict[str, Any], split: str, shuffle: bool, batch_size: int,
                distributed: bool = False) -> DataLoader:
    data = config["data"]
    dataset = MRIDataset(data["processed_dir"], split, normalization=data.get("normalization", "percentile_zscore"))
    if split == "train" and data.get("overfit_n"):
        dataset = Subset(dataset, overfit_indices(dataset, int(data["overfit_n"])))
    workers = int(data.get("num_workers", 0))
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    gen = torch.Generator().manual_seed(int(config.get("train", {}).get("seed", 42)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0, generator=gen, worker_init_fn=_worker_init)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """(labels, probs, mean_bce). BN은 eval 모드."""
    model.eval()
    ys, ps, losses = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float().cpu()
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float(), reduction="sum").item())
        ps.append(torch.sigmoid(logits).numpy())
        ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return y, p, float(sum(losses) / len(y))


def lr_factor(epoch: int, epochs: int, warmup: int) -> float:
    """epoch(0-index)에 대한 LR 배율: linear warmup -> cosine decay."""
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / (warmup + 1)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(model, config: dict[str, Any], checkpoint_dir: str | Path) -> dict[str, Any]:
    cfg = config["train"]
    is_ddp, rank, world, device = setup_distributed(cfg.get("device", "auto"))
    main = rank == 0
    set_seed(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", False)))
    amp = bool(cfg.get("amp", True))
    model.to(device)
    if is_ddp and cfg.get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    raw_model = model
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if is_ddp else model

    eff_bs = int(cfg["batch_size"])
    micro = int(cfg.get("micro_batch_size", eff_bs))
    if eff_bs % (micro * world):
        raise ValueError(f"batch_size({eff_bs})는 micro_batch_size({micro}) x world({world})의 배수여야 한다")
    accum = eff_bs // (micro * world)
    if main:
        print(f"[INFO] world={world} device={device} micro={micro} accum={accum} effective_batch={eff_bs}", flush=True)

    train_loader = make_loader(config, "train", True, micro, distributed=is_ddp)
    val_loader = make_loader(config, "val", False, micro) if main else None  # val은 rank0 단독 평가

    base_lr = float(cfg.get("lr", 1e-4))
    head_mult = float(cfg.get("head_lr_mult", 10.0))
    new_params = list(raw_model.encoder.projection.parameters()) + list(raw_model.head.parameters())
    new_ids = {id(p) for p in new_params}
    backbone_params = [p for p in raw_model.parameters() if id(p) not in new_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": base_lr, "base": base_lr},
         {"params": new_params, "lr": base_lr * head_mult, "base": base_lr * head_mult}],
        betas=tuple(cfg.get("betas", [0.9, 0.999])), weight_decay=float(cfg.get("weight_decay", 1e-2)))
    epochs, warmup = int(cfg.get("epochs", 50)), int(cfg.get("warmup_epochs", 2))
    criterion = BinaryFocalLoss(alpha=float(cfg.get("focal_alpha", 0.25)), gamma=float(cfg.get("focal_gamma", 2)))

    out_dir = Path(checkpoint_dir)
    if main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        verified = bool(train_loader.dataset.dataset.labels_verified if isinstance(train_loader.dataset, Subset)
                        else train_loader.dataset.labels_verified)
        (out_dir / "meta.json").write_text(json.dumps(run_meta({
            "label_verified": verified, "world": world, "micro_batch_size": micro, "accum": accum,
            "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset)}), indent=2), encoding="utf-8")
        if not verified:
            print("[WARN] label_verified=False: 라벨(CN/MCI/AD)이 진단 시점 기준으로 검증되지 않았다.", flush=True)
    patience = int(cfg.get("early_stopping_patience", 10))
    best_auc, best_epoch, history = -1.0, -1, []
    epoch = 0

    for epoch in range(1, epochs + 1):
        for g in optimizer.param_groups:
            g["lr"] = g["base"] * lr_factor(epoch - 1, epochs, warmup)
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        if cfg.get("freeze_bn", False):
            for m in ddp_model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        t0, running, n_seen = time.time(), 0.0, 0
        tr_y, tr_p = [], []  # train-mode(BN train) 확률 -> train_auc (추가 forward 없음; eval-mode 값과는 다를 수 있음)
        optimizer.zero_grad(set_to_none=True)
        n_steps = len(train_loader)
        full = n_steps // accum
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            do_step = step % accum == 0 or step == n_steps
            ctx = ddp_model.no_sync() if (is_ddp and not do_step) else nullcontext()
            with ctx:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                    logits = ddp_model(x)
                loss = criterion(logits, y)
                group = accum if step <= full * accum else n_steps - full * accum  # 마지막 불완전 그룹은 실제 크기로 나눔
                (loss / group).backward()
            running += loss.item() * len(y)
            n_seen += len(y)
            tr_y.append(y.detach().cpu().numpy())
            tr_p.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
            if do_step:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        stop = torch.zeros(1, device=device)
        if main:
            yv, pv, val_bce = predict(raw_model, val_loader, device, amp)
            val = summarize(yv, pv, 0.5)
            ty, tp = np.concatenate(tr_y), np.concatenate(tr_p)
            row = dict(epoch=epoch, train_loss=running / max(n_seen, 1),  # focal (train_loss와 val_focal은 같은 척도)
                       train_auc=summarize(ty, tp, 0.5)["auc"], train_acc_at_0p5=summarize(ty, tp, 0.5)["accuracy"],
                       val_focal=focal_from_probs(yv, pv, criterion.alpha, criterion.gamma),
                       val_bce=val_bce, val_auc=val["auc"], val_acc_at_0p5=val["accuracy"],
                       lr=optimizer.param_groups[0]["lr"], sec=round(time.time() - t0, 1))
            history.append(row)
            print(json.dumps(row), flush=True)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")  # 매 epoch 갱신
            auc = val["auc"] if np.isfinite(val["auc"]) else -1.0
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch
                torch.save({"state_dict": raw_model.state_dict(), "epoch": epoch, "val_auc": auc,
                            "threshold": youden_threshold(yv, pv), "config": config}, out_dir / "best.pth")
            elif epoch - best_epoch >= patience:
                print(f"[INFO] early stop at epoch {epoch} (best {best_epoch}, val_auc={best_auc:.4f})", flush=True)
                stop.fill_(1)
        if is_ddp:
            dist.broadcast(stop, src=0)  # rank0의 early stop 판단을 전 rank에 전달 (동시에 barrier 역할)
        if stop.item():
            break

    if main:
        torch.save({"state_dict": raw_model.state_dict(), "epoch": epoch, "config": config}, out_dir / "last.pth")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    return {"best_epoch": best_epoch, "best_val_auc": best_auc, "history": history}

