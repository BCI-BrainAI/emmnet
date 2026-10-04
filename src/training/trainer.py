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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.mri_dataset import MRIDataset
from evaluation.metrics import summarize, youden_threshold
from training.losses import BinaryFocalLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_loader(config: dict[str, Any], split: str, shuffle: bool, batch_size: int,
                distributed: bool = False) -> DataLoader:
    data = config["data"]
    dataset = MRIDataset(data["processed_dir"], split, normalization=data.get("normalization", "percentile_zscore"))
    workers = int(data.get("num_workers", 0))
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


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
    set_seed(int(cfg.get("seed", 42)))
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
        optimizer.zero_grad(set_to_none=True)
        n_steps = len(train_loader)
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            do_step = step % accum == 0 or step == n_steps
            ctx = ddp_model.no_sync() if (is_ddp and not do_step) else nullcontext()
            with ctx:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                    logits = ddp_model(x)
                loss = criterion(logits, y)
                (loss / accum).backward()
            running += loss.item() * len(y)
            n_seen += len(y)
            if do_step:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        stop = torch.zeros(1, device=device)
        if main:
            yv, pv, val_bce = predict(raw_model, val_loader, device, amp)
            val = summarize(yv, pv, 0.5)
            row = dict(epoch=epoch, train_loss=running / max(n_seen, 1), val_bce=val_bce, val_auc=val["auc"],
                       val_acc_at_0p5=val["accuracy"], lr=optimizer.param_groups[0]["lr"],
                       sec=round(time.time() - t0, 1))
            history.append(row)
            print(json.dumps(row), flush=True)
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

