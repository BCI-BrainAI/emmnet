"""MRI 단독 학습 루프 (단일 GPU, AMP, gradient accumulation).

- 모델 선택: val AUC 최대 epoch의 체크포인트(best.pth) 저장. test는 학습 중 사용하지 않는다.
- 임계값: best epoch의 val 예측에서 Youden J로 결정해 체크포인트에 저장한다.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def make_loader(config: dict[str, Any], split: str, shuffle: bool, batch_size: int) -> DataLoader:
    data = config["data"]
    dataset = MRIDataset(data["processed_dir"], split, normalization=data.get("normalization", "percentile_zscore"))
    workers = int(data.get("num_workers", 0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """(labels, probs, mean_loss_unweighted_bce). BN은 eval 모드."""
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


def train(model, config: dict[str, Any], checkpoint_dir: str | Path) -> dict[str, Any]:
    cfg = config["train"]
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    amp = bool(cfg.get("amp", True))
    model.to(device)

    eff_bs = int(cfg["batch_size"])
    micro = int(cfg.get("micro_batch_size", eff_bs))
    if eff_bs % micro:
        raise ValueError("batch_size must be a multiple of micro_batch_size")
    accum = eff_bs // micro

    train_loader = make_loader(config, "train", True, micro)
    val_loader = make_loader(config, "val", False, micro)

    base_lr = float(cfg.get("lr", 1e-4))
    head_mult = float(cfg.get("head_lr_mult", 10.0))
    new_params = list(model.encoder.projection.parameters()) + list(model.head.parameters())
    new_ids = {id(p) for p in new_params}
    backbone_params = [p for p in model.parameters() if id(p) not in new_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": base_lr}, {"params": new_params, "lr": base_lr * head_mult}],
        betas=tuple(cfg.get("betas", [0.9, 0.999])), weight_decay=float(cfg.get("weight_decay", 1e-2)))
    epochs = int(cfg.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = BinaryFocalLoss(alpha=float(cfg.get("focal_alpha", 0.25)), gamma=float(cfg.get("focal_gamma", 2)))

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patience = int(cfg.get("early_stopping_patience", 10))
    best_auc, best_epoch, history = -1.0, -1, []

    for epoch in range(1, epochs + 1):
        model.train()
        if cfg.get("freeze_bn", False):  # micro-batch가 작을 때 BN 통계 고정 옵션
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        t0, running, n_seen = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            loss = criterion(logits, y)
            (loss / accum).backward()
            running += loss.item() * len(y)
            n_seen += len(y)
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        yv, pv, val_bce = predict(model, val_loader, device, amp)
        val = summarize(yv, pv, 0.5)
        row = dict(epoch=epoch, train_loss=running / max(n_seen, 1), val_bce=val_bce,
                   val_auc=val["auc"], val_acc_at_0p5=val["accuracy"], sec=round(time.time() - t0, 1))
        history.append(row)
        print(json.dumps(row), flush=True)

        auc = val["auc"] if np.isfinite(val["auc"]) else -1.0
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            threshold = youden_threshold(yv, pv)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_auc": auc,
                        "threshold": threshold, "config": config}, out_dir / "best.pth")
        elif epoch - best_epoch >= patience:
            print(f"[INFO] early stop at epoch {epoch} (best {best_epoch}, val_auc={best_auc:.4f})")
            break

    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "config": config}, out_dir / "last.pth")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_epoch": best_epoch, "best_val_auc": best_auc, "history": history}
