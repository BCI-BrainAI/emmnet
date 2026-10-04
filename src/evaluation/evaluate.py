"""평가. 임계값은 체크포인트에 저장된 val 기반 Youden 값을 test에 고정 적용한다.

논문 MRI-only 수치(Acc 34.86%)는 joint 학습에서 분리된 값이라 목표로 삼지 않는다
[EMMNet Table 3 vs Table 4, p.287].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from evaluation.metrics import summarize
from training.trainer import make_loader, predict, resolve_device


def evaluate(model, config: dict[str, Any], checkpoint_path: str | Path, split: str = "test") -> dict[str, float]:
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    device = resolve_device(config["train"].get("device", "auto"))
    model.to(device)
    micro = int(config["train"].get("micro_batch_size", config["train"]["batch_size"]))
    loader = make_loader(config, split, False, micro)
    y, p, bce = predict(model, loader, device, bool(config["train"].get("amp", True)))
    threshold = float(ckpt.get("threshold", 0.5))
    out = summarize(y, p, threshold)
    out.update(bce=bce, n=float(len(y)), ckpt_epoch=float(ckpt.get("epoch", -1)))
    return out
