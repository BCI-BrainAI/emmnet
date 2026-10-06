"""평가. 임계값은 체크포인트에 저장된 val 기반 Youden 값을 test에 고정 적용한다.

- 모델/정규화/전처리 설정은 현재 YAML이 아니라 체크포인트에 저장된 학습 당시 config를 사용한다
  (학습 후 YAML이 바뀌어도 조용히 불일치하지 않도록). 데이터 경로만 override 가능.
- 체크포인트에 threshold가 없으면 에러(0.5로 조용히 대체하지 않는다).

논문 MRI-only 수치(Acc 34.86%)는 joint 학습에서 분리된 값이라 목표로 삼지 않는다
[EMMNet Table 3 vs Table 4, p.287].
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.metrics import summarize
from models.mri.classifier import MRIClassifier
from training.trainer import dataset_records, make_loader, predict, resolve_device
from utils.config import ENV_DATA, resolve_data_dir

PRED_FIELDS = ("split", "image_id", "subject_id", "research_group", "age_at_scan", "sex", "label", "prob", "pred")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location="cpu")
    if "threshold" not in ckpt:
        raise KeyError(f"{path}: threshold 없음. best.pth(val 기반 Youden 저장)만 평가에 사용할 수 있다 (last.pth 불가).")
    return ckpt


def build_model_from_config(config: dict[str, Any]) -> MRIClassifier:
    return MRIClassifier(in_channels=config["input"]["channels"], proj_dim=config["model"]["proj_dim"],
                         dropout=float(config["model"].get("dropout", 0.0)))


def load_model_from_checkpoint(path: str | Path, processed_dir: str | None = None):
    """(model, config, ckpt). config는 체크포인트의 것을 쓰고 데이터 경로만 --processed-dir/EMMNET_DATA로 교체."""
    ckpt = load_checkpoint(path)
    config = ckpt["config"]
    if processed_dir or os.environ.get(ENV_DATA):
        resolve_data_dir(config, None, processed_dir)
    model = build_model_from_config(config)
    model.load_state_dict(ckpt["state_dict"])
    return model, config, ckpt


def evaluate(model, config: dict[str, Any], checkpoint_path: str | Path, split: str = "test") -> dict[str, float]:
    """하위 호환: 주어진 config로 평가(테스트/스모크용). CLI는 evaluate_split을 쓴다."""
    ckpt = load_checkpoint(checkpoint_path)
    model.load_state_dict(ckpt["state_dict"])
    return evaluate_split(model, config, ckpt, split)[0]


def evaluate_split(model, config: dict[str, Any], ckpt: dict[str, Any], split: str):
    """(metrics, prediction_rows). 예측 행은 manifest 순서 그대로이며 image_id/subject_id로 추적 가능."""
    device = resolve_device(config["train"].get("device", "auto"))
    model.to(device)
    micro = int(config["train"].get("micro_batch_size", config["train"]["batch_size"]))
    loader = make_loader(config, split, False, micro)
    y, p, bce = predict(model, loader, device, bool(config["train"].get("amp", True)))
    threshold = float(ckpt["threshold"])
    out = summarize(y, p, threshold)
    out.update(bce=bce, n=float(len(y)), ckpt_epoch=float(ckpt.get("epoch", -1)))
    rows = []
    for r, yi, pi in zip(dataset_records(loader), y, p):
        rows.append(dict(split=split, image_id=r["image_id"], subject_id=r["subject_id"],
                         research_group=r.get("research_group", ""), age_at_scan=r.get("age_at_scan", ""),
                         sex=r.get("sex", ""), label=int(yi), prob=float(pi), pred=int(pi >= threshold)))
    return out, rows
