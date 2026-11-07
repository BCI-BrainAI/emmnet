#!/usr/bin/env python3
"""MRI encoder 256차원 feature 추출 (projection 출력, 분류 head 이전) -> .npz.

python scripts/export_features.py --checkpoint checkpoints/final_s0/best.pth --splits train val test

출력(체크포인트 폴더): features_<split>.npz
  features    (N, 256) float32   MRIClassifier.encoder 출력 (BN eval, dropout 없음)
  subject_ids (N,) str
  labels      (N,) int64         CN=0, MCI/AD=1
  image_ids   (N,) str           (같은 subject의 중복 scan 구분용)
가중치는 변경하지 않는다(eval, no_grad). test는 --splits test를 명시할 때만 처리.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.evaluate import load_model_from_checkpoint  # noqa: E402
from training.trainer import dataset_records, make_loader, resolve_device  # noqa: E402


@torch.no_grad()
def extract(model, config, split: str):
    device = resolve_device(config["train"].get("device", "auto"))
    model.to(device).eval()
    micro = int(config["train"].get("micro_batch_size", config["train"]["batch_size"]))
    loader = make_loader(config, split, False, micro)  # shuffle=False -> manifest 순서
    amp = bool(config["train"].get("amp", True))
    feats, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            z = model.encoder(x)
        feats.append(z.float().cpu().numpy())
        ys.append(y.numpy())
    recs = dataset_records(loader)
    f, y = np.concatenate(feats).astype(np.float32), np.concatenate(ys).astype(np.int64)
    assert len(recs) == len(f) == len(y), (len(recs), len(f), len(y))
    return f, y, recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Export 256-d MRI encoder features")
    ap.add_argument("--checkpoint", required=True, help="best.pth")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--processed-dir", help="데이터 경로 override (EMMNET_DATA도 가능)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = ckpt_path.parent
    model, config, ckpt = load_model_from_checkpoint(ckpt_path, args.processed_dir)
    for split in args.splits:
        f, y, recs = extract(model, config, split)
        path = out_dir / f"features_{split}.npz"
        np.savez(path, features=f, labels=y,
                 subject_ids=np.array([r["subject_id"] for r in recs]),
                 image_ids=np.array([r["image_id"] for r in recs]))
        print(f"{split}: features {f.shape}, n_subjects={len({r['subject_id'] for r in recs})}, "
              f"pos_rate={y.mean():.3f}, finite={bool(np.isfinite(f).all())} -> {path}")
    (out_dir / "features_meta.json").write_text(json.dumps(
        {"checkpoint": str(ckpt_path), "ckpt_epoch": int(ckpt.get("epoch", -1)),
         "val_auc": float(ckpt.get("val_auc", float("nan"))), "splits": args.splits,
         "proj_dim": config["model"]["proj_dim"], "normalization": config["data"].get("normalization")},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
