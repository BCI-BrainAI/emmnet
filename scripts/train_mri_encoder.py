#!/usr/bin/env python3
"""CLI: MRI encoder(+probe head) 학습.

python scripts/train_mri_encoder.py --checkpoint-dir checkpoints/run1 --set train.lr=3e-4
(데이터 경로: --processed-dir 또는 환경변수 EMMNET_DATA)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.mri.classifier import MRIClassifier  # noqa: E402
from models.mri.mri_encoder import load_med3d_pretrained  # noqa: E402
from training.trainer import train  # noqa: E402
from utils.config import add_config_args, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRI encoder (MRI-only probe)")
    add_config_args(parser)
    parser.add_argument("--checkpoint-dir", default="checkpoints/run1")
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve(), args.processed_dir, args.overrides)
    model = MRIClassifier(in_channels=config["input"]["channels"], proj_dim=config["model"]["proj_dim"],
                          dropout=float(config["model"].get("dropout", 0.0)))

    pretrained = Path(config["model"]["pretrained_path"])
    if not pretrained.exists():
        raise FileNotFoundError(f"pretrained checkpoint 없음: {pretrained}")
    missing, unexpected = load_med3d_pretrained(model.encoder, pretrained)
    print(f"[INFO] pretrained 로드. missing={missing} unexpected={unexpected}")
    if missing or unexpected:  # backbone 키는 완전히 일치해야 한다.
        raise RuntimeError("Pretrained key mismatch")

    result = train(model=model, config=config, checkpoint_dir=args.checkpoint_dir)
    print(f"[DONE] best_epoch={result['best_epoch']} best_val_auc={result['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
