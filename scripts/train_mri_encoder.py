#!/usr/bin/env python3
"""CLI: MRI encoder(+probe head) 학습.

python scripts/train_mri_encoder.py --config configs/mri_encoder.yaml --checkpoint-dir checkpoints/run1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.mri.classifier import MRIClassifier  # noqa: E402
from models.mri.mri_encoder import load_med3d_pretrained  # noqa: E402
from training.trainer import train  # noqa: E402


def load_config(config_path: Path, processed_dir: str | None = None) -> dict:
    """YAML 로드. 상대 경로는 YAML 폴더 기준 절대 경로로 변환(--processed-dir은 CWD 기준)."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    if processed_dir:
        config["data"]["processed_dir"] = str(Path(processed_dir).expanduser().resolve())
    else:
        config["data"]["processed_dir"] = str((base / config["data"]["processed_dir"]).resolve())
    config["model"]["pretrained_path"] = str((base / config["model"]["pretrained_path"]).resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRI encoder (MRI-only probe)")
    parser.add_argument("--config", default="configs/mri_encoder.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/run1")
    parser.add_argument("--processed-dir", help="YAML data.processed_dir 덮어쓰기")
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve(), args.processed_dir)
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
