#!/usr/bin/env python3
"""CLI: MRI encoder 학습."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from models.mri.mri_encoder import MRIEncoder, load_med3d_pretrained
from training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRI encoder")
    parser.add_argument("--config", default="configs/mri_encoder.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model = MRIEncoder(
        in_channels=config["input"]["channels"],
        proj_dim=config["model"]["proj_dim"],
    )

    pretrained_path = (config_path.parent / config["model"]["pretrained_path"]).resolve()
    if pretrained_path.exists():
        missing, unexpected = load_med3d_pretrained(model, pretrained_path)
        print(f"[INFO] pretrained 로드 완료. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[WARN] pretrained checkpoint 없음: {pretrained_path} -- 랜덤 초기화로 진행")

    train(model=model, config=config, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
