#!/usr/bin/env python3
"""MRI Encoder 학습 CLI. Trainer 본체는 아직 TODO — 구조(설정 로드/모델 생성/가중치 로드)만 우선 마련.

사전 조건: 저장소 루트에서 `pip install -e .` 실행 (pyproject.toml).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from models.mri_encoder import MRIEncoder, load_med3d_pretrained
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
        seg_style=config["model"]["seg_style"],
    )

    pretrained_path = (config_path.parent / config["model"]["pretrained_path"]).resolve()
    if pretrained_path.exists():
        missing, unexpected = load_med3d_pretrained(model, pretrained_path)
        print(f"[INFO] pretrained 로드 완료. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[WARN] pretrained checkpoint 없음: {pretrained_path} — 랜덤 초기화로 진행")

    train(model=model, config=config, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
