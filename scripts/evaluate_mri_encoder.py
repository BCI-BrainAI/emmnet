#!/usr/bin/env python3
"""MRI Encoder 평가 CLI. evaluate() 본체는 TODO.

사전 조건: 저장소 루트에서 `pip install -e .` 실행 (pyproject.toml).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from models.mri.mri_encoder import MRIEncoder
from evaluation.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRI encoder")
    parser.add_argument("--config", default="configs/mri_encoder.yaml")
    parser.add_argument("--checkpoint", required=True, help="checkpoints/ 하위 .pth 경로")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = MRIEncoder(
        in_channels=config["input"]["channels"],
        proj_dim=config["model"]["proj_dim"],
    )
    metrics = evaluate(model=model, config=config, checkpoint_path=args.checkpoint)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
