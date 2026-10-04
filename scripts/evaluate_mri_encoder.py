#!/usr/bin/env python3
"""MRI Encoder 평가 CLI. 임계값은 체크포인트(best.pth)에 저장된 val 기반 값을 사용.

python scripts/evaluate_mri_encoder.py --checkpoint checkpoints/run1/best.pth --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.evaluate import evaluate  # noqa: E402
from models.mri.classifier import MRIClassifier  # noqa: E402
from train_mri_encoder import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRI encoder")
    parser.add_argument("--config", default="configs/mri_encoder.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--processed-dir")
    parser.add_argument("--out-json", help="결과 JSON 저장 경로")
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve(), args.processed_dir)
    model = MRIClassifier(in_channels=config["input"]["channels"], proj_dim=config["model"]["proj_dim"],
                          dropout=float(config["model"].get("dropout", 0.0)))
    metrics = evaluate(model=model, config=config, checkpoint_path=args.checkpoint, split=args.split)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
