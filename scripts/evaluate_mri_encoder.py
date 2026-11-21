#!/usr/bin/env python3
"""MRI Encoder 평가 CLI (학습 당시 config + best.pth의 val 기반 임계값 사용).

python scripts/evaluate_mri_encoder.py --checkpoint checkpoints/run1/best.pth --splits train val
python scripts/evaluate_mri_encoder.py --checkpoint checkpoints/run1/best.pth --splits test   # run당 1회

출력(체크포인트 폴더): <split>_metrics.json, predictions_<split>.csv
test는 run당 1회만 허용(.test_used.json 잠금). 재평가는 --force.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.evaluate import PRED_FIELDS, evaluate_split, load_model_from_checkpoint  # noqa: E402
from utils.runinfo import claim_test_access, run_meta  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRI encoder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"])
    parser.add_argument("--processed-dir", help="데이터 경로만 override (환경변수 EMMNET_DATA도 가능)")
    parser.add_argument("--force", action="store_true", help="test 재평가 허용(잠금 무시, 기록은 남음)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent
    model, config, ckpt = load_model_from_checkpoint(ckpt_path, args.processed_dir)
    if "test" in args.splits:
        claim_test_access(run_dir, who="evaluate_mri_encoder.py", force=args.force)
    for split in args.splits:
        metrics, rows = evaluate_split(model, config, ckpt, split)
        print(f"== {split} (ckpt epoch {int(metrics['ckpt_epoch'])}, thr={metrics['threshold']:.4f}"
              f"{', val 기반 임계값을 val에 적용 -> 낙관적' if split == 'val' else ''})")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        out = dict(metrics, _meta=run_meta({"split": split, "checkpoint": str(ckpt_path),
                                            "normalization": config["data"].get("normalization")}))
        (run_dir / f"{split}_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        with (run_dir / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"saved: {run_dir / f'{split}_metrics.json'}, {run_dir / f'predictions_{split}.csv'}")


if __name__ == "__main__":
    main()
