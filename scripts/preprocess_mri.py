#!/usr/bin/env python3
"""MRI 전처리 CLI. 실제 로직은 src/data/mri_dataset.py (TODO).

사전 조건: 저장소 루트에서 `pip install -e .` 실행 (pyproject.toml).
"""
from __future__ import annotations

import argparse

from data.mri_dataset import build_mri_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="MRI 전처리 (zero-pad+resize 256^3, min-max)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    build_mri_dataset(raw_dir=args.raw_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
