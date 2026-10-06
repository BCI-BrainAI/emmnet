#!/usr/bin/env python3
"""percentile_zscore 통계(lo/hi/mean/std)와 전경 비율을 1회 계산해 <processed_dir>/norm_stats.csv로 캐시.

MRIDataset이 파일이 있으면 자동으로 사용한다(결과는 캐시 없을 때와 동일, 매 epoch의 percentile/std 계산 생략).
전경 비율(fg_frac) 이상치는 `volume > 0` 마스크가 깨졌다는 신호(배경 잡음, zero-padding이 양수화)이므로 경고한다.

python scripts/cache_norm_stats.py --processed-dir <out_dir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.mri_dataset import NORM_STATS_FILE, volume_norm_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.processed_dir).resolve()
    out = root / NORM_STATS_FILE
    if out.exists() and not args.force:
        raise SystemExit(f"이미 있음: {out} (--force로 재계산)")
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    stats = []
    for i, r in enumerate(rows, 1):
        s = volume_norm_stats(np.load(root / r["processed_relpath"], allow_pickle=False))
        stats.append(dict(image_id=r["image_id"], split=r["split"], **s))
        if i % 50 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader(); w.writerows(stats)
    fg = np.array([s["fg_frac"] for s in stats])
    print(f"saved {out}; fg_frac mean={fg.mean():.3f} min={fg.min():.3f} max={fg.max():.3f}")
    bad = [s["image_id"] for s in stats if s["fg_frac"] > 0.9]
    if bad:
        print(f"[WARN] fg_frac>0.9 영상 {len(bad)}개(배경이 0이 아닐 수 있음): {bad[:10]} ... -> 시각 QC 필수")
    if fg.min() < 0.05:
        print("[WARN] fg_frac<0.05 영상 존재: 거의 빈 영상 가능")


if __name__ == "__main__":
    main()
