#!/usr/bin/env python3
"""전처리 산출물에서 작은 subset 생성(스모크/디버그용). 심볼릭 링크 대신 파일 복사(링크는 Dataset이 거부).

python scripts/make_subset.py --processed-dir <out_dir> --out-dir <subset_dir> --per-split 8 4 4
split별 개수는 train val test 순서이며 클래스 균형(label 0/1 반반)으로 앞에서부터 선택한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-split", type=int, nargs=3, default=[8, 4, 4], metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()
    src, dst = Path(args.processed_dir).resolve(), Path(args.out_dir).resolve()
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"비어 있지 않은 out-dir: {dst}")
    with (src / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = []
    for split, n in zip(SPLITS, args.per_split):
        for label in ("0", "1"):
            chosen += [r for r in rows if r["split"] == split and r["label"] == label][: max(1, n // 2)]
    (dst / "volumes").mkdir(parents=True, exist_ok=True)
    for r in chosen:
        shutil.copy2(src / r["processed_relpath"], dst / r["processed_relpath"])
    with (dst / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(chosen)
    run = json.loads((src / "run.json").read_text(encoding="utf-8")) if (src / "run.json").exists() else {}
    run.update(subset_of=str(src), per_split=args.per_split)
    (dst / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"subset {len(chosen)}개 -> {dst}")


if __name__ == "__main__":
    main()
