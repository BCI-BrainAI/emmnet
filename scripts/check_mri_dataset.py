#!/usr/bin/env python3
"""전처리 산출물 점검: split/클래스 분포, 피험자 중복, 제외 사유, 정규화 통계, QC 미리보기 PNG.

python scripts/check_mri_dataset.py --processed-dir data/processed/adni_screening_scaled_v1 --n-preview 6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.mri_dataset import MRIDataset, SPLITS, med3d_normalize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--n-preview", type=int, default=6, help="split당 아닌 전체 미리보기 샘플 수")
    ap.add_argument("--n-stats", type=int, default=20, help="정규화 통계를 볼 샘플 수")
    args = ap.parse_args()
    root = Path(args.processed_dir)
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8-sig")))
    print(f"[manifest] {len(rows)} samples; run.json status:",
          json.loads((root / "run.json").read_text()).get("status"))

    print("\n[split x label]")
    table = Counter((r["split"], r["research_group"], r["label"]) for r in rows)
    for split in SPLITS:
        parts = {f"{g}->{l}": c for (s, g, l), c in sorted(table.items()) if s == split}
        print(f"  {split}: n={sum(parts.values())} {parts}")
        labels = {l for (s, g, l) in table if s == split}
        if labels != {"0", "1"}:
            print(f"  [WARN] {split}에 한 클래스만 있음 -> AUC 계산 불가")

    subj = {}
    leaks = [r["subject_id"] for r in rows if subj.setdefault(r["subject_id"], r["split"]) != r["split"]]
    print(f"\n[leakage] subjects in multiple splits: {len(set(leaks))}  (0이어야 함)")

    ex = root / "excluded.csv"
    if ex.exists():
        reasons = Counter(r["exclusion_reason"].split(":")[0] for r in csv.DictReader(ex.open(encoding="utf-8-sig")))
        print("\n[excluded]", dict(reasons))

    print("\n[label audit] researchGroup 라벨은 미검증(label_verified=False). "
          "MCI=1 병합이 팀 합의인지, 진단 변경자 제외 여부를 별도 확인할 것.")
    print("  age/sex 분포(split별 평균 나이):")
    for split in SPLITS:
        ages = [float(r["age_at_scan"]) for r in rows if r["split"] == split and r.get("age_at_scan")]
        if ages:
            print(f"   {split}: mean_age={np.mean(ages):.1f} n={len(ages)}")

    ds = MRIDataset(root, "train")
    k = min(args.n_stats, len(ds))
    stats = []
    for i in np.linspace(0, len(ds) - 1, k).astype(int):
        v = np.load(root / ds.records[i]["processed_relpath"])
        z = med3d_normalize(v)
        fg = v > 0
        stats.append((float(fg.mean()), float(z[fg].mean()), float(z[fg].std()), float(z.max())))
    s = np.array(stats)
    print(f"\n[normalization check on {k} train volumes] foreground_frac mean={s[:,0].mean():.3f} "
          f"(range {s[:,0].min():.3f}-{s[:,0].max():.3f}); z mean≈{s[:,1].mean():.2e}, std≈{s[:,2].mean():.3f}, max z={s[:,3].max():.2f}")
    if s[:, 0].min() > 0.9:
        print("  [WARN] 전경 비율이 90% 초과: 배경이 0이 아닐 수 있음(min-max 시 배경이 0으로 안 감) -> 영상 확인")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[preview] matplotlib 없음 -> 건너뜀")
        return
    n = min(args.n_preview, len(rows))
    sel = np.linspace(0, len(rows) - 1, n).astype(int)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8))
    axes = np.atleast_2d(axes).reshape(3, n)
    for j, i in enumerate(sel):
        r = rows[i]
        v = np.load(root / r["processed_relpath"])
        d, h, w = v.shape
        for a, sl in enumerate((v[d // 2], v[:, h // 2], v[:, :, w // 2])):
            axes[a, j].imshow(sl.T if a else sl, cmap="gray", origin="lower")
            axes[a, j].axis("off")
        axes[0, j].set_title(f"{r['research_group']} {r['split']}\n{r['image_id']}", fontsize=8)
    out = root / "qc_preview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"\n[preview] saved {out} -> 뇌 영역/방향/크롭이 정상인지 눈으로 확인 후 qc_status 갱신")


if __name__ == "__main__":
    main()
