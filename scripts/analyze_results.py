#!/usr/bin/env python3
"""학습 결과 분석(인수인계 §6-2, 6-5). numpy + (선택) matplotlib만 사용.

python scripts/analyze_results.py --run-dir checkpoints/base

입력: history.json, predictions_{train,val,test}.csv (evaluate_mri_encoder.py 산출물; 있는 split만 분석)
출력: analysis.json, curves.png, 콘솔 요약
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import bootstrap_auc_ci, class_quantiles, roc_auc, summarize  # noqa: E402

SPLITS = ("train", "val", "test")


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    def num(key):
        return np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows])
    return dict(y=num("label").astype(int), p=num("prob"), pred=num("pred").astype(int), age=num("age_at_scan"),
                sex=np.array([r["sex"] for r in rows]), group=np.array([r["research_group"] for r in rows]),
                image_id=np.array([r["image_id"] for r in rows]))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze_split(d: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    y, p = d["y"], d["p"]
    thr_pred = d["pred"]
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "pos_frac": float(y.mean()), "auc": roc_auc(y, p)}
    out["auc_ci95"] = list(bootstrap_auc_ci(y, p, n_boot, seed))
    tp = int(((thr_pred == 1) & (y == 1)).sum()); tn = int(((thr_pred == 0) & (y == 0)).sum())
    fp = int(((thr_pred == 1) & (y == 0)).sum()); fn = int(((thr_pred == 0) & (y == 1)).sum())
    out["at_ckpt_threshold"] = dict(tp=tp, tn=tn, fp=fp, fn=fn,
                                    sensitivity=tp / (tp + fn) if tp + fn else float("nan"),
                                    specificity=tn / (tn + fp) if tn + fp else float("nan"),
                                    accuracy=(tp + tn) / len(y), pred_pos_rate=float(thr_pred.mean()))
    out["prob_quantiles_by_class"] = class_quantiles(y, p)
    out["pos_rate_by_group"] = {g: dict(n=int((d["group"] == g).sum()), mean_prob=float(p[d["group"] == g].mean()),
                                        pred_pos_rate=float(thr_pred[d["group"] == g].mean()))
                                for g in sorted(set(d["group"])) if g}
    age = d["age"]
    out["age_only_auc"] = roc_auc(y, age) if np.isfinite(age).all() else float("nan")  # 나이를 점수로 직접 사용
    out["age_only_auc_ci95"] = list(bootstrap_auc_ci(y, age, n_boot, seed)) if np.isfinite(age).all() else [float("nan")] * 2
    out["spearman_prob_age"] = spearman(p, age)
    sex = d["sex"]
    if len(set(sex)) >= 2:  # 한 split에 성별 2종이 있을 때만(M/F 문자열 가정 없이 첫 값 기준 이진화)
        s_bin = (sex == sorted(set(sex))[0]).astype(float)
        out["sex_only_auc"] = roc_auc(y, s_bin)
    out["prob_degenerate"] = bool(np.ptp(p) < 1e-4)  # 모든 확률이 사실상 동일 -> 학습 실패 신호
    return out


def diagnose_history(h: list[dict]) -> dict:
    if not h:
        return {"error": "history 비어 있음"}
    val = np.array([r["val_auc"] for r in h], dtype=float)
    best = int(np.nanargmax(val)) if np.isfinite(val).any() else -1
    out = {"epochs_run": len(h), "best_epoch": h[best]["epoch"] if best >= 0 else None,
           "best_val_auc": float(val[best]) if best >= 0 else None, "last_epoch": h[-1]["epoch"],
           "nan_in_history": any(not math.isfinite(v) for r in h for v in r.values() if isinstance(v, (int, float))),
           "flags": []}
    f = out["flags"]
    if best == len(h) - 1:
        f.append("best_epoch==last: 미수렴/early-stop 미발동 가능 -> epochs 증가 또는 lr 점검")
    if np.isfinite(val).any() and np.nanmax(val) < 0.55:
        f.append("val_auc<0.55 전 구간: 학습 신호 없음 -> sweep 전에 데이터/전처리/라벨 점검(인수인계 §7)")
    if "train_auc" in h[0]:
        tr = np.array([r["train_auc"] for r in h], dtype=float)
        out["final_train_auc_trainmode"] = float(tr[-1])
        if tr[-1] > 0.9 and best >= 0 and val[best] < tr[-1] - 0.15:
            f.append("train_auc>>val_auc: 과적합 패턴 (train_auc는 BN train-mode 값)")
        if tr[-1] < 0.6:
            f.append("train_auc<0.6: 미학습(under-fit) 가능 -> lr/overfit 점검")
    if len(val) >= 6 and best >= 0 and best < len(h) // 2 and val[-3:].mean() < val[best] - 0.05:
        f.append("val_auc가 초중반 peak 후 하락: 과적합/불안정")
    if len(val) >= 4 and np.nanstd(val) > 0.08:
        f.append(f"val_auc 변동 큼(std={np.nanstd(val):.3f}): val 크기가 작을 수 있어 best epoch 선택이 노이즈에 민감")
    return out


def plot_curves(h: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [r["train_loss"] for r in h], label="train focal")
    if "val_focal" in h[0]:
        ax[0].plot(ep, [r["val_focal"] for r in h], label="val focal")
    ax[0].plot(ep, [r["val_bce"] for r in h], "--", label="val BCE")
    ax[0].set_title("loss (compare focal with focal only)"); ax[0].legend()
    ax[1].plot(ep, [r["val_auc"] for r in h], label="val AUC")
    if "train_auc" in h[0]:
        ax[1].plot(ep, [r["train_auc"] for r in h], label="train AUC (train-mode)")
    ax[1].axhline(0.5, color="gray", lw=0.8); ax[1].set_title("AUC"); ax[1].legend()
    ax[2].plot(ep, [r["lr"] for r in h]); ax[2].set_title("lr (backbone)")
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run_dir)

    result: dict = {"run_dir": str(run)}
    meta = run / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        result["meta"] = {k: m.get(k) for k in ("git_sha", "git_dirty", "label_verified", "n_train", "n_val", "micro_batch_size")}
        if not m.get("label_verified", True):
            print("[WARN] label_verified=False (라벨 미검증): 결과 해석에 한계")
    hist_path = run / "history.json"
    if hist_path.exists():
        h = json.loads(hist_path.read_text(encoding="utf-8"))
        result["history"] = diagnose_history(h)
        result["curves_png"] = str(run / "curves.png") if plot_curves(h, run / "curves.png") else None
    result["splits"] = {}
    for split in SPLITS:
        pf = run / f"predictions_{split}.csv"
        if pf.exists():
            result["splits"][split] = analyze_split(load_predictions(pf), args.n_boot, args.seed)
    (run / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # 콘솔 요약
    hh = result.get("history")
    if hh:
        print(f"[history] epochs={hh['epochs_run']} best_epoch={hh['best_epoch']} best_val_auc={hh['best_val_auc']:.4f} "
              f"last={hh['last_epoch']} nan={hh['nan_in_history']}")
        for fl in hh["flags"]:
            print("  [FLAG]", fl)
    for split, r in result["splits"].items():
        c = r["at_ckpt_threshold"]
        print(f"[{split}] n={r['n']} pos={r['n_pos']} AUC={r['auc']:.4f} CI95=[{r['auc_ci95'][0]:.3f},{r['auc_ci95'][1]:.3f}] "
              f"sens={c['sensitivity']:.3f} spec={c['specificity']:.3f} acc={c['accuracy']:.3f} "
              f"age-only AUC={r['age_only_auc']:.3f} rho(prob,age)={r['spearman_prob_age']:.3f}"
              + (" [DEGENERATE probs]" if r["prob_degenerate"] else ""))
        for cls, q in r["prob_quantiles_by_class"].items():
            print(f"    prob quantiles class{cls}: " + " ".join(f"{k}={v:.3f}" for k, v in q.items()))
        for g, v in r["pos_rate_by_group"].items():
            print(f"    group {g}: n={v['n']} mean_prob={v['mean_prob']:.3f} pred_pos_rate={v['pred_pos_rate']:.3f}")
    print(f"saved: {run / 'analysis.json'}")


if __name__ == "__main__":
    main()
