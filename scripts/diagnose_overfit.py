#!/usr/bin/env python3
"""overfit 실패 원인 분리 진단 (서버 GPU 1장, 수 분).

python scripts/diagnose_overfit.py [--n 16] [--steps 120] [--variants base noamp bn_eval bce scratch]
(데이터 경로: --processed-dir 또는 EMMNET_DATA / 설정: configs/mri_encoder.yaml, --set key=value)

A) 입력 점검: NaN/범위/전경 비율, 샘플 간 상관(입력이 서로 구분되는가)
B) 초기(사전학습) feature 점검: 샘플 간 feature 코사인 유사도/상대 변동(feature가 입력에 둔감한가)
C) 1-step grad norm: backbone / projection / head
D) 변형별 짧은 암기 실험: base | noamp(bf16 끔) | bn_eval(BN 통계 고정) | bce(focal 대신) | scratch(사전학습 없음)
   | focal_g0(alpha .25, gamma 0) | focal_g1(gamma 1) | focal_a5(alpha .5, gamma 2)
   각 변형 결과: 최종 train loss, eval-mode AUC, logit 표준편차(상수 출력이면 ~0)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import roc_auc  # noqa: E402
from models.mri.classifier import MRIClassifier  # noqa: E402
from models.mri.mri_encoder import load_med3d_pretrained  # noqa: E402
from training.losses import BinaryFocalLoss  # noqa: E402
from training.trainer import make_loader  # noqa: E402
from utils.config import add_config_args, load_config  # noqa: E402

VARIANTS = ["base", "noamp", "bn_eval", "bce", "scratch", "focal_g0", "focal_g1", "focal_a5"]


def build(config, pretrained: bool) -> MRIClassifier:
    m = MRIClassifier(in_channels=config["input"]["channels"], proj_dim=config["model"]["proj_dim"],
                      dropout=float(config["model"].get("dropout", 0.0)))
    if pretrained:
        missing, unexpected = load_med3d_pretrained(m.encoder, config["model"]["pretrained_path"])
        assert not missing and not unexpected, (missing, unexpected)
    return m


def const_baseline(alpha=0.25, gamma=2.0) -> tuple[float, float]:
    """balanced 라벨에서 상수 확률 p를 내는 모델의 focal loss 최솟값 (p, loss)."""
    ps = np.linspace(0.01, 0.99, 981)
    f = 0.5 * (alpha * (1 - ps) ** gamma * -np.log(ps) + (1 - alpha) * ps ** gamma * -np.log(1 - ps))
    i = int(f.argmin())
    return float(ps[i]), float(f[i])


def pdist_stats(x: torch.Tensor) -> dict:
    flat = x.flatten(1).float()
    flat = flat - flat.mean(1, keepdim=True)
    flat = F.normalize(flat, dim=1)
    c = flat @ flat.T
    iu = torch.triu_indices(len(c), len(c), 1)
    off = c[iu[0], iu[1]]
    return {"corr_mean": float(off.mean()), "corr_min": float(off.min()), "corr_max": float(off.max())}


@torch.no_grad()
def eval_all(model, x, y, device, amp):
    model.eval()
    logits = []
    for i in range(0, len(x), 4):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits.append(model(x[i:i + 4].to(device)).float().cpu())
    lg = torch.cat(logits)
    p = torch.sigmoid(lg).numpy()
    return {"auc": roc_auc(y.numpy(), p), "logit_std": float(lg.std()), "p_min": float(p.min()), "p_max": float(p.max()),
            "focal": float(BinaryFocalLoss(0.25, 2.0)(lg, y.float())), "bce": float(F.binary_cross_entropy_with_logits(lg, y.float()))}


def run_variant(name, config, x, y, device, steps, batch, lr):
    torch.manual_seed(0)
    model = build(config, pretrained=(name != "scratch")).to(device)
    amp = name != "noamp"
    focal = {"focal_g0": (0.25, 0.0), "focal_g1": (0.25, 1.0), "focal_a5": (0.5, 2.0)}.get(name, (0.25, 2.0))
    crit = (lambda lg, t: F.binary_cross_entropy_with_logits(lg.float(), t.float())) if name == "bce" else BinaryFocalLoss(*focal)
    new = list(model.encoder.projection.parameters()) + list(model.head.parameters())
    ids = {id(p) for p in new}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in ids], "lr": lr},
                             {"params": new, "lr": lr * 10}], weight_decay=1e-2)
    losses, t0, n = [], time.time(), len(x)
    g = torch.Generator().manual_seed(0)
    step = 0
    while step < steps:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            model.train()
            if name == "bn_eval":
                for mod in model.modules():
                    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                        mod.eval()
            xb, yb = x[idx].to(device), y[idx].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                lg = model(xb)
            loss = crit(lg, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break
    res = eval_all(model, x, y, device, amp)
    res.update({"steps": steps, "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])),
                "sec": round(time.time() - t0, 1)})
    del model, opt
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--variants", nargs="+", default=VARIANTS[:5], choices=VARIANTS)
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()
    overrides = list(args.overrides or []) + [f"data.overfit_n={args.n}"]
    config = load_config(Path(args.config).resolve(), args.processed_dir, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {}

    loader = make_loader(config, "train", False, args.n)
    x, y = next(iter(loader))
    print(f"[A] 입력 x={tuple(x.shape)} y={y.tolist()}")
    fg = (x != 0).flatten(1).float().mean(1)
    a = {"nan": bool(torch.isnan(x).any()), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
         "std": float(x.std()), "nonzero_frac_mean": float(fg.mean()), "nonzero_frac_min": float(fg.min())}
    a.update(pdist_stats(x))
    out["A_input"] = a
    print("   ", json.dumps(a))

    model = build(config, True).to(device)
    model.eval()
    feats, logits = [], []
    with torch.no_grad():
        for i in range(0, len(x), 4):
            xb = x[i:i + 4].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                f = model.encoder.avgpool(model.encoder.backbone(xb)).flatten(1)
                logits.append(model.head(model.encoder.projection(f)).float().squeeze(1).cpu())
            feats.append(f.float().cpu())
    f = torch.cat(feats)
    fn = F.normalize(f - 0, dim=1)
    cs = fn @ fn.T
    iu = torch.triu_indices(len(cs), len(cs), 1)
    b = {"feat_dim": f.shape[1], "pooled_cos_mean": float(cs[iu[0], iu[1]].mean()), "pooled_cos_min": float(cs[iu[0], iu[1]].min()),
         "rel_sample_std": float((f.std(0) / (f.abs().mean(0) + 1e-8)).mean()), "init_logit_std": float(torch.cat(logits).std())}
    out["B_init_features(eval BN)"] = b
    print("[B] 사전학습 초기 feature:", json.dumps(b))

    model.train()
    crit = BinaryFocalLoss(0.25, 2.0)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        lg = model(x[:args.batch].to(device))
    crit(lg, y[:args.batch].to(device)).backward()
    gn = {}
    for nm, mod in [("backbone", model.encoder.backbone), ("projection", model.encoder.projection), ("head", model.head)]:
        gn[nm] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in mod.parameters() if p.grad is not None)))
    out["C_grad_norm"] = gn
    print("[C] 1-step grad norm:", json.dumps(gn))
    del model
    torch.cuda.empty_cache()

    pc, fc = const_baseline()
    print(f"[D] 상수예측기 기준: p={pc:.2f} focal={fc:.4f} (balanced) | steps={args.steps} batch={args.batch} lr={args.lr}")
    out["D"] = {"const_baseline": {"p": pc, "focal": fc}}
    for v in args.variants:
        r = run_variant(v, config, x, y, device, args.steps, args.batch, args.lr)
        out["D"][v] = r
        print(f"    {v:8s} loss {r['loss_first10']:.4f}->{r['loss_last10']:.4f}  eval AUC={r['auc']:.3f} "
              f"focal={r['focal']:.4f} bce={r['bce']:.3f} logit_std={r['logit_std']:.3f} p=[{r['p_min']:.3f},{r['p_max']:.3f}] ({r['sec']}s)", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("saved:", args.out)
    print("해석: 정상이면 base/bce 중 하나 이상에서 loss<<기준선(0.07), eval AUC~1, logit_std>1. "
          "scratch만 성공=사전학습 feature 문제, noamp만 성공=bf16 문제, bn_eval만 성공=BN(배치4) 문제, 전부 실패=입력/라벨 파이프라인 또는 head 문제.")


if __name__ == "__main__":
    main()
