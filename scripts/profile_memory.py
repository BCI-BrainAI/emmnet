#!/usr/bin/env python3
"""GPU 메모리 사전 측정(runbook 4번의 heredoc 대체). OOM 직전 batch는 사용하지 않는다.

CUDA_VISIBLE_DEVICES=0 python scripts/profile_memory.py --sizes 1 2 4 8 --spatial 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.mri.classifier import MRIClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--spatial", type=int, default=256)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--headroom", type=float, default=0.85, help="총 메모리 대비 허용 비율(초과 시 권장에서 제외)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA 없음")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    model = MRIClassifier().cuda()
    opt = torch.optim.AdamW(model.parameters())
    ok = []
    print(f"GPU0 {torch.cuda.get_device_name(0)} total={total:.1f} GiB, spatial={args.spatial}, amp={not args.no_amp}")
    for bs in sorted(args.sizes):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 1, *(args.spatial,) * 3, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
                out = model(x).sum()
            out.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"bs={bs}: {peak:.1f} GiB ({100 * peak / total:.0f}%)")
            if peak <= args.headroom * total:
                ok.append(bs)
            del x, out
        except torch.cuda.OutOfMemoryError:
            print(f"bs={bs}: OOM")
            break
    if ok:
        eff = [b for b in ok if 16 % b == 0]
        print(f"권장 micro_batch_size = {max(eff) if eff else max(ok)} (effective 16의 약수 중 headroom 이내 최대)")
        print("적용: --set train.micro_batch_size=<값>")


if __name__ == "__main__":
    main()
