#!/usr/bin/env python3
"""서버 작업 자동화 CLI. 경로/GPU는 `init`으로 1회 저장하면 이후 타이핑이 필요 없다(표준 라이브러리만 사용).

  python scripts/emmnet.py init --data <전처리 out_dir> --gpus 0 1
  python scripts/emmnet.py setup [--install-torch]     # torch/CUDA 확인, pip install -e, pytest, 가중치 해시
  python scripts/emmnet.py check                       # 데이터 점검 + 정규화 통계 캐시
  python scripts/emmnet.py baseline base [--test]      # 기존 run 평가(train/val) + 분석 (+ test 1회)
  python scripts/emmnet.py overfit                     # overfit 점검 학습 + 분석
  python scripts/emmnet.py train base2 [--gpu 0] [--set train.lr=3e-4 ...]   # 백그라운드 학습
  python scripts/emmnet.py sweep [--lrs 1e-3 3e-4 1e-4 3e-5]                 # lr sweep, GPU에 자동 분배
  python scripts/emmnet.py final --lr 3e-4 [--seeds 0 1 2]                   # 시드별 최종 학습
  python scripts/emmnet.py test final_s0 final_s1 final_s2                   # test 1회 평가 + 평균±표준편차
  python scripts/emmnet.py status                      # 모든 run 요약 (val AUC 기준)
  python scripts/emmnet.py log base2                   # 학습 로그 tail
모든 명령에 --dry-run을 붙이면 실행하지 않고 명령만 출력한다.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = Path(os.environ.get("EMMNET_CFG", ROOT / ".emmnet.json"))
PRETRAINED_SHA = ("38b3a174", "f61da3")  # runbook 기록(resnet_18.pth)
DEFAULT_LRS = ["1e-3", "3e-4", "1e-4", "3e-5"]


# ---------------------------------------------------------------- config / helpers
def load_cfg() -> dict:
    cfg = {"data": os.environ.get("EMMNET_DATA"), "gpus": [0], "ckpt_root": "checkpoints"}
    if CFG_PATH.exists():
        cfg.update(json.loads(CFG_PATH.read_text(encoding="utf-8")))
    return cfg


def need_data(cfg: dict) -> str:
    if not cfg.get("data"):
        sys.exit("데이터 경로 미설정: python scripts/emmnet.py init --data <전처리 out_dir>")
    return cfg["data"]


def env_for(cfg: dict) -> dict:
    env = {**os.environ, "CKPT_ROOT": str(ckpt_root(cfg)), "PYTHONUNBUFFERED": "1"}
    if cfg.get("data"):
        env["EMMNET_DATA"] = cfg["data"]
    return env


def ckpt_root(cfg: dict) -> Path:
    p = Path(cfg["ckpt_root"])
    return p if p.is_absolute() else ROOT / p


def run(cmd: list[str], cfg: dict, dry: bool, check: bool = True, gpu: int | None = None) -> int:
    env = env_for(cfg)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("$ " + shlex.join(cmd), flush=True)
    if dry:
        return 0
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if check and rc != 0:
        sys.exit(f"실패(rc={rc}): {shlex.join(cmd)}")
    return rc


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, f"scripts/{script}", *args]


def exp_cmd(name: str, gpu: int, sets: list[str]) -> list[str]:
    cmd = ["bash", "scripts/run_experiment.sh", name, "--gpu", str(gpu), "--"]
    return cmd + (["--set", *sets] if sets else [])


def launch_queue(cfg: dict, gpu: int, jobs: list[list[str]], dry: bool) -> None:
    """GPU 하나에 job을 순차 실행하는 백그라운드 큐(세션 종료 후에도 유지)."""
    chain = " ; ".join(shlex.join(j) for j in jobs)
    print(f"[GPU {gpu}] 큐 {len(jobs)}개:\n  " + chain.replace(" ; ", "\n  "), flush=True)
    if dry:
        return
    root = ckpt_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    log = open(root / f"_queue_gpu{gpu}.out", "a")
    subprocess.Popen(["bash", "-c", chain], cwd=ROOT, env=env_for(cfg), stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    print(f"  -> 백그라운드 시작. 큐 로그: {root / f'_queue_gpu{gpu}.out'}", flush=True)


def run_dir(cfg: dict, name: str) -> Path:
    return ckpt_root(cfg) / name


def split_new(cfg: dict, names_sets: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    out = []
    for name, sets in names_sets:
        d = run_dir(cfg, name)
        if d.exists() and any(d.iterdir()):
            print(f"[SKIP] 이미 존재: {name}")
        else:
            out.append((name, sets))
    return out


def distribute(cfg: dict, gpus: list[int], items: list[tuple[str, list[str]]], dry: bool) -> None:
    items = split_new(cfg, items)
    if not items:
        print("새로 시작할 run 없음")
        return
    queues: dict[int, list[list[str]]] = {g: [] for g in gpus}
    for i, (name, sets) in enumerate(items):
        g = gpus[i % len(gpus)]
        queues[g].append(exp_cmd(name, g, sets))
    for g, jobs in queues.items():
        if jobs:
            launch_queue(cfg, g, jobs, dry)
    print("진행 확인: python scripts/emmnet.py status")


# ---------------------------------------------------------------- run summary
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarize_run(d: Path) -> dict | None:
    hist = read_json(d / "history.json")
    if hist is None:
        return None
    log = d / "train.log"
    done = log.exists() and "[DONE]" in log.read_text(encoding="utf-8", errors="ignore")
    recent = log.exists() and time.time() - log.stat().st_mtime < 600
    state = "done" if done else ("running" if recent else "stopped?")
    best = max(hist, key=lambda r: r["val_auc"] if r["val_auc"] == r["val_auc"] else -1) if hist else None
    tm = read_json(d / "test_metrics.json")
    return dict(name=d.name, state=state, epochs=len(hist), best_epoch=best["epoch"] if best else None,
                best_val_auc=best["val_auc"] if best else None, train_auc=hist[-1].get("train_auc") if hist else None,
                test_auc=tm["auc"] if tm else None, test_used=(d / ".test_used.json").exists())


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_status(cfg: dict, only: list[str] | None = None) -> None:
    root = ckpt_root(cfg)
    rows = [r for d in sorted(root.glob("*")) if d.is_dir() and (not only or d.name in only)
            for r in [summarize_run(d)] if r]
    if not rows:
        print(f"run 없음: {root}")
        return
    print(f"{'run':24} {'state':9} {'ep':>3} {'best_ep':>7} {'val_auc':>8} {'train_auc':>9} {'test_auc':>8}")
    for r in rows:
        print(f"{r['name']:24} {r['state']:9} {r['epochs']:>3} {fmt(r['best_epoch']):>7} {fmt(r['best_val_auc']):>8} "
              f"{fmt(r['train_auc']):>9} {fmt(r['test_auc']):>8}{'  [test 사용]' if r['test_used'] else ''}")


# ---------------------------------------------------------------- commands
def cmd_init(a, cfg):
    cfg = dict(cfg)
    if a.data:
        cfg["data"] = str(Path(a.data).expanduser().resolve())
    if a.gpus:
        cfg["gpus"] = a.gpus
    if a.ckpt_root:
        cfg["ckpt_root"] = a.ckpt_root
    manifest = Path(cfg["data"] or "") / "manifest.csv"
    if cfg.get("data") and not manifest.exists():
        print(f"[WARN] manifest.csv 없음: {manifest}")
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {CFG_PATH}\n{json.dumps(cfg, indent=2, ensure_ascii=False)}")


def cmd_setup(a, cfg):
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info >= (3, 14):
        print("[WARN] Python 3.14: cu124/cu126 torch 휠이 없을 수 있음 -> 3.11/3.12 venv 권장")
    probe = "import torch;print(torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
    if subprocess.run([sys.executable, "-c", probe], cwd=ROOT).returncode != 0 or a.install_torch:
        if not a.install_torch:
            print("[ERR] torch 없음. --install-torch로 설치(cu126)하거나 직접 설치 후 재실행")
            if not a.dry_run:
                sys.exit(1)
        run([sys.executable, "-m", "pip", "install", "torch", "--index-url", a.torch_index], cfg, a.dry_run)
    run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cfg, a.dry_run)
    run([sys.executable, "-c", probe], cfg, a.dry_run)
    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cfg, a.dry_run)
    w = ROOT / "pretrained" / "resnet_18.pth"
    if w.exists() and not a.dry_run:
        import hashlib
        h = hashlib.sha256()
        with w.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        hx = h.hexdigest()
        ok = hx.startswith(PRETRAINED_SHA[0]) and hx.endswith(PRETRAINED_SHA[1])
        print(f"resnet_18.pth sha256={hx[:8]}...{hx[-6:]} {'OK' if ok else '[WARN] 기록값과 다름'}")
    elif not w.exists():
        print("[ERR] pretrained/resnet_18.pth 없음")


def cmd_check(a, cfg):
    data = need_data(cfg)
    run(py("check_mri_dataset.py", "--processed-dir", data, "--n-preview", "8"), cfg, a.dry_run)
    if not (Path(data) / "norm_stats.csv").exists():
        run(py("cache_norm_stats.py", "--processed-dir", data), cfg, a.dry_run)
    rj = read_json(Path(data) / "run.json")
    if rj:
        print("split_counts:", rj.get("split_counts"), "| selected/processed/excluded:",
              rj.get("selected"), rj.get("processed"), rj.get("excluded"))


def eval_and_analyze(cfg, name, splits, dry):
    d = run_dir(cfg, name)
    run(py("evaluate_mri_encoder.py", "--checkpoint", str(d / "best.pth"), "--splits", *splits), cfg, dry,
        gpu=cfg["gpus"][0])
    run(py("analyze_results.py", "--run-dir", str(d)), cfg, dry)


def cmd_baseline(a, cfg):
    need_data(cfg)
    eval_and_analyze(cfg, a.name, ["train", "val"], a.dry_run)
    if a.test:
        eval_and_analyze(cfg, a.name, ["test"], a.dry_run)


def cmd_overfit(a, cfg):
    need_data(cfg)
    sets = [f"data.overfit_n={a.n}", f"train.lr={a.lr}", f"train.epochs={a.epochs}", "train.early_stopping_patience=999"]
    run(exp_cmd(a.name, cfg["gpus"][0], sets), cfg, a.dry_run)
    run(py("analyze_results.py", "--run-dir", str(run_dir(cfg, a.name))), cfg, a.dry_run)
    print("판정: train_auc(분석 출력의 [train] AUC, history의 train_auc)가 ~1.0 이면 정상")


def cmd_train(a, cfg):
    need_data(cfg)
    gpu = a.gpu if a.gpu is not None else cfg["gpus"][0]
    distribute(cfg, [gpu], [(a.name, a.set or [])], a.dry_run)


def cmd_sweep(a, cfg):
    need_data(cfg)
    gpus = a.gpus or cfg["gpus"]
    items = [(f"lr{lr}", [f"train.lr={lr}", *(a.set or [])]) for lr in a.lrs]
    distribute(cfg, gpus, items, a.dry_run)
    print("선택 기준: val AUC만 사용(test 금지). 끝나면 status 확인 -> final")


def cmd_final(a, cfg):
    need_data(cfg)
    gpus = a.gpus or cfg["gpus"]
    items = [(f"final_s{s}", [f"train.lr={a.lr}", f"train.seed={s}", *(a.set or [])]) for s in a.seeds]
    distribute(cfg, gpus, items, a.dry_run)
    print("끝나면: python scripts/emmnet.py test " + " ".join(n for n, _ in items))


def cmd_test(a, cfg):
    need_data(cfg)
    aucs = []
    for name in a.names:
        d = run_dir(cfg, name)
        if (d / ".test_used.json").exists() and not a.force:
            print(f"[SKIP] {name}: test 이미 사용됨(재평가는 --force)")
        else:
            run(py("evaluate_mri_encoder.py", "--checkpoint", str(d / "best.pth"), "--splits", "test",
                   *(["--force"] if a.force else [])), cfg, a.dry_run, gpu=cfg["gpus"][0])
            run(py("analyze_results.py", "--run-dir", str(d)), cfg, a.dry_run)
    if a.dry_run:
        return
    rows = [(n, read_json(run_dir(cfg, n) / "test_metrics.json")) for n in a.names]
    rows = [(n, m) for n, m in rows if m]
    if rows:
        import statistics as st
        print("\n[test 요약]")
        for key in ("auc", "sensitivity", "specificity", "accuracy", "balanced_accuracy"):
            vals = [m[key] for _, m in rows if m.get(key) == m.get(key)]
            if vals:
                sd = st.stdev(vals) if len(vals) > 1 else float("nan")
                print(f"  {key:18} {st.mean(vals):.4f} ± {sd:.4f}  (n_runs={len(vals)}; " +
                      ", ".join(f"{v:.3f}" for v in vals) + ")")


def cmd_status(a, cfg):
    print_status(cfg, a.names or None)


def cmd_log(a, cfg):
    log = run_dir(cfg, a.name) / "train.log"
    if not log.exists():
        sys.exit(f"로그 없음: {log}")
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    print("\n".join(lines[-a.n:]))


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="실행하지 않고 명령만 출력")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)  # 서브명령 뒤에도 --dry-run 허용
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)

    s = sub.add_parser("init", parents=[common]); s.set_defaults(fn=cmd_init)
    s.add_argument("--data"); s.add_argument("--gpus", type=int, nargs="+"); s.add_argument("--ckpt-root")
    s = sub.add_parser("setup", parents=[common]); s.set_defaults(fn=cmd_setup)
    s.add_argument("--install-torch", action="store_true")
    s.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu126")
    s = sub.add_parser("check", parents=[common]); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("baseline", parents=[common]); s.set_defaults(fn=cmd_baseline)
    s.add_argument("name", nargs="?", default="base"); s.add_argument("--test", action="store_true")
    s = sub.add_parser("overfit", parents=[common]); s.set_defaults(fn=cmd_overfit)
    s.add_argument("--name", default="overfit"); s.add_argument("--n", type=int, default=16)
    s.add_argument("--lr", default="1e-3"); s.add_argument("--epochs", type=int, default=40)
    s = sub.add_parser("train", parents=[common]); s.set_defaults(fn=cmd_train)
    s.add_argument("name"); s.add_argument("--gpu", type=int); s.add_argument("--set", nargs="+", action="extend")
    s = sub.add_parser("sweep", parents=[common]); s.set_defaults(fn=cmd_sweep)
    s.add_argument("--lrs", nargs="+", default=DEFAULT_LRS); s.add_argument("--gpus", type=int, nargs="+")
    s.add_argument("--set", nargs="+", action="extend")
    s = sub.add_parser("final", parents=[common]); s.set_defaults(fn=cmd_final)
    s.add_argument("--lr", required=True); s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--gpus", type=int, nargs="+"); s.add_argument("--set", nargs="+", action="extend")
    s = sub.add_parser("test", parents=[common]); s.set_defaults(fn=cmd_test)
    s.add_argument("names", nargs="+"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("status", parents=[common]); s.set_defaults(fn=cmd_status); s.add_argument("names", nargs="*")
    s = sub.add_parser("log", parents=[common]); s.set_defaults(fn=cmd_log); s.add_argument("name"); s.add_argument("-n", type=int, default=25)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a, load_cfg())


if __name__ == "__main__":
    main()
