"""config override, 정규화 캐시, 부트스트랩, 학습 로깅, 평가 CLI(test 잠금), 분석 스크립트."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data.mri_dataset import MRIDataset, med3d_normalize, volume_norm_stats
from evaluation.metrics import bootstrap_auc_ci, class_quantiles, confusion_metrics, roc_auc
from models.mri.classifier import MRIClassifier
from tests.test_training_pipeline import SIZE, make_synthetic
from training.losses import BinaryFocalLoss, focal_from_probs
from training.trainer import make_loader, train
from utils.config import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def _config(data_root: Path, **train_kw) -> dict:
    train_cfg = {"batch_size": 4, "micro_batch_size": 2, "epochs": 3, "lr": 1e-3, "amp": False, "device": "cpu",
                 "early_stopping_patience": 5, "seed": 0}
    train_cfg.update(train_kw)
    return {"data": {"processed_dir": str(data_root), "normalization": "percentile_zscore", "num_workers": 0},
            "input": {"channels": 1}, "model": {"proj_dim": 16}, "train": train_cfg}


def _run(cmd, **kw):
    return subprocess.run([sys.executable, *cmd], capture_output=True, text=True, timeout=300, cwd=ROOT, **kw)


def test_apply_overrides_and_unknown_key():
    cfg = {"train": {"lr": 1e-4, "betas": [0.9, 0.999]}, "data": {"overfit_n": None}}
    out = apply_overrides(cfg, ["train.lr=3e-4", "data.overfit_n=16", "train.betas=[0.8,0.9]"])
    assert out["train"]["lr"] == pytest.approx(3e-4) and out["data"]["overfit_n"] == 16 and out["train"]["betas"] == [0.8, 0.9]
    assert cfg["train"]["lr"] == 1e-4  # 원본 불변
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.lrr=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["train.lr"])


def test_load_config_env_and_cli_priority(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"processed_dir": "rel/dir"}, "model": {"pretrained_path": "p.pth"}}))
    assert load_config(cfg)["data"]["processed_dir"] == str((tmp_path / "rel/dir").resolve())
    monkeypatch.setenv("EMMNET_DATA", str(tmp_path / "env"))
    assert load_config(cfg)["data"]["processed_dir"] == str((tmp_path / "env").resolve())
    assert load_config(cfg, processed_dir=str(tmp_path / "cli"))["data"]["processed_dir"] == str((tmp_path / "cli").resolve())


def test_norm_stats_cache_matches_uncached(tmp_path):
    root = make_synthetic(tmp_path, {"train": 2, "val": 2, "test": 2})
    rows = list(csv.DictReader((root / "manifest.csv").open()))
    plain = MRIDataset(root, "train")
    with (root / "norm_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "split", "lo", "hi", "mean", "std", "fg_frac"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(image_id=r["image_id"], split=r["split"],
                            **volume_norm_stats(np.load(root / r["processed_relpath"]))))
    cached = MRIDataset(root, "train")
    assert len(cached.norm_stats) == len(rows)
    for i in range(len(plain)):
        assert torch.allclose(plain[i][0], cached[i][0], atol=1e-5)


def test_bootstrap_ci_and_quantiles():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    p = np.concatenate([rng.normal(0.35, 0.15, 40), rng.normal(0.65, 0.15, 40)])
    lo, hi = bootstrap_auc_ci(y, p, n_boot=300, seed=1)
    assert lo < roc_auc(y, p) < hi and (lo, hi) == bootstrap_auc_ci(y, p, n_boot=300, seed=1)
    assert np.isnan(bootstrap_auc_ci(np.zeros(5), np.random.rand(5))[0])
    q = class_quantiles(y, p)
    assert q["0"]["q50"] < q["1"]["q50"]
    assert "balanced_accuracy" in confusion_metrics(y, p, 0.5)


def test_focal_from_probs_matches_torch():
    logits = torch.tensor([0.3, -1.2, 2.0, -0.4]); t = torch.tensor([1.0, 0.0, 1.0, 1.0])
    ref = BinaryFocalLoss(0.25, 2.0)(logits, t).item()
    assert focal_from_probs(t.numpy(), torch.sigmoid(logits).numpy(), 0.25, 2.0) == pytest.approx(ref, rel=1e-5)


def test_training_logs_train_auc_snapshot_and_overfit_subset(tmp_path):
    root = make_synthetic(tmp_path / "d", {"train": 8, "val": 4, "test": 4}) if (tmp_path / "d").mkdir() is None else None
    cfg = _config(root)
    cfg["data"]["overfit_n"] = 4
    assert len(make_loader(cfg, "train", True, 2).dataset) == 4
    out = tmp_path / "run"
    result = train(MRIClassifier(proj_dim=16), cfg, out)
    for key in ("train_auc", "val_focal", "val_bce", "val_auc", "train_loss"):
        assert key in result["history"][0]
    assert len(json.loads((out / "history.json").read_text())) == len(result["history"])
    assert (out / "config.yaml").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["label_verified"] is False and "git_sha" in meta and meta["n_train"] == 4


def test_eval_cli_test_lock_predictions_and_analysis(tmp_path):
    root = make_synthetic(tmp_path / "d", {"train": 8, "val": 4, "test": 4}) if (tmp_path / "d").mkdir() is None else None
    out = tmp_path / "run"
    train(MRIClassifier(proj_dim=16), _config(root), out)
    ck = out / "best.pth"
    # last.pth에는 threshold가 없어 평가 거부
    bad = _run(["scripts/evaluate_mri_encoder.py", "--checkpoint", str(out / "last.pth"), "--splits", "val"])
    assert bad.returncode != 0 and "threshold" in bad.stderr

    r = _run(["scripts/evaluate_mri_encoder.py", "--checkpoint", str(ck), "--splits", "train", "val"])
    assert r.returncode == 0, r.stderr[-1500:]
    assert (out / "predictions_val.csv").exists() and (out / "val_metrics.json").exists()
    assert not (out / ".test_used.json").exists()

    t1 = _run(["scripts/evaluate_mri_encoder.py", "--checkpoint", str(ck), "--splits", "test"])
    assert t1.returncode == 0, t1.stderr[-1500:]
    t2 = _run(["scripts/evaluate_mri_encoder.py", "--checkpoint", str(ck), "--splits", "test"])
    assert t2.returncode != 0 and "이미 사용" in t2.stderr           # test 1회 잠금
    t3 = _run(["scripts/evaluate_mri_encoder.py", "--checkpoint", str(ck), "--splits", "test", "--force"])
    assert t3.returncode == 0
    rows = list(csv.DictReader((out / "predictions_test.csv").open()))
    assert len(rows) == 4 and {"image_id", "subject_id", "prob", "pred", "label"} <= set(rows[0])
    metrics = json.loads((out / "test_metrics.json").read_text())
    assert metrics["_meta"]["split"] == "test" and metrics["n"] == 4

    a = _run(["scripts/analyze_results.py", "--run-dir", str(out), "--n-boot", "50"])
    assert a.returncode == 0, a.stderr[-1500:]
    res = json.loads((out / "analysis.json").read_text())
    assert set(res["splits"]) == {"train", "val", "test"} and "age_only_auc" in res["splits"]["test"]
    assert res["history"]["epochs_run"] >= 1 and (out / "curves.png").exists()
