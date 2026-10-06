"""scripts/emmnet.py (서버 자동화 CLI): 명령 생성(--dry-run), GPU 분배, run 요약."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("emmnet_cli", ROOT / "scripts" / "emmnet.py")
cli = importlib.util.module_from_spec(spec)
sys.modules["emmnet_cli"] = cli
spec.loader.exec_module(cli)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"data": str(tmp_path / "data"), "gpus": [0, 1], "ckpt_root": str(tmp_path / "ck")}))
    monkeypatch.setattr(cli, "CFG_PATH", cfg)
    return tmp_path


def test_sweep_distributes_over_gpus_and_skips_existing(env, capsys):
    (env / "ck" / "lr3e-4").mkdir(parents=True)
    (env / "ck" / "lr3e-4" / "x").write_text("1")
    cli.main(["sweep", "--dry-run", "--lrs", "1e-3", "3e-4", "1e-4"])
    out = capsys.readouterr().out
    assert "[SKIP] 이미 존재: lr3e-4" in out
    assert "run_experiment.sh lr1e-3 --gpu 0 -- --set train.lr=1e-3" in out
    assert "run_experiment.sh lr1e-4 --gpu 1 -- --set train.lr=1e-4" in out


def test_final_builds_seed_runs(env, capsys):
    cli.main(["final", "--lr", "3e-4", "--dry-run"])
    out = capsys.readouterr().out
    for s in (0, 1, 2):
        assert f"final_s{s}" in out and f"train.seed={s}" in out and "train.lr=3e-4" in out
    assert "emmnet.py test final_s0 final_s1 final_s2" in out


def test_overfit_and_check_commands(env, capsys):
    cli.main(["overfit", "--dry-run"])
    out = capsys.readouterr().out
    assert "data.overfit_n=16" in out and "train.early_stopping_patience=999" in out and "analyze_results.py" in out
    cli.main(["check", "--dry-run"])
    assert "check_mri_dataset.py" in capsys.readouterr().out


def test_status_summarizes_run(env, capsys):
    d = env / "ck" / "r1"
    d.mkdir(parents=True)
    hist = [dict(epoch=1, val_auc=0.6, train_auc=0.7), dict(epoch=2, val_auc=0.8, train_auc=0.9)]
    (d / "history.json").write_text(json.dumps(hist))
    (d / "train.log").write_text("[DONE] best_epoch=2")
    (d / "test_metrics.json").write_text(json.dumps({"auc": 0.77}))
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "r1" in out and "done" in out and "0.800" in out and "0.770" in out


def test_missing_data_path_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CFG_PATH", tmp_path / "none.json")
    monkeypatch.delenv("EMMNET_DATA", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["check", "--dry-run"])
