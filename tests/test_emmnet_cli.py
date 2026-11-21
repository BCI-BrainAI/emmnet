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


def test_sweep_resumes_interrupted_and_skips_done(env, capsys, monkeypatch):
    monkeypatch.setattr(cli, "is_running", lambda d: False)
    (env / "ck" / "lr3e-4").mkdir(parents=True)
    (env / "ck" / "lr3e-4" / "resume.pth").write_text("1")      # 중단됨 -> 재개
    (env / "ck" / "lr1e-3").mkdir(parents=True)
    (env / "ck" / "lr1e-3" / "last.pth").write_text("1")        # 완료 -> 건너뜀
    cli.main(["sweep", "--dry-run", "--lrs", "1e-3", "3e-4", "1e-4"])
    out = capsys.readouterr().out
    assert "[SKIP] 완료됨: lr1e-3" in out
    assert "[RESUME] 중단된 run 재개: lr3e-4" in out
    assert "run_experiment.sh lr3e-4 --gpu 0 --resume -- --set train.lr=3e-4" in out
    assert "run_experiment.sh lr1e-4 --gpu 1 -- --set train.lr=1e-4" in out


def test_running_run_is_not_relaunched(env, capsys, monkeypatch):
    monkeypatch.setattr(cli, "is_running", lambda d: True)
    (env / "ck" / "lr3e-4").mkdir(parents=True)
    (env / "ck" / "lr3e-4" / "x").write_text("1")
    cli.main(["sweep", "--dry-run", "--lrs", "3e-4"])
    assert "[SKIP] 진행 중: lr3e-4" in capsys.readouterr().out


def test_detect_gpus_picks_idle(monkeypatch):
    class R:
        stdout = "0, 20000, 16376\n1, 12, 16376\n2, 400, 16376\n"
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: R())
    assert cli.detect_gpus() == [1, 2]
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert cli.detect_gpus() == [0]


def test_auto_detects_data_and_gpus_without_init(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "CFG_PATH", tmp_path / "none.json")
    monkeypatch.delenv("EMMNET_DATA", raising=False)
    monkeypatch.delenv("EMMNET_GPUS", raising=False)
    d = tmp_path / "scan" / "adni_screening_scaled_v1"
    (d / "volumes").mkdir(parents=True)
    (d / "manifest.csv").write_text("x")
    monkeypatch.setattr(cli, "SCAN_ROOTS", [str(tmp_path / "scan")])
    monkeypatch.setattr(cli, "find_data_candidates", lambda max_depth=4: [d.resolve()])
    monkeypatch.setattr(cli, "detect_gpus", lambda: [1])
    cfg = cli.load_cfg()
    assert cli.need_data(cfg) == str(d.resolve()) and cli.get_gpus(cfg) == [1]


def test_multiple_data_candidates_ask_user(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "find_data_candidates", lambda max_depth=4: [tmp_path / "a", tmp_path / "b"])
    with pytest.raises(SystemExit) as e:
        cli.need_data({"data": None})
    assert "여러 개" in str(e.value)


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
    (d / "last.pth").write_text("x")
    (d / "test_metrics.json").write_text(json.dumps({"auc": 0.77}))
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "r1" in out and "done" in out and "0.800" in out and "0.770" in out


def test_missing_data_path_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CFG_PATH", tmp_path / "none.json")
    monkeypatch.delenv("EMMNET_DATA", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["check", "--dry-run"])
