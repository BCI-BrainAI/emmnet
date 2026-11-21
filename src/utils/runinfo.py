"""실행 메타데이터(git sha, 환경)와 test split 접근 잠금."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_LOCK = ".test_used.json"


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "--no-optional-locks", *args], cwd=repo, capture_output=True, text=True, timeout=10,
                                  check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(status) if status is not None else None, "git_dirty_files": (status or "").splitlines()[:20]}


def run_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch
    meta = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv, "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "host": platform.node(), **git_info()}
    meta.update(extra or {})
    return meta


def claim_test_access(run_dir: Path, who: str, force: bool = False) -> None:
    """test split은 run당 1회만. 이미 사용했으면 --force 없이는 거부한다."""
    lock = Path(run_dir) / TEST_LOCK
    if lock.exists() and not force:
        raise PermissionError(f"test split은 이미 사용됨({lock.read_text(encoding='utf-8')[:200]}). "
                              "의도한 재평가라면 --force (기록은 남는다).")
    lock.write_text(json.dumps({"by": who, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "forced": force and lock.exists()}),
                    encoding="utf-8")
