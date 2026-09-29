"""학습 루프 (TODO)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def train(model, config: dict[str, Any], checkpoint_dir: str | Path) -> None:
    """MRI encoder 학습 진입점. Dataset/평가 방식 확정 후 구현 예정."""
    raise NotImplementedError("Trainer 미구현")
