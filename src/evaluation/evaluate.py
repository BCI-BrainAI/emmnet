"""
평가 스크립트.

논문 보고 지표(참고): Accuracy/Precision/Sensitivity/F1 [EMMNet Table 3-5].
MRI encoder 단독 검증 시 Table 4 "MRI only" 결과(Acc 34.86%, Sens 41.22%,
chance 이하)가 joint 학습에서 분리된 수치라는 점 주의 — 독립 베이스라인
재현 시 이 수치를 목표로 삼지 말 것 [EMMNet Table 3 vs Table 4, p.287].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate(model, config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, float]:
    """저장된 checkpoint로 MRI encoder 평가 (TODO). 반환값: metric name -> value."""
    raise NotImplementedError(
        "Evaluation 미구현 — trainer.py 완성 후 동일 데이터 파이프라인 재사용 예정"
    )
