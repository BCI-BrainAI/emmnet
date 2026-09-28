"""
학습 루프.

논문 근거(공유 학습 세팅, EMMNet 전체 모델 기준) [EMMNet Sec 4.2, p.286]:
- optimizer: AdamW (beta1=0.9, beta2=0.999)
- scheduler: cosine LR decay + warmup
- batch size: 16
- loss: focal loss (alpha=0.25, gamma=2) [Eq.1, p.285]
- MRI 단독 학습 시 사용 샘플 수: 4,000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def train(model, config: dict[str, Any], checkpoint_dir: str | Path) -> None:
    """
    MRI encoder 학습 진입점 (TODO).

    Parameters
    ----------
    model : emmnet.models.mri_encoder.MRIEncoder
    config : configs/mri_encoder.yaml 로드 결과
    checkpoint_dir : 학습된 가중치 저장 위치 (pretrained/ 와 분리, 우리 학습 산출물 전용)
    """
    raise NotImplementedError(
        "Trainer 미구현 — MRI dataset(mri_dataset.py)과 fusion 헤드 완성 후 작성 예정"
    )
