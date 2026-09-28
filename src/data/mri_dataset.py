"""
MRI Dataset.

전처리 스펙 (구현 시 반드시 반영, 논문 근거):
- zero-padding 후 256^3 resize (enclosing-cube crop 방식보다 성능 우수했다고
  보고됨) [EMMNet Sec 3.1, p.283]
- intensity: min-max normalization to [0, 1] (subject 단위) [EMMNet Sec 3.1, p.283]
  * Med3D 원 사전학습은 percentile truncation(0.5~99.5) + z-score였음
    [Med3D Eq.2, p.4] — 정규화 방식 불일치 인지하고 진행할 것.

Teardown — subject-level split 검증 필요:
"disjoint partitioning into training, validation, and test sets, supporting
robust evaluation without subject-level overlap" [EMMNet Sec 4.1, p.285]는
전체 데이터셋(EEG 세그먼트 기준) 차원의 서술. MRI 학습에 쓴 "4,000 training
samples"[EMMNet Sec 4.2, p.286]가 subject당 여러 augmented copy를 만든
것이라면 train/val/test 간 subject 분리를 별도로 명시적 검증해야 함
(CNN12 재현 때 지적된 leakage 이슈와 동일 패턴).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def build_mri_dataset(raw_dir: str | Path, out_dir: str | Path) -> None:
    """raw NIfTI -> zero-pad+resize 256^3 + min-max 정규화 전처리 파이프라인 (TODO)."""
    raise NotImplementedError("MRI 전처리 파이프라인 미구현 — 실제 데이터 확보 후 작성")


class MRIDataset(Dataset):
    """전처리된 MRI 텐서를 로드하는 torch Dataset (TODO)."""

    def __init__(self, processed_dir: str | Path, split: str, manifest: Any = None):
        raise NotImplementedError("MRIDataset 미구현 — subject-level split 설계 확정 후 작성")

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # pragma: no cover
        raise NotImplementedError
