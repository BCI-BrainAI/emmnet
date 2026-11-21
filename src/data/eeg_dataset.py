"""EEG Dataset (TODO). EEG 담당자와 협업 후 작성."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EEG dataset 미구현")

    def __len__(self):  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx):  # pragma: no cover
        raise NotImplementedError
