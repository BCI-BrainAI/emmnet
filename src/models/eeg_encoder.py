"""
EEG Encoder (TODO).

논문 근거(참고용, 미구현):
"The EEG encoder processes one-dimensional temporal signals using stacked
blocks of 1D convolution, batch normalization, and nonlinear activation,
producing a compact feature vector of length n." [EMMNet Sec 3.2, p.284]
1D-ResNet-18 채택, age를 EEG 채널에 추가 결합 [EMMNet Sec 4.2 p.286 / Sec 3.2 p.284].
담당자 배정 후 구현.
"""
from __future__ import annotations

import torch.nn as nn


class EEGEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("EEG encoder 미구현 — 담당자 배정 후 작성")
