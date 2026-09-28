"""
Fusion + Classification head (TODO).

논문 근거(참고용, 미구현):
- Ablation 결과 mid-level concatenation이 최고 성능(Full model 74.53%/77.37%)
  vs early concat(58.65%), mid addition(72.23%), cross-attention(51.48%),
  gated fusion(72.41%) [EMMNet Table 5, p.288].
- 주의(Teardown): Table 5에 "Mid Concat" 행이 별도로 없고, 본문의
  "Mid-level fusion via concatenation achieves the best performance"
  [Sec 5.1, p.288] 서술로 미루어 "Full model (ours)" 행이 곧 mid-concat으로
  추정됨 — 표 자체엔 라벨링이 애매함.
- fused representation -> FC 1~2개로 최종 분류 [EMMNet Sec 3.2, p.284].
"""
from __future__ import annotations

import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Fusion/classification head 미구현")
