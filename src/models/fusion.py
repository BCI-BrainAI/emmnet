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
# emmnet-integration-mri/src/models/fusion.py
import torch
import torch.nn as nn

class EMMNetFusion(nn.Module):
    """
    EMMNet Decoupled Modular Fusion (Method C)
    - EEG Feature (256) + MRI Feature (256) = 512차원 Concat
    - 빠른 초기 결과 도출을 위해 인코더 가중치 고정(Freeze)
    """
    def __init__(self, eeg_encoder, mri_encoder, num_classes=2, freeze_backbones=True):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.mri_encoder = mri_encoder

        # 1. 빠른 수렴과 모듈형 결합을 위해 인코더 고정
        if freeze_backbones:
            for p in self.eeg_encoder.parameters():
                p.requires_grad = False
            for p in self.mri_encoder.parameters():
                p.requires_grad = False

        # 2. 512차원을 입력받는 EMMNet 융합 분류기 (Adapter)
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg_x, age, mri_x):
        # EEG 인코더: [Batch, 256] 추출
        feat_eeg = self.eeg_encoder(eeg_x, age)
        
        # MRI 인코더: [Batch, 256] 추출
        # (MRI 인코더 호출 방식이 encoder.forward()인지 확인)
        feat_mri = self.mri_encoder(mri_x)
        
        # Mid-level Concatenation (총 512차원)
        fused = torch.cat([feat_eeg, feat_mri], dim=1) # [Batch, 512]
        logits = self.fusion_head(fused)
        return logits