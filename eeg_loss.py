import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    """Softmax focal loss; scalar alpha=.25 uniformly weights classes.

    A per-class alpha vector may be supplied explicitly. The paper does not
    specify whether its scalar alpha is uniform or binary asymmetric.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        alpha = torch.as_tensor(alpha, dtype=torch.float32)
        if alpha.ndim > 1 or not torch.isfinite(alpha).all() or (alpha < 0).any() or gamma < 0:
            raise ValueError('Invalid focal loss parameters')
        self.register_buffer('alpha',alpha)
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits,target,reduction='none')
        alpha = self.alpha
        if alpha.ndim:
            if alpha.numel() != logits.shape[1]:
                raise ValueError('One alpha weight per class required')
            alpha = alpha[target]
        return (alpha * (1-ce.neg().exp()).pow(self.gamma) * ce).mean()
