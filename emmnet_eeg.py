"""EMMNet adapter around unmodified CEEDNet source. See README.md.
Age must be standardized unless training-set age_mean/age_std are supplied.
"""
import math
import torch
from torch import nn
from ceednet_original.resnet_1d import ResNet1D


class EEGEncoder(nn.Module):
    """Original ResNet1D with use_age='conv' and out_dims=256.

    eeg_channels excludes the added age channel. Default 19 uses the paper's
    EEG electrodes. Use 20 to include ECG; exact EMMNet selection is unspecified.
    """
    def __init__(self, eeg_channels=19, seq_length=2000, *, age_mean=None,
                 age_std=None, activation='mish', fc_stages=3):
        super().__init__()
        if eeg_channels < 1 or seq_length < 32:
            raise ValueError('Positive channels and seq_length >= 32 required')
        if (age_mean is None) != (age_std is None):
            raise ValueError('Supply both age statistics or neither')
        if age_mean is not None and (not math.isfinite(age_mean)
                or not math.isfinite(age_std) or age_std <= 0):
            raise ValueError('Finite age statistics and positive std required')
        self.eeg_channels, self.seq_length, self.feature_dim = eeg_channels, seq_length, 256
        self.register_buffer('age_mean', None if age_mean is None else torch.tensor(float(age_mean)))
        self.register_buffer('age_std', None if age_std is None else torch.tensor(float(age_std)))
        self.backbone = ResNet1D(
            block='basic', conv_layers=[2,2,2,2], in_channels=eeg_channels,
            out_dims=256, seq_length=seq_length, base_channels=64,
            use_age='conv', fc_stages=fc_stages, dropout=0.1,
            activation=activation, base_pool='max', final_pool='average',
            zero_init_residual=False)

    def forward(self, eeg, age):
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (self.eeg_channels, self.seq_length):
            raise ValueError(f'Expected EEG [B, {self.eeg_channels}, {self.seq_length}]')
        if not eeg.is_floating_point():
            raise TypeError('EEG must be floating point')
        if not torch.isfinite(eeg).all() or not torch.isfinite(age).all():
            raise ValueError('EEG and age must be finite')
        if tuple(age.shape) not in ((eeg.shape[0],), (eeg.shape[0],1)):
            raise ValueError('Age must have shape [B] or [B,1]')
        age = age.to(device=eeg.device, dtype=eeg.dtype)
        if self.age_mean is not None:
            age = (age-self.age_mean)/(self.age_std+1e-8)
        return self.backbone(eeg, age)


class EEGClassifier(nn.Module):
    """Optional EEG-only classifier. Returns logits."""
    def __init__(self, num_classes=2, **encoder_options):
        super().__init__()
        if num_classes < 2:
            raise ValueError('At least two classes required')
        self.encoder = EEGEncoder(**encoder_options)
        self.classifier = nn.Linear(256,num_classes)

    def forward(self,eeg,age):
        return self.classifier(self.encoder(eeg,age))


class EMMNetFeatureFusion(nn.Module):
    """MRI encoder is external; pass paired MRI features [B,256].
    One/two FC layers are supported. Exact EMMNet head settings are unknown.
    """
    def __init__(self,num_classes=2,fusion_layers=1,fusion_hidden=256,**encoder_options):
        super().__init__()
        if num_classes < 2 or fusion_layers not in (1,2) or fusion_hidden < 1:
            raise ValueError('Invalid classifier configuration')
        self.eeg_encoder = EEGEncoder(**encoder_options)
        self.classifier = (nn.Linear(512,num_classes) if fusion_layers == 1 else
            nn.Sequential(nn.Linear(512,fusion_hidden),nn.ReLU(),nn.Linear(fusion_hidden,num_classes)))

    def forward(self,eeg,age,mri_features):
        if tuple(mri_features.shape) != (eeg.shape[0],256):
            raise ValueError('MRI features must have shape [B,256]')
        return self.classifier(torch.cat((self.eeg_encoder(eeg,age),mri_features),dim=1))


if __name__ == '__main__':
    model = EEGClassifier().eval()
    with torch.no_grad():
        print(model(torch.zeros(1,19,2000), torch.zeros(1)).shape)
