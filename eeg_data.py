"""Paper-aligned EEG loading, preprocessing and subject-disjoint metadata."""
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, resample_poly
from torch.utils.data import Dataset

EEG_CHANNELS = ('Fp1','F3','C3','P3','O1','Fp2','F4','C4','P4','O2',
                'F7','T3','T5','F8','T4','T6','Fz','Cz','Pz')

def channel_key(name):
    name = str(name).upper().strip().removeprefix('EEG ').split('-')[0].strip()
    return {'T7':'T3','P7':'T5','T8':'T4','P8':'T6','ECG':'EKG'}.get(name,name)

def subject_key(value):
    value = str(value)
    return str(int(value)) if value.isdigit() else value

def load_splits(path, annotation=None, num_classes=2):
    metadata = json.loads(Path(path).read_text(encoding='utf-8'))
    ages = {}
    if annotation:
        entries = json.loads(Path(annotation).read_text(encoding='utf-8'))['data']
        if isinstance(entries, dict):
            entries = [dict(v, serial=k) for k,v in entries.items()]
        ages = {subject_key(x['serial']): x.get('age') for x in entries}
    result, seen = {}, set()
    for split,key in [('train','train_split'),('val','validation_split'),('test','test_split')]:
        if key not in metadata or not metadata[key]:
            raise ValueError(f'Missing or empty {key}')
        result[split] = []
        for source in metadata[key]:
            item = dict(source)
            sid = subject_key(item['serial'])
            if sid in seen:
                raise ValueError(f'Duplicate subject or split leakage: {sid}')
            seen.add(sid)
            age = item.get('age', ages.get(sid))
            if age is None or not np.isfinite(float(age)) or float(age) < 0:
                raise ValueError(f'Valid age required for subject {sid}; supply annotation.json')
            label = item['class_label']
            if isinstance(label, bool) or int(label) != label or not 0 <= int(label) < num_classes:
                raise ValueError(f'Invalid class_label for {sid}: {label}')
            item.update(serial=sid, age=float(age), class_label=int(label))
            result[split].append(item)
    return result

def preprocess_signal(signal, sfreq, channel_names, include_ecg=False):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or len(channel_names) != signal.shape[0] or not np.isfinite(signal).all():
        raise ValueError('Finite EEG [channels,time] and matching channel names required')
    if not np.isfinite(sfreq) or sfreq <= 140:
        raise ValueError('Source sampling rate must exceed 140 Hz for 70 Hz band-pass')
    names = [channel_key(n) for n in channel_names]
    wanted = list(EEG_CHANNELS) + (['EKG'] if include_ecg else [])
    indices = []
    for name in wanted:
        matches = [i for i,n in enumerate(names) if n == channel_key(name)]
        if len(matches) != 1:
            raise ValueError(f'Exactly one channel {name} required; found {len(matches)}')
        indices.append(matches[0])
    signal = signal[indices]
    # Filter the entire recording before cropping, without ICA/artifact removal.
    sos = butter(4, [0.5,70], btype='bandpass', fs=sfreq, output='sos')
    signal = sosfiltfilt(sos, signal, axis=-1)
    ratio = Fraction(200 / float(sfreq)).limit_denominator(10000)
    signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
    return np.asarray(signal * 1e6, dtype=np.float32)  # EDF/MNE volts -> microvolts

class EMMNetEEGDataset(Dataset):
    def __init__(self, samples, eeg_dir, split='train', seq_length=2000,
                 include_ecg=False, stats=None, noise_std=0.001, eval_crops=8):
        if split not in ('train','val','test') or seq_length < 32 or eval_crops < 1 or noise_std < 0:
            raise ValueError('Invalid dataset options')
        self.samples, self.eeg_dir, self.split = samples, Path(eeg_dir), split
        self.seq_length, self.include_ecg = seq_length, include_ecg
        self.stats, self.noise_std, self.eval_crops = stats, noise_std, eval_crops
        self.paths = []
        for item in samples:
            sid = item['serial']
            filename = f'{int(sid):05d}' if sid.isdigit() else sid
            path = self.eeg_dir / item.get('eeg_file', filename+'.edf')
            if not path.is_file():
                raise FileNotFoundError(path)
            self.paths.append(path)

    def __len__(self):
        return len(self.samples)

    def load_recording(self, idx):
        path = self.paths[idx]
        if path.suffix.lower() == '.npz':
            # Portable input contract: signal in volts, sfreq, channel_names.
            with np.load(path, allow_pickle=False) as record:
                signal = preprocess_signal(record['signal'], float(record['sfreq']),
                                           record['channel_names'], self.include_ecg)
        else:
            import mne
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            try:
                signal = preprocess_signal(raw.get_data(), raw.info['sfreq'], raw.ch_names, self.include_ecg)
            finally:
                raw.close()
        if signal.shape[-1] < self.seq_length:
            raise ValueError(f'{path}: recording shorter than {self.seq_length/200:g} seconds')
        return signal

    def __getitem__(self, idx):
        signal = self.load_recording(idx)
        end = signal.shape[-1] - self.seq_length
        starts = ([int(np.random.randint(end+1))] if self.split == 'train' else
                  np.linspace(0,end,self.eval_crops,dtype=int).tolist())
        eeg = np.stack([signal[:,s:s+self.seq_length] for s in starts])
        age = self.samples[idx]['age']
        if self.stats:
            eeg = (eeg - np.asarray(self.stats['signal_mean'])[None,:,None]) / np.asarray(self.stats['signal_std'])[None,:,None]
            age = (age - self.stats['age_mean']) / self.stats['age_std']
        if self.split == 'train' and self.noise_std:
            eeg = eeg + np.random.normal(0,self.noise_std,eeg.shape)
        eeg = torch.as_tensor(eeg.copy(), dtype=torch.float32)
        return {'eeg':eeg[0] if self.split == 'train' else eeg,
                'age':torch.tensor(age,dtype=torch.float32),
                'label':torch.tensor(self.samples[idx]['class_label'],dtype=torch.long),
                'subject_id':self.samples[idx]['serial']}

def fit_statistics(dataset):
    """Train subjects only; stable parallel variance over full recordings."""
    count, mean, m2 = 0, None, None
    for idx in range(len(dataset)):
        x = dataset.load_recording(idx).astype(np.float64)
        n = x.shape[1]
        current = x.mean(axis=1)
        var_sum = ((x-current[:,None])**2).sum(axis=1)
        if mean is None:
            count, mean, m2 = n, current, var_sum
        else:
            delta = current-mean
            m2 += var_sum + delta**2 * count*n/(count+n)
            mean += delta*n/(count+n)
            count += n
    ages = np.array([x['age'] for x in dataset.samples])
    return {'signal_mean':mean.tolist(), 'signal_std':np.maximum(np.sqrt(m2/count),1e-8).tolist(),
            'age_mean':float(ages.mean()), 'age_std':max(float(ages.std()),1e-8)}
