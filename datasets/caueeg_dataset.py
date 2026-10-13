import os
import json
import torch
from torch.utils.data import Dataset
import mne
import numpy as np

class EMMNetEEGDataset(Dataset):
    def __init__(self, split_csv_path, eeg_dir, split='train', seq_length=2000, transform=None):
        """
        CAUEEG 공식 dementia.json 구조에 완벽히 대응하는 데이터셋 클래스
        """
        self.eeg_dir = eeg_dir
        self.split = split.lower()
        self.seq_length = seq_length
        self.transform = transform
        
        if not os.path.exists(split_csv_path):
            raise FileNotFoundError(f"Metadata JSON file not found at: {split_csv_path}")
            
        with open(split_csv_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        # 1. split 이름에 맞춰 올바른 키 매핑 ('train' -> 'train_split', 'val' 등)
        target_key = None
        for k in self.metadata.keys():
            if self.split in k.lower():
                target_key = k
                break
                
        if not target_key and self.split == 'val':
            # 'val'로 요청했을 때 'validation_split' 키 매핑
            for k in self.metadata.keys():
                if 'validation' in k.lower():
                    target_key = k
                    break

        self.samples_meta = self.metadata.get(target_key, []) if target_key else []
        print(f"[DEBUG] Matched key '{target_key}' with {len(self.samples_meta)} items.")

        # 2. 실제 파일 존재 여부 검증 및 샘플 구성
        self.samples = []
        missing_count = 0
        
        for item in self.samples_meta:
            subject_id = item.get('serial')
            label = item.get('class_label')
            
            if subject_id is None or label is None:
                continue
                
            # EDF 파일 경로 구성 (signal/edf 폴더 안의 5자리 번호.edf)
            eeg_filename = f"{int(subject_id):05d}.edf" if str(subject_id).isdigit() else f"{subject_id}.edf"
            eeg_path = os.path.join(eeg_dir, eeg_filename)
            
            if os.path.exists(eeg_path):
                self.samples.append({
                    'subject_id': subject_id,
                    'eeg_path': eeg_path,
                    'label': label
                })
            else:
                missing_count += 1
                
        if missing_count > 0:
            print(f"[DEBUG] Warning: {missing_count} EDF files were not found in {eeg_dir}")
                
        print(f"[EEG {split.upper()} Dataset] Successfully loaded {len(self.samples)} subjects.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # MNE를 이용해 EDF 파일 읽기
        raw = mne.io.read_raw_edf(sample['eeg_path'], preload=True, verbose=False)
        data, times = raw[:, :]  # shape: (channels, total_time_steps)
        
        # 시퀀스 길이 맞추기
        if data.shape[1] >= self.seq_length:
            eeg_data = data[:, :self.seq_length]
        else:
            pad_size = self.seq_length - data.shape[1]
            eeg_data = np.pad(data, ((0, 0), (0, pad_size)), 'constant')
            
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)
            
        return {
            'eeg': eeg_tensor,
            'label': label_tensor,
            'subject_id': sample['subject_id']
        }