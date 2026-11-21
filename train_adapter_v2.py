import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Feature 스케일 정규화 함수 (L2 Normalize)
def normalize_features(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm

# 2. 전수 활용 및 가중치 균형 Dataset
class BalancedMultimodalDataset(Dataset):
    def __init__(self, split='train', seed=42):
        np.random.seed(seed)
        eeg = np.load(f'data/eeg_features_{split}.npz')
        mri = np.load(f'data/mri_features_{split}.npz')
        
        # L2 정규화를 거쳐 두 모달리티의 스케일을 일치시킴
        eeg_feat = normalize_features(eeg['features'])
        mri_feat = normalize_features(mri['features'])
        eeg_lbl, mri_lbl = eeg['labels'], mri['labels']
        
        # 클래스 분리
        eeg_0, eeg_1 = eeg_feat[eeg_lbl == 0], eeg_feat[eeg_lbl == 1]
        mri_0, mri_1 = mri_feat[mri_lbl == 0], mri_feat[mri_lbl == 1]
        
        # Train일 경우 EEG 전체를 활용하기 위해 MRI를 반복 순환(oversampling)하여 매칭
        if split == 'train':
            # EEG 샘플 수에 맞춰 짝 생성 (버려지는 EEG 데이터 제거)
            mri_0_rep = mri_0[np.random.choice(len(mri_0), len(eeg_0), replace=True)]
            mri_1_rep = mri_1[np.random.choice(len(mri_1), len(eeg_1), replace=True)]
            
            pair_0 = np.concatenate([eeg_0, mri_0_rep], axis=1)
            pair_1 = np.concatenate([eeg_1, mri_1_rep], axis=1)
            
            self.features = np.concatenate([pair_0, pair_1], axis=0).astype(np.float32)
            self.labels = np.concatenate([np.zeros(len(eeg_0)), np.ones(len(eeg_1))], axis=0).astype(np.int64)
        else:
            # Val/Test는 평가 기준(MRI 개수)에 맞춤
            n_0, n_1 = len(mri_0), len(mri_1)
            idx_eeg_0 = np.random.choice(len(eeg_0), n_0, replace=False)
            idx_eeg_1 = np.random.choice(len(eeg_1), n_1, replace=False)
            
            pair_0 = np.concatenate([eeg_0[idx_eeg_0], mri_0], axis=1)
            pair_1 = np.concatenate([eeg_1[idx_eeg_1], mri_1], axis=1)
            
            self.features = np.concatenate([pair_0, pair_1], axis=0).astype(np.float32)
            self.labels = np.concatenate([np.zeros(n_0), np.ones(n_1)], axis=0).astype(np.int64)

        perm = np.random.permutation(len(self.labels))
        self.features = torch.tensor(self.features[perm])
        self.labels = torch.tensor(self.labels[perm])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 3. 개선된 어댑터 모델 (Residual Skip-Connection 적용)
class ResidualAdapter(nn.Module):
    def __init__(self, in_features=512, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.dropout(self.relu(self.bn1(self.fc1(x))))
        h = h + self.dropout(self.relu(self.bn2(self.fc2(h)))) # Residual Connection
        return self.out(h)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    sens = recall_score(all_targets, all_preds, zero_division=0)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    return total_loss / len(loader.dataset), acc, sens, prec, f1

if __name__ == '__main__':
    train_ds = BalancedMultimodalDataset('train')
    val_ds = BalancedMultimodalDataset('val')
    test_ds = BalancedMultimodalDataset('test')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = ResidualAdapter(in_features=512, hidden_dim=128, num_classes=2)
    # 질환군(Abnormal) 판별력을 높이도록 클래스 가중치(Class Weight) 부여
    class_weights = torch.tensor([1.5, 1.0]).float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    
    best_val_f1 = 0.0
    
    print("=== Training Enhanced Multimodal Adapter (v2) ===")
    for epoch in range(1, 61):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        val_loss, val_acc, val_sens, val_prec, val_f1 = evaluate(model, val_loader, criterion)
        
        # F1-score 기준으로 최적 체크포인트 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'adapter_v2_best.pt')
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/60] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Sens: {val_sens*100:.2f}% | Val F1: {val_f1*100:.2f}%")
            
    model.load_state_dict(torch.load('adapter_v2_best.pt'))
    test_loss, test_acc, test_sens, test_prec, test_f1 = evaluate(model, test_loader, criterion)
    print("\n================ FINAL TEST RESULT (v2) ================")
    print(f"Test Accuracy   : {test_acc*100:.2f}%")
    print(f"Test Sensitivity: {test_sens*100:.2f}% (Abnormal 검출률)")
    print(f"Test Precision  : {test_prec*100:.2f}%")
    print(f"Test F1-Score   : {test_f1*100:.2f}%")
    print("=========================================================")
