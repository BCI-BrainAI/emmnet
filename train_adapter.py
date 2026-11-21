import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Focal Loss 정의 (EMMNet 논문 프로토콜: alpha=0.25, gamma=2.0)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

# 2. 클래스 기반 결합 Dataset
class MultimodalFeatureDataset(Dataset):
    def __init__(self, split='train', seed=42):
        np.random.seed(seed)
        eeg = np.load(f'data/eeg_features_{split}.npz')
        mri = np.load(f'data/mri_features_{split}.npz')
        
        eeg_feat, eeg_lbl = eeg['features'], eeg['labels']
        mri_feat, mri_lbl = mri['features'], mri['labels']
        
        # 클래스별(0, 1) 분리
        eeg_c0 = eeg_feat[eeg_lbl == 0]
        eeg_c1 = eeg_feat[eeg_lbl == 1]
        mri_c0 = mri_feat[mri_lbl == 0]
        mri_c1 = mri_lbl_1 = mri_feat[mri_lbl == 1]
        
        # MRI 개수에 맞춰 매칭 (Train 기준 155명 정상, 354명 비정상)
        n_c0 = len(mri_c0)
        n_c1 = len(mri_c1)
        
        # EEG 쪽에서 동일 클래스 샘플 매칭 (재현성 셔플)
        idx_eeg_c0 = np.random.choice(len(eeg_c0), n_c0, replace=False)
        idx_eeg_c1 = np.random.choice(len(eeg_c1), n_c1, replace=False)
        
        # 256D + 256D -> 512D Concat
        pair_c0 = np.concatenate([eeg_c0[idx_eeg_c0], mri_c0], axis=1)
        pair_c1 = np.concatenate([eeg_c1[idx_eeg_c1], mri_c1], axis=1)
        
        self.features = np.concatenate([pair_c0, pair_c1], axis=0).astype(np.float32)
        self.labels = np.concatenate([np.zeros(n_c0), np.ones(n_c1)], axis=0).astype(np.int64)
        
        # 전체 셔플
        perm = np.random.permutation(len(self.labels))
        self.features = torch.tensor(self.features[perm])
        self.labels = torch.tensor(self.labels[perm])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 3. 경량 Adapter 분류기 (512D -> 128D -> 2D)
class MultimodalAdapter(nn.Module):
    def __init__(self, in_features=512, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

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
    sens = recall_score(all_targets, all_preds, zero_division=0) # Abnormal Sensitivity
    prec = precision_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    return total_loss / len(loader.dataset), acc, sens, prec, f1

# 4. 학습 루프 실행
if __name__ == '__main__':
    train_ds = MultimodalFeatureDataset('train')
    val_ds = MultimodalFeatureDataset('val')
    test_ds = MultimodalFeatureDataset('test')
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = MultimodalAdapter(in_features=512, hidden_dim=128, num_classes=2)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    best_val_loss = float('inf')
    best_metrics = None
    
    print("=== Training Multimodal Adapter (512-D Concat) ===")
    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)
        
        val_loss, val_acc, val_sens, _, _ = evaluate(model, val_loader, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'adapter_best.pt')
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/50] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Sens: {val_sens*100:.2f}%")
            
    # 최적 가중치 로드 후 Test 평가
    model.load_state_dict(torch.load('adapter_best.pt'))
    test_loss, test_acc, test_sens, test_prec, test_f1 = evaluate(model, test_loader, criterion)
    print("\n================ FINAL TEST RESULT ================")
    print(f"Test Accuracy   : {test_acc*100:.2f}%")
    print(f"Test Sensitivity: {test_sens*100:.2f}% (Abnormal 검출률)")
    print(f"Test Precision  : {test_prec*100:.2f}%")
    print(f"Test F1-Score   : {test_f1*100:.2f}%")
    print("====================================================")
