import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.caueeg_dataset import EMMNetEEGDataset
from models.resnet_1d import ResNet1D

print("--- 1. 데이터로더 및 모델 준비 ---")
SPLIT_JSON_PATH = "local/datasets/caueeg-dataset/dementia.json"
EEG_DIR = "local/datasets/caueeg-dataset/signal/edf/"

dataset = EMMNetEEGDataset(split_csv_path=SPLIT_JSON_PATH, eeg_dir=EEG_DIR, split='train', seq_length=2000)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델 생성 (out_dims를 3으로 변경)
model = ResNet1D(
    block="basic",
    conv_layers=[2, 2, 2, 2],
    in_channels=21,
    out_dims=3,             # 👈 2에서 3으로 수정! (0, 1, 2 라벨을 수용하기 위함)
    seq_length=2000,
    base_channels=32,
    use_age="no",
    fc_stages=1,
)

# 모델을 학습(Train) 모드로 전환
model.train()

print("\n--- 2. 손실 함수(Loss Function) 및 옵티마이저(Optimizer) 설정 ---")
# 분류 문제이므로 CrossEntropyLoss 사용
criterion = nn.CrossEntropyLoss()

# 모델의 파라미터를 옵티마이저(Adam)에 전달
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- 3. 1개 배치에 대한 학습 루프(Training Step) 실행 ---")
# 배치 하나 가져오기
batch = next(iter(loader))
eeg_tensor = batch['eeg']
labels = batch['label'] # 정답 라벨 (예: 0 또는 1 등)

print(f"입력 뇌파 Shape : {eeg_tensor.shape}")
print(f"실제 정답 라벨   : {labels}")

# 1단계: 기존에 쌓였던 기울기(Gradient) 초기화
optimizer.zero_grad()

# 2단계: Forward Pass (모델 예측)
dummy_age = torch.zeros(eeg_tensor.size(0))
outputs = model(eeg_tensor, dummy_age)
print(f"모델 예측 결과   : {outputs}")

# 3단계: Loss(오차) 계산
loss = criterion(outputs, labels)
print(f"계산된 Loss 값   : {loss.item():.4f}")

# 4단계: Backward Pass (기울기 계산)
loss.backward()

# 5단계: Optimizer Step (가중치 업데이트)
optimizer.step()

print("\n 축하합니다! 손실 계산부터 가중치 업데이트까지 학습 루프가 정상적으로 작동했습니다!")