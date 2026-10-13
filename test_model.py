import torch
from torch.utils.data import DataLoader
from datasets.caueeg_dataset import EMMNetEEGDataset
from models.resnet_1d import ResNet1D

print("--- 1. 데이터로더 준비 ---")
SPLIT_JSON_PATH = "local/datasets/caueeg-dataset/dementia.json"
EEG_DIR = "local/datasets/caueeg-dataset/signal/edf/"

# 훈련 데이터셋 및 데이터로더 생성 (배치 크기 2개)
dataset = EMMNetEEGDataset(split_csv_path=SPLIT_JSON_PATH, eeg_dir=EEG_DIR, split='train', seq_length=2000)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 첫 번째 배치 가져오기
batch = next(iter(loader))
eeg_tensor = batch['eeg'] # Shape: torch.Size([2, 21, 2000])
labels = batch['label']

print(f"인풋 EEG 텐서 Shape : {eeg_tensor.shape}")
print(f"라벨(Label)        : {labels}")

print("\n--- 2. ResNet1D 모델 생성 및 설정 ---")
# 1D ResNet 모델 인스턴스화
model = ResNet1D(
    block="basic",
    conv_layers=[2, 2, 2, 2],  # ResNet-18 구조 스타일
    in_channels=21,            # 뇌파 채널 수 (21개)
    out_dims=2,                # 분류할 클래스 수 (예: 정상/치매 등 이진 분류)
    seq_length=2000,           # 시퀀스 길이
    base_channels=32,          # 기본 채널 수
    use_age="no",              # 나이 정보 사용 여부 (우선 "no"로 설정)
    fc_stages=1,               # 최종 FC 레이어 단계 수
    activation="relu",
)

# 모델을 평가 모드로 전환
model.eval()

print("\n--- 3. 모델에 뇌파 텐서 통과시키기 (Forward Pass) ---")
with torch.no_grad():
    # ResNet1D 코드를 보면 forward(self, x, age) 형태로 되어 있어
    # use_age="no"여도 자리를 채워주기 위해 dummy_age를 함께 전달합니다.
    dummy_age = torch.zeros(eeg_tensor.size(0))
    output = model(eeg_tensor, dummy_age)

print(f"모델 출력(Output) Shape : {output.shape}")
print(f"모델이 뱉어낸 예측 점수  :\n{output}")

print("\n🎉 축하합니다! 데이터와 ResNet1D 모델이 성공적으로 연결되었습니다!")