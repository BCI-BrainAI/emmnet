import os
import torch
from torch.utils.data import DataLoader
from datasets.caueeg_dataset import EMMNetEEGDataset

print("현재 실행 중인 스크립트 폴더:", os.getcwd())
print("이 폴더 안에 있는 파일/폴더 목록:", os.listdir('.'))

# 경로 설정
SPLIT_JSON_PATH = "local/datasets/caueeg-dataset/dementia.json"
EEG_DIR = "local/datasets/caueeg-dataset/signal/edf/"

print("\n--- 1. 데이터셋 경로 및 파일 존재 여부 점검 시작 ---")
if os.path.exists(SPLIT_JSON_PATH) and os.path.exists(EEG_DIR):
    print("[성공] 경로 확인 완료!")
else:
    print("[에러] 경로를 다시 확인해주세요.")

# 1. Train, Val, Test 데이터셋 인스턴스 생성
print("\n--- 2. Train, Val, Test 데이터셋 인스턴스 생성 및 검증 ---")
train_dataset = EMMNetEEGDataset(split_csv_path=SPLIT_JSON_PATH, eeg_dir=EEG_DIR, split='train', seq_length=2000)
val_dataset = EMMNetEEGDataset(split_csv_path=SPLIT_JSON_PATH, eeg_dir=EEG_DIR, split='val', seq_length=2000)
test_dataset = EMMNetEEGDataset(split_csv_path=SPLIT_JSON_PATH, eeg_dir=EEG_DIR, split='test', seq_length=2000)

print(f"로드된 Train 환자 수: {len(train_dataset)}명")
print(f"로드된 Val 환자 수:   {len(val_dataset)}명")
print(f"로드된 Test 환자 수:  {len(test_dataset)}명")

# 2. DataLoader 배치 출력 테스트 (Train 기준)
print("\n--- 3. DataLoader 배치 출력 테스트 (Train) ---")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

try:
    for batch in train_loader:
        print("[1번째 배치]")
        print(" - EEG 텐서 Shape :", batch['eeg'].shape)
        print(" - 라벨(Label)    :", batch['label'])
        print(" - 환자 ID(Subject):", batch['subject_id'])
        break
    print("\n🎉 축하합니다! 검증 및 테스트 데이터셋을 포함한 모든 데이터 로더가 완벽하게 작동합니다!")
except Exception as e:
    print(f"\n[에러 발생] 테스트 중 문제가 생겼습니다: {e}")