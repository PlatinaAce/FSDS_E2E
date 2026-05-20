import os
import glob
import random
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import math

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
DATA_ROOT = '/home/ace/fsds_dataset/training_data' # training_data 루트 폴더

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
MAX_SPEED = 10.0  # 목표 속도 상한 (m/s), 모델 출력 정규화에 사용

# 차량 파라미터 (1/r 변환용)
WHEELBASE = 1.55          # 축거 (m)
MAX_STEER_ANGLE = 0.5    # 최대 조향각 (rad, ~28.6°)
MAX_INV_RADIUS = math.tan(MAX_STEER_ANGLE) / WHEELBASE  # 최대 역회전반경

# Augmentation 파라미터
SHIFT_RANGE = 80         # 좌우 이동 단위 (px, 원본 640px 기준)
STEERING_CORRECTION = 0.15  # 이동 시 보정할 1/r 값
ROTATION_RANGE = 10      # 회전 증강 최대 각도 (°)
ROTATION_CORRECTION = 0.01  # 회전 1°당 1/r 보정 계수
VSHIFT_RANGE = 10        # 상하 이동 최대 픽셀

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# ==========================================
# 2. 커스텀 데이터셋 클래스 (CSV와 이미지 로드)
# ==========================================
class FSDSDataset(Dataset):
    def __init__(self, csv_file, cam1_dir, cam2_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.cam1_dir = cam1_dir
        self.cam2_dir = cam2_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 불러오기 (cam1 + cam2)
        cam1_name = os.path.join(self.cam1_dir, self.data.iloc[idx, 0])
        cam2_name = os.path.join(self.cam2_dir, self.data.iloc[idx, 1])
        cam1_img = Image.open(cam1_name).convert('RGB')
        cam2_img = Image.open(cam2_name).convert('RGB')

        # 이미지 Crop (Top 0.52, Bottom 0.7)
        cam1_np = np.asarray(cam1_img)
        cam2_np = np.asarray(cam2_img)
        h, w = cam1_np.shape[:2]
        top = int(h * 0.52)
        bottom = int(h * 0.70)
        
        cam1_np = cam1_np[top:bottom, :]
        cam2_np = cam2_np[top:bottom, :]
        
        cam1_img = Image.fromarray(cam1_np)
        cam2_img = Image.fromarray(cam2_np)

        # 정답지(InverseRadius, Speed) 불러오기
        inv_radius = self.data.iloc[idx, 2]
        speed = self.data.iloc[idx, 3]

        # ---- Augmentation 1: 좌우 랜덤 이동 + 1/r 보정 ----
        if random.random() < 0.6:
            shift_px = random.randint(-SHIFT_RANGE, SHIFT_RANGE)
            cam1_np = np.asarray(cam1_img)
            cam2_np = np.asarray(cam2_img)
            M = np.float32([[1, 0, shift_px], [0, 1, 0]])
            cam1_np = cv2.warpAffine(cam1_np, M, (cam1_np.shape[1], cam1_np.shape[0]),
                                      borderMode=cv2.BORDER_REPLICATE)
            cam2_np = cv2.warpAffine(cam2_np, M, (cam2_np.shape[1], cam2_np.shape[0]),
                                      borderMode=cv2.BORDER_REPLICATE)
            cam1_img = Image.fromarray(cam1_np)
            cam2_img = Image.fromarray(cam2_np)
            inv_radius -= (shift_px / SHIFT_RANGE) * STEERING_CORRECTION

        # ---- Augmentation 2: 회전 (Yaw) + 1/r 보정 ----
        if random.random() < 0.4:
            angle_deg = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            cam1_np = np.asarray(cam1_img)
            cam2_np = np.asarray(cam2_img)
            h, w = cam1_np.shape[:2]
            R = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
            cam1_np = cv2.warpAffine(cam1_np, R, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cam2_np = cv2.warpAffine(cam2_np, R, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cam1_img = Image.fromarray(cam1_np)
            cam2_img = Image.fromarray(cam2_np)
            inv_radius -= angle_deg * ROTATION_CORRECTION

        # ---- Augmentation 3: 상하 이동 (Pitch) ----
        if random.random() < 0.3:
            vshift = random.randint(-VSHIFT_RANGE, VSHIFT_RANGE)
            cam1_np = np.asarray(cam1_img)
            cam2_np = np.asarray(cam2_img)
            M_v = np.float32([[1, 0, 0], [0, 1, vshift]])
            cam1_np = cv2.warpAffine(cam1_np, M_v, (cam1_np.shape[1], cam1_np.shape[0]),
                                      borderMode=cv2.BORDER_REPLICATE)
            cam2_np = cv2.warpAffine(cam2_np, M_v, (cam2_np.shape[1], cam2_np.shape[0]),
                                      borderMode=cv2.BORDER_REPLICATE)
            cam1_img = Image.fromarray(cam1_np)
            cam2_img = Image.fromarray(cam2_np)

        # ---- Augmentation 4: 좌우 반전 ----
        if random.random() < 0.5:
            cam1_img = cam1_img.transpose(Image.FLIP_LEFT_RIGHT)
            cam2_img = cam2_img.transpose(Image.FLIP_LEFT_RIGHT)
            inv_radius = -inv_radius

        # ---- Augmentation 5: 랜덤 그림자 ----
        if random.random() < 0.3:
            cam1_img = self._add_random_shadow(cam1_img)
            cam2_img = self._add_random_shadow(cam2_img)

        # ---- Augmentation 6: 밝기 및 색상 지터 (Hue shift 추가) ----
        if random.random() < 0.4:
            # Hue를 변화시켜 노란색과 파란색이 특정 색상값에 과적합되지 않도록 유도
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)
            cam1_img = jitter(cam1_img)
            cam2_img = jitter(cam2_img)

        # ---- Augmentation 7: 안쪽 영역 가림 (Edge Cutout) ----
        # 조향이 어느 정도 꺾여있을 때만 적용 (직진할 때는 양쪽을 다 보게 뒴)
        # 확률 40% 적용
        if random.random() < 0.4 and abs(inv_radius) > 0.05:
            cam1_img = self._add_edge_mask(cam1_img, inv_radius)
            cam2_img = self._add_edge_mask(cam2_img, inv_radius)

        inv_radius = max(-1.0, min(1.0, inv_radius))
        speed_norm = min(speed / MAX_SPEED, 1.0)
        targets = torch.tensor([inv_radius, speed_norm], dtype=torch.float32)

        # RGB → YUV 변환 후 텐서화
        cam1_yuv = cv2.cvtColor(np.asarray(cam1_img), cv2.COLOR_RGB2YUV)
        cam2_yuv = cv2.cvtColor(np.asarray(cam2_img), cv2.COLOR_RGB2YUV)
        cam1_pil = Image.fromarray(cam1_yuv)
        cam2_pil = Image.fromarray(cam2_yuv)

        if self.transform:
            cam1_tensor = self.transform(cam1_pil)
            cam2_tensor = self.transform(cam2_pil)
        else:
            to_t = transforms.ToTensor()
            cam1_tensor = to_t(cam1_pil)
            cam2_tensor = to_t(cam2_pil)

        # 6채널 입력 (cam1 YUV + cam2 YUV)
        image = torch.cat([cam1_tensor, cam2_tensor], dim=0)

        return image, targets

    @staticmethod
    def _add_random_shadow(img):
        """이미지에 랜덤 다각형 그림자를 추가"""
        img_np = np.asarray(img).copy()
        h, w = img_np.shape[:2]
        # 랜덤 사다리꼴 그림자 영역 생성
        x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
        pts = np.array([[x1, 0], [x2, 0], [x2 + random.randint(-60, 60), h],
                        [x1 + random.randint(-60, 60), h]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        # 그림자 적용 (밝기 30~70% 감소)
        shadow_factor = random.uniform(0.3, 0.7)
        img_np[mask == 255] = (img_np[mask == 255] * shadow_factor).astype(np.uint8)
        return Image.fromarray(img_np)

    @staticmethod
    def _add_edge_mask(img, steer_cmd):
        """조향 방향에 따라 모델이 의존하는 안쪽(가장자리) 콘 영역을 검은색으로 가림(Cutout).
        강제로 바깥쪽 콘이나 넓은 시야를 보도록 유도함."""
        img_np = np.asarray(img).copy()
        h, w = img_np.shape[:2]
        
        # 박스 크기 비율 (화면 전체의 약 20~30% 너비, 높이는 전체 상하단)
        box_w = int(w * random.uniform(0.2, 0.3))
        
        # 좌회전(steer > 0)일 때는 왼쪽 시야를 의도적으로 가림 (파란색 콘이 있는 안쪽 영역)
        # 우회전(steer < 0)일 때는 오른쪽 시야를 의도적으로 가림 (노란색 콘이 있는 안쪽 영역)
        # 직진일 때는 적용 안 하거나 랜덤 적용할 수 있지만, 조향이 클 때만 적용
        if steer_cmd > 0.05:
            # 왼쪽 가리기
            img_np[:, 0:box_w] = 0
        elif steer_cmd < -0.05:
            # 오른쪽 가리기
            img_np[:, w - box_w:w] = 0
            
        return Image.fromarray(img_np)

# ==========================================
# 3. 모델 정의 (NVIDIA PilotNet 수정본)
# ==========================================
class PilotNet(nn.Module):
    def __init__(self, in_channels=3):
        super(PilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1152, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),  # Steering, Speed(정규화)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # Steering: tanh → [-1, 1]
        steering = torch.tanh(x[:, 0:1])
        # Speed(정규화): sigmoid → [0, 1]
        speed_norm = torch.sigmoid(x[:, 1:2])
        return torch.cat([steering, speed_norm], dim=1)

# ==========================================
# 4. 학습 (Training) 실행
# ==========================================
def main():
    # 이미지 전처리 (PilotNet 입력 규격인 200x66으로 리사이즈, YUV 정규화)
    transform = transforms.Compose([
        transforms.Resize((66, 200)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터 로더 준비 (training_data 하위 모든 dataset.csv를 합침)
    csv_files = sorted(glob.glob(os.path.join(DATA_ROOT, '**/dataset.csv'), recursive=True))
    if not csv_files:
        print(f"[오류] {DATA_ROOT} 안에 dataset.csv 파일이 없습니다!")
        return

    # 1. Scene-based Split (방지: Data Leakage)
    # 이미지 프레임 단위가 아닌, 주행 시나리오(폴더/CSV) 단위로 섞어서 분할합니다.
    random.shuffle(csv_files)
    num_val_scenes = max(1, int(len(csv_files) * 0.15)) # 최소 1개의 시퀀스는 Validation으로 할당
    
    val_csv_files = csv_files[:num_val_scenes]
    train_csv_files = csv_files[num_val_scenes:]

    print(f"총 {len(csv_files)}개의 주행 시퀀스 중 Train: {len(train_csv_files)}개, Validation: {len(val_csv_files)}개로 분할합니다.")

    train_datasets = []
    for csv_file in train_csv_files:
        bag_dir = os.path.dirname(csv_file)
        cam1_dir = os.path.join(bag_dir, 'images', 'cam1')
        cam2_dir = os.path.join(bag_dir, 'images', 'cam2')
        train_datasets.append(FSDSDataset(csv_file=csv_file, cam1_dir=cam1_dir, cam2_dir=cam2_dir, transform=transform))

    val_datasets = []
    for csv_file in val_csv_files:
        bag_dir = os.path.dirname(csv_file)
        cam1_dir = os.path.join(bag_dir, 'images', 'cam1')
        cam2_dir = os.path.join(bag_dir, 'images', 'cam2')
        val_datasets.append(FSDSDataset(csv_file=csv_file, cam1_dir=cam1_dir, cam2_dir=cam2_dir, transform=transform))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 모델, 손실 함수, 최적화 기법 설정
    model = PilotNet(in_channels=6).to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"학습 데이터 {train_size}개, 검증 데이터 {val_size}개로 학습을 시작합니다...")

    best_val_loss = float('inf')
    best_model_path = f'/home/ace/fsds_dataset/pretrained_model/pilotnet_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            # 1. Forward Pass (예측)
            optimizer.zero_grad()
            outputs = model(images)
            
            # 2. Loss 계산 (조향과 속도 가중치 분리)
            loss_steer = mse_loss(outputs[:, 0], targets[:, 0])
            loss_speed = mse_loss(outputs[:, 1], targets[:, 1])
            loss = (0.8 * loss_steer) + (0.2 * loss_speed)
            
            # 3. Backward Pass (역전파 및 가중치 업데이트)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 검증(Validation) 페이즈
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                
                v_loss_steer = mse_loss(outputs[:, 0], targets[:, 0])
                v_loss_speed = mse_loss(outputs[:, 1], targets[:, 1])
                v_loss = (0.8 * v_loss_steer) + (0.2 * v_loss_speed)
                val_loss += v_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Best Model 저장됨! (Val Loss: {best_val_loss:.4f})")

    print(f"\n학습 완료! 최종 Best 모델 경로: {best_model_path}")

if __name__ == '__main__':
    main()