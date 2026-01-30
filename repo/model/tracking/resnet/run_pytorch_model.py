import torch
import torch.nn as nn
import cv2
import time
import numpy as np
from autocar3g.camera import Camera
from autocar3g.driving import Driving

import os

os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
os.environ['OMP_NUM_THREADS'] = '1'

# 1. PyTorch 모델 로드 (가상의 가공 클래스 또는 직접 로드)
# 만약 autocar3g에서 PyTorch 전용 클래스를 제공하지 않는다면 직접 모델을 불러와야 합니다.
device = torch.device("cpu")

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(x + out)
    
class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResBlock(32),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResBlock(64),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            ResBlock(128),
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),  # 회귀 출력 2개 (x,y)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.head(x)
        return x

model = TrackNet().to(device)
checkpoint = torch.load('repo/model/Track_Model_pytorch.pt', map_location=device)

if 'model_state' in checkpoint:
    # 딕셔너리 내부에 'model_state'라는 키로 가중치가 저장된 경우
    model.load_state_dict(checkpoint['model_state'])
    print("가중치 로드 완료!")
else:
    # 만약 키 이름이 다르다면 전체 구조를 출력해서 확인해봐야 합니다.
    model.load_state_dict(checkpoint)
model.eval()

cam = Camera()
cam.start()
car = Driving()

throttle = 10

while True:
    try:
        frame = cam.read()
        if frame is None: continue

        # 1. 전처리
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        # 2. 추론
        with torch.no_grad():
            output = model(input_tensor)
        
        # 3. 조향 적용 (x값 사용)
        target_x = output[0, 0].item()
        steer = (target_x - 0.5) * 2
        
        car.steering = steer
        car.throttle = throttle

    except KeyboardInterrupt:
        car.throttle = 0
        car.steering = 0
        break