from autocar3g import AI, Camera
from autocar3g.driving import Driving
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 카메라 시작
cam = Camera()
cam.start() 
 
# 자동차 제어
car = Driving()
throttle = 10

# 모델 로드
print("Loading Track_Model.h5...")
model = keras.models.load_model("/abr/coss11/repo/Track_Model.h5")
print("Model loaded successfully!")

# 모델 정보
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

try:
    while True:
        # 카메라에서 프레임 받기
        frame = cam.read()
        
        if frame is None:
            continue
            
        # 이미지 전처리 (학습 시와 동일하게)
        img = frame[120:270, :]  # 높이 150, 너비 400
        img = np.expand_dims(img, axis=0)  # 배치 차원 추가
        img = img.astype('float32')
        
        # 모델 예측
        pred = model.predict(img, verbose=0)
        x, y = pred[0]
        
        print(f"Prediction: x={x:.4f}, y={y:.4f}")
        
        # 스티어링 제어 (x값이 0.5이면 중앙)
        steer = (x - 0.5) * 2  # -1.0 ~ 1.0 범위
        
        # 자동차 제어
        car.throttle = throttle
        car.steering = steer
        
        time.sleep(0.05)  # 약 20fps
        
except KeyboardInterrupt:
    print("\nStopping...")
    car.throttle = 0
    car.steering = 0
    cam.stop()
    print("Done!")
