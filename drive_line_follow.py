import cv2
import numpy as np
import time
import sys
from enum import Enum

from autocar3g.driving import Driving
from autocar3g.camera import Camera

# =========================================================
# 시작 시 L / R 선택
# =========================================================
if len(sys.argv) < 2:
    print("사용법: python3 drive_line_follow.py [L/R]")
    sys.exit(1)

arg = sys.argv[1].upper()
if arg == "L":
    TRACK_SIDE = "LEFT"
elif arg == "R":
    TRACK_SIDE = "RIGHT"
else:
    raise ValueError("L 또는 R만 입력 가능")

print("▶ TRACK SIDE:", TRACK_SIDE)

# =========================================================
# 파라미터
# =========================================================
BASE_THROTTLE = 25
ROI_RATIO = 0.5
OUTWARD_OFFSET = 20
MAX_JUMP = 50

KP = 0.004
KD = 0.001

LOST_THRESHOLD = 5
RECOVER_FRAMES = 5

# =========================================================
# PID
# =========================================================
class PID:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev = 0

    def step(self, e):
        d = e - self.prev
        self.prev = e
        return self.kp * e + self.kd * d

# =========================================================
# FSM
# =========================================================
class State(Enum):
    NORMAL = 0
    LOST = 1
    SEARCH = 2
    RECOVER = 3

# =========================================================
# Kalman
# =========================================================
def make_kalman():
    k = cv2.KalmanFilter(2, 1)
    k.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
    k.measurementMatrix = np.array([[1, 0]], np.float32)
    k.processNoiseCov = np.array([[1e-2, 0], [0, 1e-3]], np.float32)
    k.measurementNoiseCov = np.array([[1e-1]], np.float32)
    k.errorCovPost = np.eye(2, dtype=np.float32)
    return k

kal_L = make_kalman()
kal_R = make_kalman()

# =========================================================
# 유틸
# =========================================================
def select_best(contours, side, mid_x):
    cand = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        area = cv2.contourArea(c)
        if area < 200:
            continue
        if side == "LEFT" and cx < mid_x:
            cand.append((c, cx))
        elif side == "RIGHT" and cx >= mid_x:
            cand.append((c, cx))
    if not cand:
        return None
    return max(cand, key=lambda x: x[1]) if side == "LEFT" else min(cand, key=lambda x: x[1])

# =========================================================
# 초기화
# =========================================================
car = Driving()
cam = Camera()
cam.start()
time.sleep(1)

pid = PID(KP, KD)
state = State.NORMAL
lost_cnt = 0
recover_cnt = 0

prev_L = None
prev_R = None

# =========================================================
# 메인 루프
# =========================================================
while True:
    frame = cam.read()
    h, w, _ = frame.shape

    roi = frame[int(h * (1 - ROI_RATIO)):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mid_x = w // 2

    # -------------------------
    # 좌 / 우 라인 후보
    # -------------------------
    best_L = select_best(contours, "LEFT", mid_x)
    best_R = select_best(contours, "RIGHT", mid_x)

    # -------------------------
    # Kalman 예측
    # -------------------------
    cx_L = int(kal_L.predict()[0])
    cx_R = int(kal_R.predict()[0])

    if best_L:
        _, raw = best_L
        if prev_L is None or abs(raw - prev_L) < MAX_JUMP:
            kal_L.correct(np.array([[np.float32(raw)]]))
            cx_L = int(kal_L.predict()[0])
            prev_L = raw

    if best_R:
        _, raw = best_R
        if prev_R is None or abs(raw - prev_R) < MAX_JUMP:
            kal_R.correct(np.array([[np.float32(raw)]]))
            cx_R = int(kal_R.predict()[0])
            prev_R = raw

    # -------------------------
    # 선택한 쪽 기준
    # -------------------------
    if TRACK_SIDE == "LEFT":
        cx_use = cx_L - OUTWARD_OFFSET
        search_steer = -0.6
    else:
        cx_use = cx_R + OUTWARD_OFFSET
        search_steer = 0.6

    target = w // 2
    error = target - cx_use

    # -------------------------
    # FSM 상태 전이
    # -------------------------
    if best_L or best_R:
        lost_cnt = 0
        if state in (State.LOST, State.SEARCH):
            state = State.RECOVER
            recover_cnt = 0
    else:
        lost_cnt += 1
        if lost_cnt > LOST_THRESHOLD:
            state = State.SEARCH

    # -------------------------
    # FSM 제어
    # -------------------------
    if state == State.NORMAL:
        steer = pid.step(error)
        car.steering = float(np.clip(steer, -1.0, 1.0))
        car.throttle = BASE_THROTTLE

    elif state == State.RECOVER:
        recover_cnt += 1
        steer = pid.step(error) * 0.5
        car.steering = float(np.clip(steer, -0.5, 0.5))
        car.throttle = int(BASE_THROTTLE * 0.7)
        if recover_cnt >= RECOVER_FRAMES:
            state = State.NORMAL

    elif state == State.SEARCH:
        car.steering = search_steer
        car.throttle = int(BASE_THROTTLE * 0.5)

    time.sleep(0.02)

# =========================================================
# 종료 (비상)
# =========================================================
car.throttle = 0
car.steering = 0
cam.stop()
