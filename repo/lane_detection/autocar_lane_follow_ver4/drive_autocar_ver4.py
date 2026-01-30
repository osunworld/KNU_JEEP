import sys
import time
import os
import math
import cv2

from autocar3g.camera import Camera
from autocar3g.driving import Driving

from lane_detector_ver4 import LaneDetectorV4
from pid_controller_ver4 import PIDControllerV4
from human_override import HumanOverride


# =========================
# Arg
# =========================
if len(sys.argv) < 2:
    raise RuntimeError("Usage: python3 drive_autocar_ver4.py L or R")

TURN_MODE = sys.argv[1].upper()
if TURN_MODE not in ["L", "R"]:
    raise RuntimeError("TURN_MODE must be L or R")


# =========================
# Params
# =========================
BASE_THROTTLE = 15
TURN_BIAS = 0.25
HUMAN_GAIN = 1.5

SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Init
# =========================
cam = Camera()
cam.start()

car = Driving()

frame = cam.read()
H, W, _ = frame.shape

lane_detector = LaneDetectorV4(W)
pid = PIDControllerV4(kp=0.9, ki=0.02, kd=0.05)

human = HumanOverride(step=0.06, max_bias=0.6)


# =========================
# Loop
# =========================
try:
    while True:
        frame = cam.read()
        if frame is None:
            continue

        lane_near, lane_far, vehicle_center, curvature = lane_detector.process(frame)
        if lane_near is None:
            continue

        # =========================
        # Error
        # =========================
        lateral_error = (vehicle_center - lane_near) / (W / 2)

        dx = lane_far - lane_near
        dy = lane_detector.dy
        heading_error = dx / max(dy, 1)

        # =========================
        # Gain scheduling
        # =========================
        if curvature > W * 0.12:
            K_LAT, K_HEAD = 1.3, 1.6
            steer_scale = 1.4
            max_steer = 1.4
            throttle = 7
        elif curvature > W * 0.08:
            K_LAT, K_HEAD = 1.2, 1.4
            steer_scale = 1.25
            max_steer = 1.3
            throttle = 9
        elif curvature > W * 0.05:
            K_LAT, K_HEAD = 1.1, 1.1
            steer_scale = 1.1
            max_steer = 1.15
            throttle = 12
        else:
            K_LAT, K_HEAD = 1.0, 0.7
            steer_scale = 1.0
            max_steer = 1.0
            throttle = BASE_THROTTLE

        # =========================
        # Steering
        # =========================
        pid_term = pid.compute(lateral_error)
        raw = K_LAT * pid_term + K_HEAD * heading_error

        auto_steer = steer_scale * math.tanh(raw)

        # TURN prior (사람 없을 때만)
        human_bias = human.get()
        if abs(human_bias) < 0.05:
            if TURN_MODE == "L":
                auto_steer -= TURN_BIAS
            elif TURN_MODE == "R":
                auto_steer += TURN_BIAS

        # 사람 개입 강화
        steering = auto_steer + HUMAN_GAIN * human_bias

        # clamp (동적 조향각)
        steering = max(-max_steer, min(max_steer, steering))

        # =========================
        # Apply
        # =========================
        car.steering = steering
        car.throttle = throttle

        time.sleep(0.03)

except KeyboardInterrupt:
    pass

finally:
    human.stop()
    car.throttle = 0
    car.steering = 0
    cam.stop()
