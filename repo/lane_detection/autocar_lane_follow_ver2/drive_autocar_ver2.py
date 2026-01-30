import sys
import time
import os
import math
import cv2

from autocar3g.camera import Camera
from autocar3g.driving import Driving

from lane_detector_ver2 import LaneDetectorV2
from pid_controller_ver2 import PIDControllerV2
from human_override import HumanOverride


if len(sys.argv) < 2:
    raise RuntimeError("Usage: python3 drive_autocar_ver2.py L or R")

TURN_MODE = sys.argv[1].upper()
if TURN_MODE not in ["L", "R"]:
    raise RuntimeError("TURN_MODE must be L or R")


SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_INTERVAL = 3
JPG_QUALITY = 90


cam = Camera()
cam.start()

car = Driving()

frame = cam.read()
H, W, _ = frame.shape

lane_detector = LaneDetectorV2(W)
pid = PIDControllerV2(kp=0.8, kd=0.04)

human = HumanOverride(step=0.05, max_bias=0.4)

BASE_THROTTLE = 15
MAX_STEER = 1.0

K_LAT = 1.0
K_HEAD = 0.7

frame_idx = 0


try:
    while True:
        frame = cam.read()
        if frame is None:
            continue

        if frame_idx % SAVE_INTERVAL == 0:
            cv2.imwrite(
                f"frame_{frame_idx:06d}.jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]
            )

        lane_near, lane_far, vehicle_center = lane_detector.process(frame)
        if lane_near is None or lane_far is None:
            frame_idx += 1
            continue

        lateral_error = (vehicle_center - lane_near) / (W / 2)
        heading_error = (lane_far - lane_near) / (W / 2)

        pid_term = pid.compute(lateral_error)

        raw = K_LAT * pid_term + K_HEAD * heading_error
        auto_steer = MAX_STEER * math.tanh(raw)

        human_bias = human.get()
        steering = auto_steer + human_bias

        if TURN_MODE == "L":
            steering = min(0.0, steering)
        elif TURN_MODE == "R":
            steering = max(0.0, steering)

        steering = max(-MAX_STEER, min(MAX_STEER, steering))

        steer_abs = abs(steering)
        if steer_abs > 0.6:
            throttle = 10
        elif steer_abs > 0.4:
            throttle = 12
        else:
            throttle = BASE_THROTTLE

        car.steering = steering
        car.throttle = throttle

        frame_idx += 1
        time.sleep(0.03)

except KeyboardInterrupt:
    pass

finally:
    human.stop()
    car.throttle = 0
    car.steering = 0
    cam.stop()
