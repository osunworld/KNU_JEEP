# drive_autocar.py
import time
from autocar3g.camera import Camera
from autocar3g.driving import Driving

from lane_detector import LaneDetector
from pid_controller import PIDController

# =========================
# Init
# =========================
cam = Camera()
cam.start()

car = Driving()

frame = cam.read()
H, W, _ = frame.shape

lane_detector = LaneDetector(W)
pid = PIDController(kp=0.8, kd=0.15, max_steer=0.7)

BASE_THROTTLE = 18

# =========================
# Drive Loop
# =========================
try:
    while True:
        frame = cam.read()
        if frame is None:
            continue

        lane_center, vehicle_center = lane_detector.process(frame)
        if lane_center is None:
            continue

        error = (vehicle_center - lane_center) / (W / 2)
        steering = pid.compute(error)

        car.steering = steering
        car.throttle = BASE_THROTTLE

        time.sleep(0.03)

except KeyboardInterrupt:
    print("⛔ STOP")

finally:
    car.throttle = 0
    car.steering = 0
    cam.stop()
