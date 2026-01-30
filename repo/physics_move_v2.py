import time
from autocar3g.driving import Driving

car = Driving()

THROTTLE_STRAIGHT = 8
THROTTLE_TURN = 4

STEER_CENTER = 0.0

T_STRAIGHT = 1.3

def straight():
    car.steering = STEER_CENTER
    time.sleep(0.1)

    car.throttle = THROTTLE_STRAIGHT
    time.sleep(T_STRAIGHT)

    car.throttle = 0
    time.sleep(0.2)

def right_turn_ellipse():
    # 조향 증가
    for steer in [-0.3, -0.6, -0.8]:
        car.steering = steer
        time.sleep(0.15)

    # 곡선 주행
    car.throttle = THROTTLE_TURN
    time.sleep(1.4)

    # 조향 복귀
    for steer in [-0.6, -0.3, 0.0]:
        car.steering = steer
        time.sleep(0.15)

    car.throttle = 0
    time.sleep(0.2)

try:
    print("▶ RIGHT TRACK START")

    straight()
    right_turn_ellipse()
    right_turn_ellipse()

    straight()
    right_turn_ellipse()
    right_turn_ellipse()

    print("✅ DONE")

finally:
    car.throttle = 0
    car.steering = 0
    print("🛑 STOP")
