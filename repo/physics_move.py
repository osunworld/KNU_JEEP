import time
from autocar3g.driving import Driving

car = Driving()

# ===== 파라미터 =====
THROTTLE_STRAIGHT = 8
THROTTLE_TURN = 3

STEER_RIGHT = -1
STEER_CENTER = 0

T_STRAIGHT = 1.3     # 약 40cm
T_TURN_90 = 2.0      # 90도 회전

def straight():
    car.steering = STEER_CENTER
    time.sleep(0.1)

    car.throttle = THROTTLE_STRAIGHT
    time.sleep(T_STRAIGHT)

    car.throttle = 0
    time.sleep(0.1)

def right_turn_90():
    # 1️⃣ 조향
    car.steering = STEER_RIGHT
    time.sleep(0.3)

    # 2️⃣ 저속 회전
    car.throttle = THROTTLE_TURN
    time.sleep(T_TURN_90)

    # 3️⃣ 반드시 리셋
    car.throttle = 0
    car.steering = STEER_CENTER
    time.sleep(0.2)

try:
    print("▶ RIGHT TRACK START")

    straight()
    right_turn_90()
    right_turn_90()

    straight()
    right_turn_90()
    right_turn_90()

    print("✅ DONE")

finally:
    car.throttle = 0
    car.steering = 0
    print("🛑 STOP")
