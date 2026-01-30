import time
from autocar3g.driving import Driving

# =========================
# 객체 생성
# =========================
car = Driving()

# =========================
# 파라미터 (속도는 전부 정수)
# =========================
THROTTLE_STRAIGHT = 9

THROTTLE_TURN_1 = 4
THROTTLE_TURN_2 = 6

STEER_CENTER = 0.0

T_STRAIGHT_1 = 1.32
T_STRAIGHT_2 = 0.60

# =========================
# 직선 주행
# =========================
def straight(duration):
    car.steering = STEER_CENTER
    time.sleep(0.1)

    car.throttle = THROTTLE_STRAIGHT
    time.sleep(duration)

    car.throttle = 0
    time.sleep(0.20)

# =========================
# 1️⃣ 첫 번째 회전 (유지)
# =========================
def right_turn_entry():
    for steer in [-0.06, -0.12, -0.22, -0.38]:
        car.steering = steer
        time.sleep(0.08)

    car.throttle = THROTTLE_TURN_1
    time.sleep(2.02)

    car.steering = -0.45
    time.sleep(0.25)

    car.steering = -0.08
    time.sleep(0.1)
    # throttle 유지

# =========================
# 2️⃣ 두 번째 회전 (⏱ 시간만 더 길게)
# =========================
def right_turn_connect():
    for steer in [-0.25, -0.30, -0.52]:
        car.steering = steer
        time.sleep(0.07)

    # 최대 조향 유지 (곡률 동일)
    time.sleep(0.65)

    # 🔧 회전 유지 시간 증가
    car.throttle = THROTTLE_TURN_2
    time.sleep(1.92)   # ← 1.62 → 1.72

    # 출구 정리
    for steer in [-0.35, -0.20, 0.0]:
        car.steering = steer
        time.sleep(0.08)

    car.throttle = 0
    car.steering = STEER_CENTER
    time.sleep(0.25)
    
def right_turn_connect():
    for steer in [-0.25, -0.30, -0.52]:
        car.steering = steer
        time.sleep(0.07)

    # 최대 조향 유지 (곡률 동일)
    time.sleep(0.65)

    # 🔧 회전 유지 시간 증가
    car.throttle = THROTTLE_TURN_2
    time.sleep(1.92)   # ← 1.62 → 1.72

    # 출구 정리
    for steer in [-0.35, -0.20, 0.0]:
        car.steering = steer
        time.sleep(0.08)

    car.throttle = 0
    car.steering = STEER_CENTER
    time.sleep(0.05)

# =========================
# 메인 시퀀스
# =========================
try:
    print("▶ RIGHT TRACK (2ND TURN TIME EXTENDED) START")

    straight(T_STRAIGHT_1)

    right_turn_entry()
    print("1")
    right_turn_connect()

    straight(T_STRAIGHT_2)

    right_turn_entry()
    right_turn_connect()

    print("✅ TRACK COMPLETE")

finally:
    car.throttle = 0
    car.steering = 0
    print("🛑 STOP")
