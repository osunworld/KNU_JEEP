import time
from autocar3g.driving import Driving

# =========================
# 객체 생성
# =========================
car = Driving()

# =========================
# 파라미터 (속도는 전부 정수)
# =========================
THROTTLE_STRAIGHT = 9   # 🔧 직선 속도 증가

THROTTLE_TURN_1 = 4    # 1번째 회전 (자세 만들기)
THROTTLE_TURN_2 = 6    # 2번째 회전 (확 돌기)

STEER_CENTER = 0.0

T_STRAIGHT_1 = 1.15    # 🔧 직선 시간 감소
T_STRAIGHT_2 = 0.80

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
# 1️⃣ 첫 번째 회전 (진입 + 끝 보강)
# =========================
def right_turn_entry():
    # 진입
    for steer in [-0.06, -0.12, -0.22, -0.38]:
        car.steering = steer
        time.sleep(0.08)

    car.throttle = THROTTLE_TURN_1
    time.sleep(1.6)   # 유지 (각 중요)

    # 끝부분 회전 보강
    car.steering = -0.45
    time.sleep(0.25)

    # 완전히 풀지 않고 연결 유지
    car.steering = -0.08
    time.sleep(0.1)
    # throttle 유지

# =========================
# 2️⃣ 두 번째 회전 (즉시 연결 + 곡률 강화)
# =========================
def right_turn_connect():
    for steer in [-0.25, -0.50, -0.75]:
        car.steering = steer
        time.sleep(0.12)

    # 최대 조향 유지
    time.sleep(0.35)

    car.throttle = THROTTLE_TURN_2
    time.sleep(0.95)

    # 출구 정리
    for steer in [-0.50, -0.25, 0.0]:
        car.steering = steer
        time.sleep(0.10)

    car.throttle = 0
    time.sleep(0.25)

# =========================
# 메인 시퀀스
# =========================
try:
    print("▶ RIGHT TRACK (STRAIGHT SPEED UP / TIME REBALANCED) START")

    straight(T_STRAIGHT_1)

    right_turn_entry()
    right_turn_connect()

    straight(T_STRAIGHT_2)

    right_turn_entry()
    right_turn_connect()

    print("✅ TRACK COMPLETE")

finally:
    car.throttle = 0
    car.steering = 0
    print("🛑 STOP")
