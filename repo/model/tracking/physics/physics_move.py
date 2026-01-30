import time
import math
import csv

from autocar3g.driving import Driving
from autocar3g.encoder import Encoder
from autocar3g.imu import Imu


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class PathFollower:
    def __init__(self, csv_path):
        self.px, self.py, self.pv = [], [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            # 컬럼명이 정확히 x,y,v 라는 전제
            for row in reader:
                self.px.append(float(row["x"]))
                self.py.append(float(row["y"]))
                self.pv.append(float(row["v"]))

        self.N = len(self.px)
        if self.N == 0:
            raise ValueError(f"CSV is empty or invalid: {csv_path}")

        self.last_near = 0  # 인덱스 진행 방지용

    def find_nearest(self, x, y, window=80):
        # 뒤로 튀는 것 방지: last_near 주변만 탐색
        s = max(0, self.last_near - window)
        e = min(self.N, self.last_near + window)

        best_i = s
        best_d2 = 1e18
        for i in range(s, e):
            dx = self.px[i] - x
            dy = self.py[i] - y
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        self.last_near = max(self.last_near, best_i)
        return self.last_near

    def find_lookahead(self, i_near, Ld):
        dist = 0.0
        i = i_near
        while dist < Ld and i + 1 < self.N:
            dx = self.px[i+1] - self.px[i]
            dy = self.py[i+1] - self.py[i]
            dist += math.hypot(dx, dy)
            i += 1
        return i

    def pure_pursuit_steer(self, x, y, yaw, v, wheelbase, steer_max_rad):
        i_near = self.find_nearest(x, y)

        # Lookahead: 속도 비례 + 최소값
        Ld = max(0.4, 0.6 * max(v, 0.1))
        i_tgt = self.find_lookahead(i_near, Ld)

        # 목표점(월드)
        tx = self.px[i_tgt] - x
        ty = self.py[i_tgt] - y

        # 차량좌표로 회전(월드->차량)
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        x_t = c * tx - s * ty
        y_t = s * tx + c * ty

        L = math.hypot(x_t, y_t) + 1e-9
        kappa = 2.0 * y_t / (L * L)
        steer_rad = math.atan(wheelbase * kappa)

        # steer_rad -> [-1, 1]로 정규화 (최대 조향각 기준)
        steer_norm = clamp(steer_rad / steer_max_rad, -1.0, 1.0)

        v_ref = float(self.pv[i_near])  # 또는 i_tgt의 v를 써도 됨
        return steer_norm, v_ref, i_near, i_tgt


class DeadReckoning:
    def __init__(self, wheel_radius, ticks_per_rev, track_width):
        self.wheel_radius = wheel_radius
        self.ticks_per_rev = ticks_per_rev
        self.track_width = track_width

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.last_enc = None

    def ticks_to_dist(self, dticks):
        # (dticks / ticks_per_rev) * (2*pi*R)
        return (dticks / self.ticks_per_rev) * (2.0 * math.pi * self.wheel_radius)

    def update(self, enc_lr, yaw_rad):
        # enc_lr: (left_ticks, right_ticks) absolute reading
        if self.last_enc is None:
            self.last_enc = enc_lr
            self.yaw = yaw_rad
            return self.x, self.y, self.yaw

        dl_ticks = enc_lr[0] - self.last_enc[0]
        dr_ticks = enc_lr[1] - self.last_enc[1]
        self.last_enc = enc_lr

        dl = self.ticks_to_dist(dl_ticks)
        dr = self.ticks_to_dist(dr_ticks)
        ds = 0.5 * (dl + dr)

        # yaw는 IMU 것을 우선 사용(오차 누적 완화)
        self.yaw = yaw_rad

        self.x += ds * math.cos(self.yaw)
        self.y += ds * math.sin(self.yaw)

        return self.x, self.y, self.yaw


def yaw_deg_to_rad(yaw_deg):
    # imu.euler()가 (yaw, roll, pitch)이며 degree로 보임(예제 출력이 109.xxx)
    return math.radians(yaw_deg)


def throttle_from_speed(v_ref):
    """
    CSV v 단위가 m/s라고 가정.
    throttle은 -99~99.
    일단은 단순 비례로 시작하고, 나중에 엔코더 기반 속도 PID 추천.
    """
    k = 8.0  # 튜닝 필요
    return int(clamp(k * v_ref, -99, 99))


def main():
    car = Driving()
    enc = Encoder()
    imu = Imu()

    follower = PathFollower("repo/physics/path_red_spline_rounded.csv")

    # ===== 반드시 본인 차량 스펙으로 설정 =====
    wheel_radius = 0.03      # [m] 예시: 3cm
    ticks_per_rev = 360.0    # 예시
    track_width = 0.14       # [m] 좌우 바퀴 간 거리 예시
    wheelbase = 0.15         # [m] 앞-뒤 축간거리(모르면 대략이라도)
    steer_max_rad = math.radians(25.0)  # [-1,1]이 대응하는 최대 조향각(캘리브 필요)
    # ======================================

    dr = DeadReckoning(wheel_radius, ticks_per_rev, track_width)

    dt = 0.05  # 20Hz
    car.throttle = 0
    car.steering = 0
    time.sleep(0.5)

    try:
        while True:
            # 센서 읽기
            yaw_deg, roll_deg, pitch_deg = imu.euler()
            yaw = yaw_deg_to_rad(yaw_deg)

            enc_lr = enc.read()  # (left, right) absolute ticks
            x, y, yaw = dr.update(enc_lr, yaw)

            # 현재 속도 v 추정(원하면 엔코더로 계산해서 넣기)
            v_est = 0.0  # 임시값: Lookahead 계산에만 쓰임 (나중에 엔코더로 추정 권장)

            # 경로 추종
            steer, v_ref, i_near, i_tgt = follower.pure_pursuit_steer(
                x, y, yaw, v_est, wheelbase, steer_max_rad
            )

            # 속도 명령
            thr = throttle_from_speed(v_ref)
            thr = clamp(thr, -15, 15)

            # 차량에 적용
            car.steering = float(steer)
            car.throttle = int(thr)

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        car.throttle = 0
        car.steering = 0
        time.sleep(0.2)


if __name__ == "__main__":
    main()