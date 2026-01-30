import cv2
import numpy as np
import math
import random
import time

# =========================
# Robot Driver (POP AutoCar / Driving wrapper)
# =========================
class CarDriver:
    """
    - Jetson(POP) 환경이면 실제 조향/속도 제어
    - 아니면 print로만 동작(디버그)
    """
    def __init__(self, enable_drive=True):
        self.enable_drive = enable_drive
        self.car = None

        if not enable_drive:
            print("[DRIVER] enable_drive=False -> dry run mode")
            return

        try:
            # 1) pop.driving 우선
            from pop import driving
            self.car = driving.Driving()
            print("[DRIVER] Using pop.driving.Driving()")
        except Exception:
            try:
                # 2) pop.Pilot fallback
                from pop import Pilot
                self.car = Pilot.AutoCar()
                print("[DRIVER] Using pop.Pilot.AutoCar()")
            except Exception as e:
                print("[DRIVER] POP module not available -> dry run mode:", e)
                self.enable_drive = False
                self.car = None

    def set_steering_servo(self, servo_deg_60_120: float):
        servo_deg_60_120 = float(servo_deg_60_120)
        servo_deg_60_120 = max(60.0, min(120.0, servo_deg_60_120))

        if self.enable_drive and self.car is not None:
            # pop은 보통 -0.6~0.6 또는 steering에 float 쓰는 케이스가 많아서
            # 여기서는 "servo=60~120"을 "[-0.6, +0.6]"으로 선형 변환해서 넣어줌.
            # 만약 네 차량이 steering에 바로 60~120을 받는 구조면 아래 mapping을 제거하고 그대로 넣으면 됨.
            steer_norm = (servo_deg_60_120 - 90.0) / 30.0  # 60->-1, 90->0, 120->+1
            steer_norm = max(-1.0, min(1.0, steer_norm))
            steer_norm = steer_norm * 0.6  # 실제 제한
            self.car.steering = float(steer_norm)
        else:
            print(f"[DRY] steering_servo={servo_deg_60_120:.1f}")

    def set_speed(self, speed: float):
        speed = float(speed)
        if self.enable_drive and self.car is not None:
            if speed > 0:
                self.car.forward(speed)
            elif speed < 0:
                self.car.backward(abs(speed))
            else:
                self.car.stop()
        else:
            print(f"[DRY] speed={speed:.1f}")

    def stop(self):
        if self.enable_drive and self.car is not None:
            self.car.stop()
            self.car.steering = 0.0
        else:
            print("[DRY] stop()")


# =========================
# Lane Follower + Control
# =========================
class LaneFollowerController:
    def __init__(
        self,
        # —— vision scan ——
        y_ratio=0.55,
        white_thresh=200,
        min_run=3,
        robot_anchor=None,

        # —— mapping: angle -> servo ——
        servo_center=90.0,
        servo_min=60.0,
        servo_max=120.0,
        deg_per_servo=1.0,   # angle_deg 1도당 servo 몇 도 움직일지 (튜닝 포인트)
        deadband_deg=2.0,    # |angle| <= deadband면 조향 0으로(=center 유지)

        # —— speed schedule ——
        speed_fast=8.0,      # 직진에 가까울 때
        speed_mid=6.0,
        speed_slow=4.0,      # 크게 꺾을 때
        err_mid=10.0,        # |angle| 경계
        err_high=20.0,

        # —— fallback 탐색 ——
        fallback_hold_frames=15,
        fallback_servo_offset=18.0,  # fallback 시 center에서 얼마나 꺾을지(servo 도 단위)
        seed=42,
    ):
        self.y_ratio = y_ratio
        self.white_thresh = white_thresh
        self.min_run = min_run
        self.robot_anchor = robot_anchor

        self.servo_center = servo_center
        self.servo_min = servo_min
        self.servo_max = servo_max
        self.deg_per_servo = deg_per_servo
        self.deadband_deg = deadband_deg

        self.speed_fast = speed_fast
        self.speed_mid = speed_mid
        self.speed_slow = speed_slow
        self.err_mid = err_mid
        self.err_high = err_high

        self.fallback_hold_frames = fallback_hold_frames
        self.fallback_servo_offset = fallback_servo_offset
        self._fallback_dir = None
        self._fallback_left = 0

        random.seed(seed)

    # --------- helpers ----------
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def angle_to_servo(self, angle_deg: float):
        """
        angle_deg(+)오른쪽, (-)왼쪽  -> servo(60~120), center=90
        """
        angle_deg = float(angle_deg)

        # deadband
        if abs(angle_deg) <= self.deadband_deg:
            return self.servo_center

        # 선형 매핑: angle이 +면 servo도 +로 이동
        servo = self.servo_center + (angle_deg * self.deg_per_servo)
        servo = self.clamp(servo, self.servo_min, self.servo_max)
        return servo

    def angle_to_speed(self, angle_deg: float, mode: str):
        """
        오차(|angle|)가 클수록 감속.
        fallback이면 더 느리게.
        """
        e = abs(float(angle_deg))

        if mode == "fallback":
            return max(2.0, self.speed_slow - 1.0)

        if e < self.err_mid:
            return self.speed_fast
        elif e < self.err_high:
            return self.speed_mid
        else:
            return self.speed_slow

    # --------- vision ----------
    def _compute_lane_angle(self, frame_bgr, debug=True):
        h, w = frame_bgr.shape[:2]
        y_scan = int(h * self.y_ratio)
        y_scan = max(0, min(y_scan, h - 1))

        if self.robot_anchor is None:
            rx, ry = (w // 2, int(h * 0.88))
        else:
            rx, ry = self.robot_anchor

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, bin_white = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        bin_white = cv2.morphologyEx(bin_white, cv2.MORPH_OPEN, kernel, iterations=1)

        row = bin_white[y_scan, :]
        if np.count_nonzero(row) == 0:
            return None

        cx = w // 2
        row01 = (row > 0).astype(np.uint8)

        # left run
        xL = None
        x = cx - 1
        while x >= 0:
            if row01[x] == 1:
                x_end = x
                while x >= 0 and row01[x] == 1:
                    x -= 1
                x_start = x + 1
                if (x_end - x_start + 1) >= self.min_run:
                    xL = x_end
                    break
            x -= 1

        # right run
        xR = None
        x = cx + 1
        while x < w:
            if row01[x] == 1:
                x_start = x
                while x < w and row01[x] == 1:
                    x += 1
                x_end = x - 1
                if (x_end - x_start + 1) >= self.min_run:
                    xR = x_start
                    break
            x += 1

        if xL is None or xR is None:
            return None

        mid_x = int((xL + xR) / 2)
        target_pt = (mid_x, y_scan)

        dx = target_pt[0] - rx
        dy = ry - target_pt[1]
        angle_rad = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_rad)

        dbg = frame_bgr.copy()
        if debug:
            cv2.line(dbg, (0, y_scan), (w - 1, y_scan), (255, 0, 0), 1)
            cv2.circle(dbg, (int(xL), y_scan), 4, (0, 255, 255), -1)
            cv2.circle(dbg, (int(xR), y_scan), 4, (0, 255, 255), -1)
            cv2.circle(dbg, target_pt, 5, (0, 0, 255), -1)
            cv2.circle(dbg, (rx, ry), 5, (0, 255, 0), -1)
            cv2.line(dbg, (rx, ry), target_pt, (0, 0, 255), 2)

        return {
            "angle_deg": float(angle_deg),
            "debug_img": dbg,
            "bin_white": bin_white,
        }

    # --------- main step ----------
    def step(self, frame_bgr, debug=True):
        lane = self._compute_lane_angle(frame_bgr, debug=debug)

        # lane success
        if lane is not None:
            self._fallback_dir = None
            self._fallback_left = 0

            angle_deg = lane["angle_deg"]
            servo = self.angle_to_servo(angle_deg)
            speed = self.angle_to_speed(angle_deg, mode="lane")

            return {
                "mode": "lane",
                "angle_deg": angle_deg,
                "servo": servo,
                "speed": speed,
                "debug_img": lane["debug_img"],
                "bin_white": lane["bin_white"],
            }

        # fallback
        if self._fallback_left <= 0 or self._fallback_dir is None:
            self._fallback_dir = random.choice([-1, +1])
            self._fallback_left = self.fallback_hold_frames

        self._fallback_left -= 1

        # fallback은 angle 대신 servo offset 기반으로
        servo = self.servo_center + self._fallback_dir * self.fallback_servo_offset
        servo = self.clamp(servo, self.servo_min, self.servo_max)

        # fallback 속도(느리게)
        speed = self.angle_to_speed(angle_deg=self.err_high, mode="fallback")

        dbg = frame_bgr.copy()
        if debug:
            txt = f"FALLBACK {'L' if self._fallback_dir<0 else 'R'} hold={self._fallback_left}"
            cv2.putText(dbg, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(dbg, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        return {
            "mode": "fallback",
            "angle_deg": None,
            "servo": float(servo),
            "speed": float(speed),
            "debug_img": dbg,
            "bin_white": None,
        }


# =========================
# Run (camera/video)
# =========================
def main():
    # ---- input 설정 ----
    VIDEO_PATH = 0  # 웹캠이면 0, 파일이면 "test.mp4"

    # ---- 실제 구동 여부 ----
    ENABLE_DRIVE = True   # Jetson에서 실제 주행이면 True, 맥/PC 디버그면 False

    # ---- 컨트롤 파라미터 (튜닝 포인트) ----
    ctrl = LaneFollowerController(
        y_ratio=0.55,
        white_thresh=200,
        min_run=3,

        servo_center=90,
        servo_min=60,
        servo_max=120,
        deg_per_servo=0.8,     # ✅ 핵심 튜닝: angle 1도당 servo 0.8도 움직임
        deadband_deg=2.0,      # ✅ 작은 흔들림 제거

        speed_fast=8,
        speed_mid=6,
        speed_slow=4,
        err_mid=10,
        err_high=20,

        fallback_hold_frames=15,
        fallback_servo_offset=18,  # ✅ 차선 없을때는 확실히 탐색
    )

    driver = CarDriver(enable_drive=ENABLE_DRIVE)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video/cam: {VIDEO_PATH}")

    win = "LaneFollow Debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out = ctrl.step(frame, debug=True)

            # ---- 실제 로봇 제어 ----
            driver.set_steering_servo(out["servo"])
            driver.set_speed(out["speed"])

            # ---- 디버그 출력 ----
            disp = out["debug_img"]
            mode = out["mode"]
            servo = out["servo"]
            speed = out["speed"]
            angle = out["angle_deg"]

            text = f"mode:{mode}  servo:{servo:.1f}  speed:{speed:.1f}"
            if angle is not None:
                text += f"  angle:{angle:.2f}"

            # text overlay
            cv2.rectangle(disp, (5, 5), (5 + 640, 45), (0, 0, 0), -1)
            cv2.putText(disp, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        driver.stop()


if __name__ == "__main__":
    main()