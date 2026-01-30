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
            from pop import driving
            self.car = driving.Driving()
            print("[DRIVER] Using pop.driving.Driving()")
        except Exception:
            try:
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
            # 60~120 -> -0.6~+0.6 변환(네 POP 구현이 이 범위를 받는다고 가정)
            steer_norm = (servo_deg_60_120 - 90.0) / 30.0  # 60->-1, 90->0, 120->+1
            steer_norm = max(-1.0, min(1.0, steer_norm))
            steer_norm = steer_norm * 0.6
            try:
                self.car.steering = float(steer_norm)
            except Exception as e:
                print("[DRIVER] steering set failed:", e)
        else:
            print(f"[DRY] steering_servo={servo_deg_60_120:.1f}")

    def set_speed(self, speed: float):
        speed = float(speed)

        if self.enable_drive and self.car is not None:
            try:
                if speed > 0:
                    self.car.forward(speed)
                elif speed < 0:
                    self.car.backward(abs(speed))
                else:
                    self.car.stop()
            except Exception as e:
                print("[DRIVER] speed set failed:", e)
        else:
            print(f"[DRY] speed={speed:.1f}")

    def stop(self):
        if self.enable_drive and self.car is not None:
            try:
                self.car.stop()
            except Exception:
                pass
            try:
                self.car.steering = 0.0
            except Exception:
                pass
        else:
            print("[DRY] stop()")


# =========================
# Lane Follower + Control
# =========================
class LaneFollowerController:
    def __init__(
        self,
        y_ratio=0.55,
        white_thresh=200,
        min_run=3,
        robot_anchor=None,

        servo_center=90.0,
        servo_min=60.0,
        servo_max=120.0,
        deg_per_servo=0.8,
        deadband_deg=2.0,

        speed_fast=8.0,
        speed_mid=6.0,
        speed_slow=4.0,
        err_mid=10.0,
        err_high=20.0,

        fallback_hold_frames=15,
        fallback_servo_offset=18.0,
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

    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def angle_to_servo(self, angle_deg: float):
        angle_deg = float(angle_deg)
        if abs(angle_deg) <= self.deadband_deg:
            return self.servo_center
        servo = self.servo_center + (angle_deg * self.deg_per_servo)
        return self.clamp(servo, self.servo_min, self.servo_max)

    def angle_to_speed(self, angle_deg: float, mode: str):
        e = abs(float(angle_deg))
        if mode == "fallback":
            return max(2.0, self.speed_slow - 1.0)

        if e < self.err_mid:
            return self.speed_fast
        elif e < self.err_high:
            return self.speed_mid
        else:
            return self.speed_slow

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
        angle_deg = math.degrees(math.atan2(dx, dy))

        dbg = frame_bgr.copy()
        if debug:
            cv2.line(dbg, (0, y_scan), (w - 1, y_scan), (255, 0, 0), 1)
            cv2.circle(dbg, (int(xL), y_scan), 4, (0, 255, 255), -1)
            cv2.circle(dbg, (int(xR), y_scan), 4, (0, 255, 255), -1)
            cv2.circle(dbg, target_pt, 5, (0, 0, 255), -1)
            cv2.circle(dbg, (rx, ry), 5, (0, 255, 0), -1)
            cv2.line(dbg, (rx, ry), target_pt, (0, 0, 255), 2)

        return {"angle_deg": float(angle_deg), "debug_img": dbg}

    def step(self, frame_bgr, debug=True):
        lane = self._compute_lane_angle(frame_bgr, debug=debug)

        if lane is not None:
            self._fallback_dir = None
            self._fallback_left = 0

            angle_deg = lane["angle_deg"]
            servo = self.angle_to_servo(angle_deg)
            speed = self.angle_to_speed(angle_deg, mode="lane")

            return {
                "mode": "lane",
                "angle_deg": angle_deg,
                "servo": float(servo),
                "speed": float(speed),
                "debug_img": lane["debug_img"],
            }

        # fallback
        if self._fallback_left <= 0 or self._fallback_dir is None:
            self._fallback_dir = random.choice([-1, +1])
            self._fallback_left = self.fallback_hold_frames
        self._fallback_left -= 1

        servo = self.servo_center + self._fallback_dir * self.fallback_servo_offset
        servo = self.clamp(servo, self.servo_min, self.servo_max)
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
        }


# =========================
# Camera open helper
# =========================
def open_camera_auto(max_idx=10, w=320, h=240, fps=30):
    """
    Jetson에서 /dev/video* 여러 개일 수 있어서 자동으로 되는 카메라를 찾음.
    1) V4L2로 시도
    2) 안 되면 CAP_ANY로도 한 번 더 시도
    """
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        for idx in range(max_idx):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            cap.set(cv2.CAP_PROP_FPS, int(fps))

            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[CAM] opened backend={backend} index={idx}, shape={frame.shape}")
                return cap

            cap.release()

    return None


# =========================
# Run
# =========================
def main():
    ENABLE_DRIVE = True  # ✅ 실제 로봇이면 True / 디버그면 False

    ctrl = LaneFollowerController(
        y_ratio=0.55,
        white_thresh=200,  # ✅ 차선이 잘 안 잡히면 160~220 사이 튜닝
        min_run=3,
        servo_center=90,
        servo_min=60,
        servo_max=120,
        deg_per_servo=0.8,
        deadband_deg=2.0,
        speed_fast=8,
        speed_mid=6,
        speed_slow=4,
        err_mid=10,
        err_high=20,
        fallback_hold_frames=15,
        fallback_servo_offset=18,
    )

    driver = CarDriver(enable_drive=ENABLE_DRIVE)

    cap = open_camera_auto(max_idx=10, w=320, h=240, fps=30)
    if cap is None:
        driver.stop()
        raise RuntimeError("No camera found. (Check /dev/video*, power, permissions)")

    win = "LaneFollow Debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CAM] read failed -> stop & exit")
                driver.stop()
                break

            out = ctrl.step(frame, debug=True)

            # ---- 실제 로봇 제어 ----
            driver.set_steering_servo(out["servo"])
            driver.set_speed(out["speed"])

            # ---- 디버그 표시 ----
            disp = out["debug_img"]
            text = f"mode:{out['mode']}  servo:{out['servo']:.1f}  speed:{out['speed']:.1f}"
            if out["angle_deg"] is not None:
                text += f"  angle:{out['angle_deg']:.2f}"

            # ✅ 텍스트 길이에 맞춰 배경 박스 자동
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(disp, (5, 5), (5 + tw + 20, 5 + th + base + 20), (0, 0, 0), -1)
            cv2.putText(disp, text, (15, 5 + th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        driver.stop()


if __name__ == "__main__":
    main()