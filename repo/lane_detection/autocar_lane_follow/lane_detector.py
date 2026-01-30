# lane_detector.py
import cv2
import numpy as np

class LaneDetector:
    def __init__(self, img_width):
        self.W = img_width

        self.last_lane_center = None
        self.last_lane_width  = None

        # ROI
        self.ROI_Y_RATIO = 0.50

        # Vehicle mask
        self.VEH_Y_START = 0.70
        self.VEH_X_LEFT  = 0.30
        self.VEH_X_RIGHT = 0.70

        self.VEH_SUB_Y_START = 0.85
        self.VEH_SUB_X_LEFT  = 0.25
        self.VEH_SUB_X_RIGHT = 0.75

        # White lane
        self.WHITE_LOWER = np.array([0, 0, 180])
        self.WHITE_UPPER = np.array([180, 40, 255])

        # Histogram
        self.MIN_PEAK_VALUE = 2500
        self.MIN_PEAK_DIST  = 25
        self.MIN_LANE_WIDTH = 80

    # -------------------------
    def _mask_vehicle(self, frame):
        h, w, _ = frame.shape

        y0 = int(h * self.VEH_Y_START)
        x0 = int(w * self.VEH_X_LEFT)
        x1 = int(w * self.VEH_X_RIGHT)
        frame[y0:h, x0:x1] = 0

        y_sub = int(h * self.VEH_SUB_Y_START)
        x0w = int(w * self.VEH_SUB_X_LEFT)
        x1w = int(w * self.VEH_SUB_X_RIGHT)
        frame[y_sub:h, x0w:x1w] = 0

        return (x0 + x1) // 2  # vehicle center x

    # -------------------------
    def _center_from_two_peaks(self, hist, offset):
        idx = np.argsort(hist)[::-1]
        idx = [i for i in idx if hist[i] > self.MIN_PEAK_VALUE]

        if len(idx) < 2:
            return None

        peaks = []
        for i in idx:
            if not peaks:
                peaks.append(i)
            elif abs(i - peaks[0]) > self.MIN_PEAK_DIST:
                peaks.append(i)
                break

        if len(peaks) < 2:
            return None

        return int((peaks[0] + peaks[1]) / 2) + offset

    # -------------------------
    def process(self, frame):
        H, W, _ = frame.shape
        vehicle_center_x = self._mask_vehicle(frame)

        roi_y = int(H * self.ROI_Y_RATIO)
        roi = frame[roi_y:H, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)

        lane_bin = cv2.bitwise_or(white, edges)
        lane_bin = cv2.morphologyEx(
            lane_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        hist = np.sum(lane_bin[lane_bin.shape[0]//2:, :], axis=0)
        mid = W // 2

        left_center  = self._center_from_two_peaks(hist[:mid], 0)
        right_center = self._center_from_two_peaks(hist[mid:], mid)

        # =========================
        # Fallback logic
        # =========================
        if left_center is not None and right_center is not None:
            # 정상
            lane_center = (left_center + right_center) // 2
            self.last_lane_center = lane_center
            self.last_lane_width  = right_center - left_center

        elif left_center is not None and self.last_lane_width is not None:
            # 왼쪽만
            virtual_right = left_center + self.last_lane_width
            lane_center = (left_center + virtual_right) // 2
            self.last_lane_center = lane_center

        elif right_center is not None and self.last_lane_width is not None:
            # 오른쪽만
            virtual_left = right_center - self.last_lane_width
            lane_center = (virtual_left + right_center) // 2
            self.last_lane_center = lane_center

        else:
            # 완전 소실
            lane_center = self.last_lane_center

        return lane_center, vehicle_center_x
