import cv2
import numpy as np


class LaneDetectorV2:
    def __init__(self, img_width):
        self.W = img_width
        self.last_lane_center = None
        self.last_lane_width = None

        self.ROI_Y_RATIO = 0.50

        self.VEH_Y_MAIN = 0.70
        self.VEH_X_MAIN_L = 0.30
        self.VEH_X_MAIN_R = 0.70

        self.VEH_Y_SUB = 0.85
        self.VEH_X_SUB_L = 0.25
        self.VEH_X_SUB_R = 0.75

        self.WHITE_LOWER = np.array([0, 0, 180])
        self.WHITE_UPPER = np.array([180, 40, 255])

        self.MIN_PEAK_VALUE = 2500
        self.MIN_PEAK_DIST = 25

    def _mask_vehicle(self, frame):
        h, w, _ = frame.shape

        y0 = int(h * self.VEH_Y_MAIN)
        x0 = int(w * self.VEH_X_MAIN_L)
        x1 = int(w * self.VEH_X_MAIN_R)
        frame[y0:h, x0:x1] = 0

        y1 = int(h * self.VEH_Y_SUB)
        x2 = int(w * self.VEH_X_SUB_L)
        x3 = int(w * self.VEH_X_SUB_R)
        frame[y1:h, x2:x3] = 0

        return (x0 + x1) // 2

    def _two_peak_center(self, hist, offset):
        idx = np.argsort(hist)[::-1]
        idx = [i for i in idx if hist[i] > self.MIN_PEAK_VALUE]

        if len(idx) < 2:
            return None

        p0 = idx[0]
        for i in idx[1:]:
            if abs(i - p0) > self.MIN_PEAK_DIST:
                return int((p0 + i) / 2) + offset
        return None

    def _compute_curvature(self):
        if self.last_lane_center is None:
            return 0.0
        img_center = self.W // 2
        return abs(self.last_lane_center - img_center) / (self.W / 2)

    def _select_lookahead_band(self, lane_bin):
        h = lane_bin.shape[0]
        curvature = self._compute_curvature()

        if curvature < 0.15:
            y0, y1 = int(h * 0.15), int(h * 0.35)
        elif curvature < 0.30:
            y0, y1 = int(h * 0.25), int(h * 0.50)
        else:
            y0, y1 = int(h * 0.45), int(h * 0.70)

        return lane_bin[y0:y1, :]

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

        band = self._select_lookahead_band(lane_bin)
        hist = np.sum(band, axis=0)

        mid = W // 2
        left_center = self._two_peak_center(hist[:mid], 0)
        right_center = self._two_peak_center(hist[mid:], mid)

        if left_center is not None and right_center is not None:
            lane_center = (left_center + right_center) // 2
            self.last_lane_center = lane_center
            self.last_lane_width = right_center - left_center
        elif left_center is not None and self.last_lane_width is not None:
            lane_center = left_center + self.last_lane_width // 2
            self.last_lane_center = lane_center
        elif right_center is not None and self.last_lane_width is not None:
            lane_center = right_center - self.last_lane_width // 2
            self.last_lane_center = lane_center
        else:
            lane_center = self.last_lane_center

        return lane_center, vehicle_center_x
