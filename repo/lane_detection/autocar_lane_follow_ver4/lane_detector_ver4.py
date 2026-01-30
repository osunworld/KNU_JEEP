import cv2
import numpy as np


class LaneDetectorV4:
    def __init__(self, img_width):
        self.W = img_width
        self.last_lane_width = None
        self.prev_lane_near = None

        self.WHITE_LOWER = np.array([0, 0, 180])
        self.WHITE_UPPER = np.array([180, 40, 255])

        self.MIN_PEAK_VALUE = 2500
        self.MIN_PEAK_DIST = 25

        self.dy = None

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

    def process(self, frame):
        H, W, _ = frame.shape
        vehicle_center = W // 2

        roi = frame[int(H * 0.5):H, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)

        lane_bin = cv2.bitwise_or(white, edges)
        lane_bin = cv2.morphologyEx(
            lane_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        h = lane_bin.shape[0]

        near_y1, near_y2 = int(h * 0.6), int(h * 0.8)
        far_y1, far_y2 = int(h * 0.25), int(h * 0.45)

        self.dy = abs(far_y1 - near_y1)

        band_near = lane_bin[near_y1:near_y2, :]
        band_far = lane_bin[far_y1:far_y2, :]

        hist_near = np.sum(band_near, axis=0)
        hist_far = np.sum(band_far, axis=0)

        mid = W // 2

        left_near = self._two_peak_center(hist_near[:mid], 0)
        right_near = self._two_peak_center(hist_near[mid:], mid)
        left_far = self._two_peak_center(hist_far[:mid], 0)
        right_far = self._two_peak_center(hist_far[mid:], mid)

        lane_near, lane_far = None, None

        if left_near is not None and right_near is not None:
            lane_near = (left_near + right_near) // 2
            self.last_lane_width = right_near - left_near

        if left_far is not None and right_far is not None:
            lane_far = (left_far + right_far) // 2

        if lane_near is None and self.last_lane_width is not None:
            if left_near is not None:
                lane_near = left_near + self.last_lane_width // 2
            elif right_near is not None:
                lane_near = right_near - self.last_lane_width // 2

        if lane_far is None and lane_near is not None and self.prev_lane_near is not None:
            lane_far = lane_near + (lane_near - self.prev_lane_near)

        if lane_far is None:
            lane_far = lane_near

        curvature = abs(lane_far - lane_near)

        self.prev_lane_near = lane_near

        return lane_near, lane_far, vehicle_center, curvature
