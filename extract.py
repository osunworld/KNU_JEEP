import cv2
import numpy as np

# =========================
# 입력 / 출력
# =========================
INPUT_VIDEO  = "input_drive.mp4"
OUTPUT_VIDEO = "output_lane_final_clean.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("비디오 열기 실패")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (W, H))

# =========================
# 파라미터
# =========================
ROI_Y_RATIO = 0.50
EMA_ALPHA   = 0.7

# 🚗 차량 마스크 (MAIN + SUB)
VEH_Y_START = 0.70
VEH_X_LEFT  = 0.30
VEH_X_RIGHT = 0.70

VEH_SUB_Y_START = 0.85
VEH_SUB_X_LEFT  = 0.25
VEH_SUB_X_RIGHT = 0.75

# 흰색 차선
WHITE_LOWER = np.array([0, 0, 180])
WHITE_UPPER = np.array([180, 40, 255])

# 히스토그램 조건
MIN_PEAK_VALUE = 2500
MIN_PEAK_DIST  = 25
MIN_LANE_WIDTH = 80

last_lane_center = W // 2

# =========================
# 차량 마스킹 (기존 + 추가)
# =========================
def mask_vehicle_dual(frame):
    h, w, _ = frame.shape

    # MAIN
    y0 = int(h * VEH_Y_START)
    y1 = h
    x0 = int(w * VEH_X_LEFT)
    x1 = int(w * VEH_X_RIGHT)
    frame[y0:y1, x0:x1] = 0

    # SUB (아래쪽)
    y_sub = int(h * VEH_SUB_Y_START)
    x0w = int(w * VEH_SUB_X_LEFT)
    x1w = int(w * VEH_SUB_X_RIGHT)
    frame[y_sub:y1, x0w:x1w] = 0

    return (x0, y0, x1, y1), (x0w, y_sub, x1w, y1)

# =========================
# 히스토그램 → 피크 2개 중심
# =========================
def lane_center_from_two_peaks(hist, offset):
    idx = np.argsort(hist)[::-1]
    idx = [i for i in idx if hist[i] > MIN_PEAK_VALUE]

    if len(idx) < 2:
        return None

    peaks = []
    for i in idx:
        if not peaks:
            peaks.append(i)
        elif abs(i - peaks[0]) > MIN_PEAK_DIST:
            peaks.append(i)
            break

    if len(peaks) < 2:
        return None

    return int((peaks[0] + peaks[1]) / 2) + offset

# =========================
# 메인 루프
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- 차량 마스킹 ----------
    (vx0, vy0, vx1, vy1), (sx0, sy0, sx1, sy1) = mask_vehicle_dual(frame)
    vehicle_center_x = (vx0 + vx1) // 2

    # ---------- ROI ----------
    roi_y = int(H * ROI_Y_RATIO)
    roi = frame[roi_y:H, :]

    # ---------- 흰선 + Edge ----------
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    lane_bin = cv2.bitwise_or(white, edges)
    lane_bin = cv2.morphologyEx(
        lane_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    # ---------- Histogram ----------
    hist = np.sum(lane_bin[lane_bin.shape[0] // 2 :, :], axis=0)
    mid = W // 2

    left_center  = lane_center_from_two_peaks(hist[:mid], 0)
    right_center = lane_center_from_two_peaks(hist[mid:], mid)

    # ---------- 차선 중앙 ----------
    if (
        left_center is not None and
        right_center is not None and
        right_center - left_center > MIN_LANE_WIDTH
    ):
        lane_center = (left_center + right_center) // 2
        lane_center = int(
            EMA_ALPHA * last_lane_center +
            (1 - EMA_ALPHA) * lane_center
        )
        last_lane_center = lane_center
    else:
        lane_center = last_lane_center

    # =========================
    # 시각화 (저장용)
    # =========================
    out = frame.copy()

    # 차선 binary (파랑 채널)
    out[roi_y:H, :, 0] = np.maximum(out[roi_y:H, :, 0], lane_bin)

    # 왼쪽 / 오른쪽 차선 중심 (파랑)
    if left_center is not None:
        cv2.line(out, (left_center, roi_y), (left_center, H), (255, 0, 0), 2)
    if right_center is not None:
        cv2.line(out, (right_center, roi_y), (right_center, H), (255, 0, 0), 2)

    # 차선 전체 중심 (초록)
    cv2.line(out, (lane_center, roi_y), (lane_center, H), (0, 255, 0), 3)

    # 차량 중심 (보라)
    cv2.line(out, (vehicle_center_x, vy0), (vehicle_center_x, H), (255, 0, 255), 2)

    # 차량 마스크 표시
    overlay = out.copy()
    cv2.rectangle(overlay, (vx0, vy0), (vx1, vy1), (0, 0, 255), -1)      # MAIN
    cv2.rectangle(overlay, (sx0, sy0), (sx1, sy1), (0, 80, 255), -1)    # SUB
    out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

    writer.write(out)

cap.release()
writer.release()

print("✅ 파란선 = 차선 중심만 표시하는 최종 코드 완료")
