import cv2
import numpy as np
import os

# =========================
# 설정
# =========================
VIDEO_PATHS = [
    "/abr/coss11/repo/robot_data/out76hgk.mp4",
    "/abr/coss11/repo/robot_data/out125.mp4",
    "/abr/coss11/repo/robot_data/out445.mp4",
    "/abr/coss11/repo/robot_data/outljkm12.mp4",
    "/abr/coss11/repo/robot_data/outjknk.mp4",
    "/abr/coss11/repo/robot_data/outadcsd.mp4",
    "/abr/coss11/repo/robot_data/outzxc.mp4",
]

SAVE_DIR = "frames_no_green"
FRAME_STEP = 40
MAX_SAMPLES = 266

# bbox (x1, y1, x2, y2)
bbox = (185, 55, 235, 130)
x1, y1, x2, y2 = bbox

# Green HSV 범위
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

GREEN_RATIO_THRESH = 0.05  # 이 이상이면 green 있음

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 수집 루프
# =========================
saved_total = 0

for video_path in VIDEO_PATHS:
    if saved_total >= MAX_SAMPLES:
        break

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARN] 비디오 열기 실패: {video_path}")
        continue

    frame_idx = 0

    while True:
        if saved_total >= MAX_SAMPLES:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            h, w, _ = frame.shape

            # bbox clamp
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)

            crop = frame[y1c:y2c, x1c:x2c]

            # Green check
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.count_nonzero(mask) / mask.size

            # ❌ Green 없을 때만 저장
            if green_ratio < GREEN_RATIO_THRESH:
                save_path = os.path.join(
                    SAVE_DIR,
                    f"frame_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(save_path, crop)
                saved_total += 1

        frame_idx += 1

    cap.release()

print(f"수집 완료: {saved_total} / {MAX_SAMPLES}")
