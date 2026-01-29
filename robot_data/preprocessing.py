import cv2
import numpy as np

INPUT_VIDEO  = "outjknk.mp4"
OUTPUT_VIDEO = "outjknk_bbox.mp4"

# -------------------------
# bbox (x1, y1, x2, y2)
# -------------------------
bbox = (185, 55, 235, 130)


cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("비디오를 열 수 없습니다.")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    OUTPUT_VIDEO, fourcc, fps, (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1, x2, y2 = bbox

    # bbox 그리기 (초록색)
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),   # BGR
        2
    )

    # optional: 중심점
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    writer.write(frame)

cap.release()
writer.release()

print("Saved:", OUTPUT_VIDEO)
