import cv2
import numpy as np

input_video = "outzxc.mp4"
output_video = "out_edge_masked.mp4"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError("비디오를 열 수 없습니다.")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h),
    False   # grayscale output
)

# CLAHE (흐린 영상 필수)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Contrast enhancement
    gray = clahe.apply(gray)

    # 3. Gaussian Blur (엣지 안정화)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Auto Canny
    v = np.median(blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper)

    # 5. Morphology (끊긴 엣지 연결)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    writer.write(edges)

cap.release()
writer.release()
print("✅ Edge 기반 영상 저장 완료:", output_video)
