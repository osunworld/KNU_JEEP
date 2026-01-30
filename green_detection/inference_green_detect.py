import cv2
import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# Config
# ----------------------------
VIDEO_PATH = "test366.mp4"
PT_PATH = "green_ped_model_3.pt"
THRESH = 0.3

# bbox = (x1, y1, x2, y2)
x1, y1, x2, y2 = 185, 55, 235, 130
ROI_X, ROI_Y = x1, y1
ROI_W, ROI_H = (x2 - x1), (y2 - y1)   # 50, 75

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Model (학습 코드와 동일해야 함)
# ----------------------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 75x50 -> 37x25
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 37x25 -> 18x12
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> 64x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),  # logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ----------------------------
# Load model
# ----------------------------
ckpt = torch.load(PT_PATH, map_location=DEVICE)
model = TinyCNN().to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ----------------------------
# Video inference + playback
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

delay_ms = 1
win_name = "Green Pedestrian Detector"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

with torch.no_grad():
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        pred, prob = 0, 0.0

        # ROI가 프레임 밖으로 나가는 경우 방지
        xA = max(0, min(ROI_X, w - 1))
        yA = max(0, min(ROI_Y, h - 1))
        xB = max(0, min(ROI_X + ROI_W, w))
        yB = max(0, min(ROI_Y + ROI_H, h))

        roi = frame[yA:yB, xA:xB]

        if roi.shape[0] == ROI_H and roi.shape[1] == ROI_W:
            x = roi.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))       # HWC -> CHW
            x = torch.from_numpy(x).unsqueeze(0) # (1,3,75,50)
            x = x.to(DEVICE)

            logits = model(x)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= THRESH else 0

        disp = frame.copy()

        # ROI 박스
        box_color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        cv2.rectangle(disp, (xA, yA), (xB, yB), box_color, 2)

        # ROI 패널 (우측 상단)
        roi_panel = None
        if roi.size:
            roi_vis = cv2.resize(roi, (ROI_W * 3, ROI_H * 3), interpolation=cv2.INTER_NEAREST)
            rh, rw = roi_vis.shape[:2]
            x0 = max(0, w - rw - 10)
            y0 = 10
            disp[y0:y0 + rh, x0:x0 + rw] = roi_vis
            roi_panel = (x0, y0, x0 + rw, y0 + rh)

        # ✅ 텍스트는 1개만(좌상단) 출력
        text = f"frame: {frame_idx}  pred: {pred}  prob: {prob:.3f}  thr: {THRESH:.2f}"
        tx, ty = 15, 35

        # 텍스트 박스가 ROI 패널과 겹치면 아래로 내리기
        if roi_panel is not None:
            px1, py1, px2, py2 = roi_panel
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_box = (tx, ty - th, tx + tw, ty + baseline)

            def intersects(a, b):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

            if intersects(text_box, roi_panel):
                ty = py2 + 35

        # 텍스트 배경 + 텍스트
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(disp, (tx - 5, ty - th - 5), (tx + tw + 5, ty + baseline + 5), (0, 0, 0), -1)
        cv2.putText(disp, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, disp)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(' ') or key2 == ord('s'):
                    break
                if key2 == 27 or key2 == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

cap.release()
cv2.destroyAllWindows()