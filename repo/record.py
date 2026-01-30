import cv2

# ===============================
# setting
# ===============================
CAM_ID = 0
OUTPUT_MP4 = "camera_only.mp4"
FPS = 20.0
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# ===============================
# open camera
# ===============================
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("camera open failed")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_MP4,
    FOURCC,
    FPS,
    (width, height)
)

# ===============================
#  loop record
# ===============================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)

except KeyboardInterrupt:
    print("\n[INFO] Recording stopped")

cap.release()
writer.release()
print("[INFO] Saved successfully")
