import cv2
import os

cap = cv2.VideoCapture("test372.mp4")
os.makedirs("track_dataset", exist_ok=True)
i = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    i += 1
    if i % 20 == 0:
        cv2.imwrite(f"track_dataset/frame_{i:05d}.jpg", frame)

cap.release()