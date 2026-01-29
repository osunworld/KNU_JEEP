import cv2

cap = cv2.VideoCapture("/abr/coss11/repo/robot_data/out76hgk.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")

cap.release()
