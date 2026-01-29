import cv2
import os

# 설정
video_path = 'outzxc.mp4'
output_folder = 'output_images'
frame_interval = 30  # 10프레임마다 1장씩 저장 (조절 가능)

# 폴더 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)
count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
        
    # 설정한 간격에 맞을 때만 저장
    if count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        saved_count += 1
        
    count += 1

cap.release()
print("변환 완료!")