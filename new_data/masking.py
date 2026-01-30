import cv2
import glob
import os

INPUT_DIR  = "Right"
OUTPUT_DIR = "Right_mask"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in glob.glob(f"{INPUT_DIR}/*.jpg"):
    img = cv2.imread(path)
    if img is None:
        continue

    img[0:117//2, :] = 0

    filename = os.path.basename(path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)

print("All images masked.")
