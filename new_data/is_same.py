import os
import cv2
import numpy as np

DIR = "Right"

def images_are_identical(p1, p2):
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)

    if img1 is None or img2 is None:
        return False
    if img1.shape != img2.shape:
        return False
    return np.array_equal(img1, img2)

deleted = 0
kept = 0

for fname in os.listdir(DIR):
    if not fname.lower().endswith("1.jpg"):
        continue

    orig_name = fname[:-5] + ".jpg"   # xxx1.jpg -> xxx.jpg
    p1 = os.path.join(DIR, fname)
    p0 = os.path.join(DIR, orig_name)

    if not os.path.exists(p0):
        continue

    if images_are_identical(p0, p1):
        os.remove(p1)
        print(f"[DELETE] {fname} (same as {orig_name})")
        deleted += 1
    else:
        print(f"[KEEP  ] {fname} (different from {orig_name})")
        kept += 1

print(f"\nSummary: deleted={deleted}, kept={kept}")
