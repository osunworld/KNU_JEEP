import os
import shutil

SRC_DIR = "R"
DST_DIR = "Right"

os.makedirs(DST_DIR, exist_ok=True)

for filename in os.listdir(SRC_DIR):
    if not filename.lower().endswith(".jpg"):
        continue

    src_path = os.path.join(SRC_DIR, filename)
    dst_path = os.path.join(DST_DIR, filename)

    if os.path.exists(dst_path):
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}1{ext}"
        dst_path = os.path.join(DST_DIR, new_filename)

    shutil.move(src_path, dst_path)
    print(f"Moved: {src_path} -> {dst_path}")
