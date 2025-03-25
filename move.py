import os
import shutil

src_dir = "data/test/unknown"
dst_dir = "data/test"

for fname in os.listdir(src_dir):
    shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

# 最後把空資料夾刪掉
os.rmdir(src_dir)
