# debug_read.py
import sys
import cv2
from pathlib import Path

p = Path(sys.argv[1])
img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
print("Path:", p)
print("Exists:", p.exists())
print("OpenCV readable?:", img is not None)
if img is not None:
    print("Shape:", img.shape, "dtype:", img.dtype)
