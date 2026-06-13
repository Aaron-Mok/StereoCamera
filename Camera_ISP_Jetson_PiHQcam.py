import subprocess
import numpy as np
import cv2
import os
from ISP import *
from ISP.binning import bin_bayer_2x2
from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
from conversion_utils import *

capture_dir = "captures"
os.makedirs(capture_dir, exist_ok=True)
capture_count = 1

p, frame_bytes = initialize_camera_jetson(device="/dev/video1", W=3840, H=2160, exposure=33000, gain=200, frame_rate=30000000)

while True:
    raw_u16 = capture_raw_frame_jetson(p)
    raw_u16_binned = bin_bayer_2x2(raw_u16)                                   # 1080×1920

    # Display
    cv2.imshow("raw_u16", raw_u16_binned)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        filename = os.path.join(capture_dir, f"raw_u16_{capture_count:02d}.png")
        cv2.imwrite(filename, raw_u16)
        print(f"[INFO] Captured image saved to {filename}")

        capture_count += 1

p.terminate()
cv2.destroyAllWindows()
