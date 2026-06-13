import subprocess
import numpy as np
import cv2
import os
from ISP import *
from ISP.binning import extract_green
from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
from conversion_utils import *

capture_dir = "captures"
os.makedirs(capture_dir, exist_ok=True)
capture_count = 1

W=3840
H=2160
exposure=33000
gain=200
frame_rate=30000000

pR, frame_bytes = initialize_camera_jetson("/dev/video1", W, H, exposure, gain, frame_rate)
pL, _ = initialize_camera_jetson("/dev/video0", W, H, exposure, gain, frame_rate)

while True:
    raw_u16_R = capture_raw_frame_jetson(pR) # 2160×3840
    raw_u16_L = capture_raw_frame_jetson(pL) # 2160×3840
    raw_u16_green_R = extract_green(raw_u16_R) # 1080×1920 with green pixels only
    raw_u16_green_L = extract_green(raw_u16_L) # 1080×1920 with green pixels only

    # Display
    cv2.imshow("raw_u16_green_R", raw_u16_green_R)
    cv2.imshow("raw_u16_green_L", raw_u16_green_L)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        filename = os.path.join(capture_dir, f"raw_u16_{capture_count:02d}.png")
        cv2.imwrite(filename, raw_u16_green_R)
        print(f"[INFO] Captured image saved to {filename}")
        filename = os.path.join(capture_dir, f"raw_u16_green_{capture_count:02d}.png")
        cv2.imwrite(filename, raw_u16_green_L)
        print(f"[INFO] Captured image saved to {filename}")

        capture_count += 1

p.terminate()
cv2.destroyAllWindows()