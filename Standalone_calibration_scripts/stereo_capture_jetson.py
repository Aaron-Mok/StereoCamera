#!/usr/bin/env python3
"""
Capture stereo image pairs for offline calibration.
Live preview at 960×540 for framing; saves full 1920×1080 green-channel images.

Controls:
  c - start 10 captures (10-second lead-in, then every 5 seconds)
  q - quit
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import time
import math
from datetime import datetime

from ISP.binning import extract_green
from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
from conversion_utils import bit16_to_bit8

# ─── Settings ──────────────────────────────────────────────────────────────────
LEFT_DEVICE  = "/dev/video0"
RIGHT_DEVICE = "/dev/video1"

W, H       = 3840, 2160
EXPOSURE   = 33000
GAIN       = 200
FRAME_RATE = 30000000

OUT_DIR = "stereo_calib_pairs"
BATCH_SIZE = 10
START_DELAY_S = 10
CAPTURE_INTERVAL_S = 5
# ───────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[i] Initialising cameras ...")
    pL, _ = initialize_camera_jetson(LEFT_DEVICE,  W, H, EXPOSURE, GAIN, FRAME_RATE)
    pR, _ = initialize_camera_jetson(RIGHT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)

    print("[i] Press 'c' for 10 pairs: first in 10 seconds, then every 5 seconds; 'q' quits.")
    pair_count = 0
    batch_count = 0
    next_capture = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        raw_L = capture_raw_frame_jetson(pL, W, H)
        raw_R = capture_raw_frame_jetson(pR, W, H)
        if raw_L is None or raw_R is None:
            print("[!] Frame grab failed.")
            break

        # Full 1920×1080 green channel — saved on capture
        greenL = bit16_to_bit8(extract_green(raw_L))
        greenR = bit16_to_bit8(extract_green(raw_R))

        now = time.monotonic()
        if next_capture is not None and now >= next_capture:
            # Use one local date/time suffix for both images in the pair.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            lp = os.path.join(OUT_DIR, f"left_{ts}.png")
            rp = os.path.join(OUT_DIR, f"right_{ts}.png")
            saved_left = cv2.imwrite(lp, greenL)
            saved_right = cv2.imwrite(rp, greenR)
            if not (saved_left and saved_right):
                print("[!] Failed to save a complete pair; batch stopped.")
                next_capture = None
            else:
                pair_count += 1
                batch_count += 1
                print(f"[+] Batch {batch_count}/{BATCH_SIZE}, total {pair_count}: {lp}, {rp}")
                if batch_count == BATCH_SIZE:
                    next_capture = None
                    print("[i] Batch complete. Press 'c' to start another batch.")
                else:
                    next_capture = time.monotonic() + CAPTURE_INTERVAL_S

        # Downscale to 960×540 on GPU for live preview
        gpu_L = cv2.cuda_GpuMat(); gpu_L.upload(greenL)
        gpu_R = cv2.cuda_GpuMat(); gpu_R.upload(greenR)
        prevL = cv2.cuda.resize(gpu_L, (960, 540)).download()
        prevR = cv2.cuda.resize(gpu_R, (960, 540)).download()

        # Cameras are mounted upside down; rotate only the live previews.
        prevL = cv2.rotate(prevL, cv2.ROTATE_180)
        prevR = cv2.rotate(prevR, cv2.ROTATE_180)

        cv2.putText(prevL, f"Pairs: {pair_count}", (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if next_capture is None:
            status = "c=start 10 captures  q=quit"
        else:
            remaining = max(0, math.ceil(next_capture - time.monotonic()))
            status = f"Batch {batch_count}/{BATCH_SIZE} saved - next in {remaining}s  q=quit"
        for preview in (prevL, prevR):
            cv2.putText(preview, status, (20, 80), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imshow("Left",  prevL)
        cv2.imshow("Right", prevR)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and next_capture is None:
            batch_count = 0
            next_capture = time.monotonic() + START_DELAY_S
            print("[i] Batch started. First capture in 10 seconds.")

    cv2.destroyAllWindows()
    pL.terminate()
    pR.terminate()


if __name__ == "__main__":
    main()
