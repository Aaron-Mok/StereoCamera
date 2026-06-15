#!/usr/bin/env python3
"""
Stereo calibration script for dual Pi HQ cameras on Jetson Nano.
Cameras are read as raw Bayer uint16 via v4l2-ctl; green channel is
extracted for calibration since it has the most pixels and sharpest detail.

Controls:
  c  - capture a pair (only works when both cameras detect the checkerboard)
  s  - run stereo calibration with collected pairs
  q  - quit without calibrating
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import time

from ISP.binning import extract_green, bin_4x4
from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
from conversion_utils import bit16_to_bit8

# ─── Settings ──────────────────────────────────────────────────────────────────
LEFT_DEVICE  = "/dev/video0"
RIGHT_DEVICE = "/dev/video1"

W, H         = 3840, 2160          # sensor resolution
EXPOSURE     = 33000
GAIN         = 200
FRAME_RATE   = 30000000

# After extract_green: (H//2, W//2) = (1080, 1920)
IMG_W, IMG_H = W // 8, H // 8  # after 4x4 binning for corner detection

# Checkerboard inner-corner count (not number of squares)
CB_COLS       = 8
CB_ROWS       = 5
SQUARE_SIZE_M = 0.025              # square side length in metres

MIN_PAIRS = 15

OUT_DIR  = "stereo_calib_pairs"
YAML_OUT = "stereo_params.yml"

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
SUBPIX_WIN       = (11, 11)
SUBPIX_ZERO_ZONE = (-1, -1)
# ───────────────────────────────────────────────────────────────────────────────


def raw_to_gray8(raw_u16: np.ndarray) -> np.ndarray:
    """Extract green channel and convert to uint8 for display / detection."""
    green = extract_green(raw_u16)  # still uint16, but only green pixels
    return bit16_to_bit8(bin_4x4(green))  # simple 4x4 binning to reduce noise and speed up corner detection


def make_object_points(pattern_size, square_size_m):
    cb_cols, cb_rows = pattern_size
    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def save_yaml(filepath, data_dict):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    for k, v in data_dict.items():
        fs.write(k, v)
    fs.release()


_display_frame = None

def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and _display_frame is not None:
        val = _display_frame[y, x]
        print(f"({x}, {y}) = {val}    ", end="\r")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[i] Initialising left camera  ...")
    pL, _ = initialize_camera_jetson(LEFT_DEVICE,  W, H, EXPOSURE, GAIN, FRAME_RATE)
    print("[i] Initialising right camera ...")
    pR, _ = initialize_camera_jetson(RIGHT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)

    pattern_size = (CB_COLS, CB_ROWS)
    objp = make_object_points(pattern_size, SQUARE_SIZE_M)
    image_size   = (IMG_W, IMG_H)   # (width, height) as OpenCV expects

    objpoints  = []
    imgpointsL = []
    imgpointsR = []
    pair_count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE  |
                cv2.CALIB_CB_FAST_CHECK)

    print("[i] Press 'c' to capture a pair when both detections are green.")
    print("[i] Press 's' to run calibration (need at least", MIN_PAIRS, "pairs).")
    print("[i] Press 'q' to quit.")

    # cv2.setMouseCallback("Stereo", _mouse_cb)

    while True:
        raw_L = capture_raw_frame_jetson(pL, W, H)
        raw_R = capture_raw_frame_jetson(pR, W, H)
        if raw_L is None or raw_R is None:
            print("[!] Frame grab failed — exiting.")
            break

        grayL = raw_to_gray8(raw_L)
        grayR = raw_to_gray8(raw_R)

        # Detect on half-size image — much faster, scale corners back up
        foundL, cornersL = cv2.findChessboardCorners(grayL, pattern_size, cb_flags)
        foundR, cornersR = cv2.findChessboardCorners(grayR, pattern_size, cb_flags)

        if foundL:
            cornersL = cv2.cornerSubPix(grayL, cornersL, SUBPIX_WIN, SUBPIX_ZERO_ZONE,
                                        TERMINATION_CRITERIA)
        if foundR:
            cornersR = cv2.cornerSubPix(grayR, cornersR, SUBPIX_WIN, SUBPIX_ZERO_ZONE,
                                        TERMINATION_CRITERIA)

        showL = grayL
        showR = grayR

        if foundL:
            cv2.drawChessboardCorners(showL, pattern_size, cornersL, foundL)
        if foundR:
            cv2.drawChessboardCorners(showR, pattern_size, cornersR, foundR)

        colorL = (0, 200, 0) if foundL else (0, 0, 255)
        colorR = (0, 200, 0) if foundR else (0, 0, 255)
        cv2.putText(showL, f"Left:  {'OK' if foundL else 'NO'}",  (20, 40),  font, 1.0, colorL, 2, cv2.LINE_AA)
        cv2.putText(showR, f"Right: {'OK' if foundR else 'NO'}", (20, 40),  font, 1.0, colorR, 2, cv2.LINE_AA)
        cv2.putText(showL, f"Pairs: {pair_count}",               (20, 80),  font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(showR, f"c=capture  s=calibrate  q=quit",    (20, 80),  font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Left", showL)
        cv2.imshow("Right", showR)


        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[i] Quit — no calibration saved.")
            break

        elif key == ord('c'):
            if foundL and foundR:
                ts = int(time.time() * 1000)
                lp = os.path.join(OUT_DIR, f"left_{ts}.png")
                rp = os.path.join(OUT_DIR, f"right_{ts}.png")
                cv2.imwrite(lp, grayL)
                cv2.imwrite(rp, grayR)
                objpoints.append(objp.copy())
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                pair_count += 1
                print(f"[+] Pair #{pair_count} saved → {lp}, {rp}")
            else:
                print("[!] Skipped — need checkerboard visible in BOTH cameras.")

        elif key == ord('s'):
            if pair_count < MIN_PAIRS:
                print(f"[!] Need at least {MIN_PAIRS} pairs; have {pair_count}.")
                continue

            print("[i] Running per-camera calibration ...")
            retL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
            retR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)
            print(f"[i] RMS  left={retL:.4f}  right={retR:.4f}")

            sc_flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                        cv2.CALIB_RATIONAL_MODEL       |
                        cv2.CALIB_FIX_K3)
            if hasattr(cv2, "CALIB_FIX_TAUX_TAUY"):
                sc_flags |= cv2.CALIB_FIX_TAUX_TAUY

            sc_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)

            print("[i] Running stereoCalibrate ...")
            rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpointsL, imgpointsR,
                K1, D1, K2, D2,
                image_size, criteria=sc_criteria, flags=sc_flags
            )
            print(f"[i] Stereo RMS : {rms:.4f}")
            print(f"[i] Baseline   : {np.linalg.norm(T)*100:.2f} cm")

            print("[i] Computing stereoRectify + remap tables ...")
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

            out = {
                "image_width":  image_size[0],
                "image_height": image_size[1],
                "K1": K1, "D1": D1,
                "K2": K2, "D2": D2,
                "R":  R,  "T":  T,  "E": E, "F": F,
                "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
                "roi1": np.array(roi1, dtype=np.int32),
                "roi2": np.array(roi2, dtype=np.int32),
                "map1x": map1x, "map1y": map1y,
                "map2x": map2x, "map2y": map2y,
                "square_size_m": float(SQUARE_SIZE_M),
                "cb_cols": int(CB_COLS),
                "cb_rows": int(CB_ROWS),
            }
            save_yaml(YAML_OUT, out)
            print(f"[✓] Saved to '{YAML_OUT}'")

            # ── Rectified live preview ──────────────────────────────────────
            print("[i] Rectified preview — press 'q' to exit.")
            while True:
                raw_L2 = capture_raw_frame_jetson(pL, W, H)
                raw_R2 = capture_raw_frame_jetson(pR, W, H)
                if raw_L2 is None or raw_R2 is None:
                    break
                gL2 = raw_to_gray8(raw_L2)
                gR2 = raw_to_gray8(raw_R2)
                rectL = cv2.remap(gL2, map1x, map1y, cv2.INTER_LINEAR)
                rectR = cv2.remap(gR2, map2x, map2y, cv2.INTER_LINEAR)
                both  = cv2.hconcat([
                    cv2.cvtColor(rectL, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(rectR, cv2.COLOR_GRAY2BGR),
                ])
                for y in range(0, both.shape[0], 60):
                    cv2.line(both, (0, y), (both.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("Rectified Preview (q to quit)", both)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break

    cv2.destroyAllWindows()
    pL.terminate()
    pR.terminate()


if __name__ == "__main__":
    main()
