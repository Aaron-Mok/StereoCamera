#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time
import sys

# Make parent-of-parent importable (so "from StereoCamera import camera_utils" works)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from StereoCamera import camera_utils

# ----------------------------
# USER SETTINGS
# ----------------------------
LEFT_CAM_INDEX  = 0
RIGHT_CAM_INDEX = 1
RESOLUTION      = (1280, 720)   # requested output size
FPS             = 30
FLIP_FRAMES     = True         # flip both frames 180° (cv2.flip(..., -1))

# Checkerboard settings
# INNER CORNERS (e.g., a 10x7 squares board has 9x6 inner corners)
CB_ROWS = 6
CB_COLS = 9
SQUARE_SIZE_M = 0.024  # meters (e.g., 24 mm)

# Minimum pairs before allowing calibration
MIN_PAIRS = 15

# Output files/folders
OUT_DIR = "stereo_calib_pairs"
YAML_OUT = "stereo_params.yml"

# Detection/Refinement parameters
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
SUBPIX_WIN = (11, 11)
SUBPIX_ZERO_ZONE = (-1, -1)

# ----------------------------
# Helpers
# ----------------------------

def make_object_points(cb_cols, cb_rows, square_size_m):
    """Create the 3D object points for a single checkerboard view (Z=0 plane)."""
    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp

def draw_corners(img, pattern_size, corners, found):
    disp = img.copy()
    cv2.drawChessboardCorners(disp, pattern_size, corners, found)
    return disp

def save_yaml(filepath, data_dict):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    for k, v in data_dict.items():
        fs.write(k, v)
    fs.release()

def grab_frame(picam2, flip=FLIP_FRAMES):
    """Capture a frame from a Picamera2 instance and optionally flip it 180°."""
    frame = picam2.capture_array()
    if frame is None:
        return None
    if flip:
        frame = cv2.flip(frame, -1)
    return frame

# ----------------------------
# Main
# ----------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Initialize both cameras via your helper (assumed to start pipelines)
    left_picam2,  _ = camera_utils.initialize_camera_with_ISP(LEFT_CAM_INDEX,  RESOLUTION)
    right_picam2, _ = camera_utils.initialize_camera_with_ISP(RIGHT_CAM_INDEX, RESOLUTION)

    time.sleep(2.0)
    meta = left_picam2.capture_metadata()
    exp_time = meta.get("ExposureTime", 10000)
    gain     = meta.get("AnalogueGain", 1.0)
    col_gains = meta.get("ColourGains", (1.0, 1.0))
    
    for cam in (left_picam2, right_picam2):
        meta = cam.capture_metadata()
        controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": exp_time,
            "AnalogueGain": gain,
            "ColourGains": col_gains,
        }
        cam.set_controls(controls)

    # Quick sanity grab
    frameL = grab_frame(left_picam2)
    frameR = grab_frame(right_picam2)
    if frameL is None or frameR is None:
        raise RuntimeError("Failed to capture from one or both cameras. Check indices/permissions.")

    pattern_size = (CB_COLS, CB_ROWS)
    objp = make_object_points(CB_COLS, CB_ROWS, SQUARE_SIZE_M)

    objpoints = []   # 3D points in world (per pair)
    imgpointsL = []  # 2D points in left image
    imgpointsR = []  # 2D points in right image

    pair_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("[i] Press 'c' to capture a pair when both detections are green.")
    print("[i] Press 's' to run calibration when you have enough pairs.")
    print("[i] Press 'q' to quit without calibrating.")

    while True:
        # --- Frame grab loop for Picamera2 ---
        frameL = grab_frame(left_picam2)
        frameR = grab_frame(right_picam2)
        if frameL is None or frameR is None:
            print("[!] Frame grab failed")
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Try to find chessboard corners
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)
        foundL, cornersL = cv2.findChessboardCorners(grayL, pattern_size, flags)
        foundR, cornersR = cv2.findChessboardCorners(grayR, pattern_size, flags)

        showL = frameL.copy()
        showR = frameR.copy()

        if foundL:
            cornersL = cv2.cornerSubPix(grayL, cornersL, SUBPIX_WIN, SUBPIX_ZERO_ZONE, TERMINATION_CRITERIA)
            showL = draw_corners(showL, pattern_size, cornersL, True)
        if foundR:
            cornersR = cv2.cornerSubPix(grayR, cornersR, SUBPIX_WIN, SUBPIX_ZERO_ZONE, TERMINATION_CRITERIA)
            showR = draw_corners(showR, pattern_size, cornersR, True)

        # Status overlays
        colorL = (0, 200, 0) if foundL else (0, 0, 255)
        colorR = (0, 200, 0) if foundR else (0, 0, 255)
        cv2.putText(showL, f"Left corners: {'OK' if foundL else 'NO'}", (20, 40), font, 0.8, colorL, 2, cv2.LINE_AA)
        cv2.putText(showR, f"Right corners: {'OK' if foundR else 'NO'}", (20, 40), font, 0.8, colorR, 2, cv2.LINE_AA)
        cv2.putText(showL, f"Pairs: {pair_count}", (20, 80), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(showR, f"Press 'c' to capture", (20, 80), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Show windows
        cv2.imshow("Left", showL)
        cv2.imshow("Right", showR)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[i] Quit without calibration.")
            break

        elif key == ord('c'):
            if foundL and foundR:
                timestamp = int(time.time() * 1000)
                left_path  = os.path.join(OUT_DIR, f"left_{timestamp}.png")
                right_path = os.path.join(OUT_DIR, f"right_{timestamp}.png")
                cv2.imwrite(left_path, frameL)
                cv2.imwrite(right_path, frameR)

                objpoints.append(objp.copy())
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                pair_count += 1
                print(f"[+] Captured pair #{pair_count} -> {left_path}, {right_path}")
            else:
                print("[!] Capture skipped: need detections on BOTH images.")

        elif key == ord('s'):
            if pair_count < MIN_PAIRS:
                print(f"[!] Need at least {MIN_PAIRS} pairs before calibration. Currently: {pair_count}")
                continue

            print("[i] Running per-camera calibration...")
            image_size = (grayL.shape[1], grayL.shape[0])  # (width, height)

            # Calibrate each camera individually (good initial guesses)
            retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
                objpoints, imgpointsL, image_size, None, None
            )
            retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
                objpoints, imgpointsR, image_size, None, None
            )
            print(f"[i] RMS Left: {retL:.4f}, RMS Right: {retR:.4f}")

            # Stereo calibration using initial intrinsics
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
                     cv2.CALIB_RATIONAL_MODEL |
                     cv2.CALIB_FIX_K3)
            # Some OpenCV builds have extra flags like CALIB_FIX_TAUX_TAUY; enable if present:
            if hasattr(cv2, "CALIB_FIX_TAUX_TAUY"):
                flags |= cv2.CALIB_FIX_TAUX_TAUY

            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
            print("[i] Running stereoCalibrate...")
            rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpointsL, imgpointsR,
                K1, D1, K2, D2,
                image_size, criteria=criteria, flags=flags
            )
            print(f"[i] Stereo RMS: {rms:.4f}")
            print(f"[i] Baseline (m): {np.linalg.norm(T):.6f}")

            # Rectification + maps
            print("[i] Computing stereoRectify + maps...")
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

            # Save everything to YAML
            out = {
                "image_width": image_size[0],
                "image_height": image_size[1],
                "K1": K1, "D1": D1, "K2": K2, "D2": D2,
                "R": R, "T": T, "E": E, "F": F,
                "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
                "roi1": np.array(roi1, dtype=np.int32),
                "roi2": np.array(roi2, dtype=np.int32),
                "map1x": map1x, "map1y": map1y, "map2x": map2x, "map2y": map2y,
                "square_size_m": float(SQUARE_SIZE_M),
                "cb_cols": int(CB_COLS),
                "cb_rows": int(CB_ROWS),
            }
            save_yaml(YAML_OUT, out)
            print(f"[✓] Saved calibration & rectification to '{YAML_OUT}'")

            # Optional: quick rectified preview on live feed
            print("[i] Showing rectified preview (press 'q' to quit).")
            while True:
                frL2 = grab_frame(left_picam2)
                frR2 = grab_frame(right_picam2)
                if frL2 is None or frR2 is None:
                    break
                rectL = cv2.remap(frL2, map1x, map1y, cv2.INTER_LINEAR)
                rectR = cv2.remap(frR2, map2x, map2y, cv2.INTER_LINEAR)
                both = cv2.hconcat([rectL, rectR])
                # horizontal guide lines
                h = both.shape[0]
                for y in range(0, h, 60):
                    cv2.line(both, (0, y), (both.shape[1]-1, y), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("Rectified Preview (q to quit)", both)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break  # after preview, exit main loop

    # Cleanup
    cv2.destroyAllWindows()
    try:
        left_picam2.stop()
    except Exception:
        pass
    try:
        right_picam2.stop()
    except Exception:
        pass

if __name__ == "__main__":
    main()