#!/usr/bin/env python3
"""
Offline stereo calibration from saved image pairs.
Runs findChessboardCornersSB at full 1920×1080 — no speed constraint.

Usage:
  python3 stereo_calibrate_offline.py [pairs_dir] [output_yaml]
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import glob

# ─── Settings ──────────────────────────────────────────────────────────────────
CB_COLS       = 8
CB_ROWS       = 5
SQUARE_SIZE_M = 0.025

PAIRS_DIR = "stereo_calib_pairs"
YAML_OUT  = "stereo_params.yml"

MIN_PAIRS = 5
# ───────────────────────────────────────────────────────────────────────────────


def load_pairs(pairs_dir):
    lefts  = sorted(glob.glob(os.path.join(pairs_dir, "left_*.png")))
    rights = sorted(glob.glob(os.path.join(pairs_dir, "right_*.png")))
    if len(lefts) != len(rights):
        raise ValueError(f"Mismatched pairs: {len(lefts)} left, {len(rights)} right")
    return list(zip(lefts, rights))


def make_object_points(pattern_size, square_size_m):
    cb_cols, cb_rows = pattern_size
    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def find_corners(gray, pattern_size, save_path=None):
    """Two-stage detection:
    1. findChessboardCorners (robust to lighting) to locate the board.
    2. Mask out background with mid-gray, then run findChessboardCornersSB
       for accurate subpixel corners without background interference.
    Corners are returned in full input resolution coordinates.
    """
    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE  |
                cv2.CALIB_CB_FAST_CHECK)
    found, corners_rough = cv2.findChessboardCorners(gray, pattern_size, cb_flags)
    if not found:
        return False, None

    # Bounding box of the board with padding
    x, y, w, h = cv2.boundingRect(cv2.convexHull(corners_rough))
    pad = max(w, h) // 6
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)

    # Mid-gray background so the board edges don't create false saddle points
    masked = np.full_like(gray, 128)
    masked[y1:y2, x1:x2] = gray[y1:y2, x1:x2]

    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    found_sb, corners_sb = cv2.findChessboardCornersSB(masked, pattern_size, sb_flags)

    if save_path is not None:
        # Save masked image with corners drawn at full 1920×1080 resolution
        vis = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        if found_sb:
            cv2.drawChessboardCorners(vis, pattern_size, corners_sb, found_sb)
        status = "SB OK" if found_sb else "SB FAILED"
        cv2.putText(vis, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0) if found_sb else (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite(save_path, vis)

    return found_sb, corners_sb


def save_yaml(filepath, data_dict):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    for k, v in data_dict.items():
        fs.write(k, v)
    fs.release()


def main():
    pairs_dir = sys.argv[1] if len(sys.argv) > 1 else PAIRS_DIR
    yaml_out  = sys.argv[2] if len(sys.argv) > 2 else YAML_OUT

    pairs = load_pairs(pairs_dir)
    print(f"[i] Found {len(pairs)} pairs in '{pairs_dir}'")

    debug_dir = os.path.join(pairs_dir, "debug_corners")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"[i] Saving debug images to '{debug_dir}'")

    pattern_size = (CB_COLS, CB_ROWS)
    objp = make_object_points(pattern_size, SQUARE_SIZE_M)

    objpoints  = []
    imgpointsL = []
    imgpointsR = []
    image_size = None
    rejected   = 0

    for i, (lp, rp) in enumerate(pairs):
        imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            print(f"[!] Could not read pair {i+1}, skipping.")
            rejected += 1
            continue

        if image_size is None:
            image_size = (imgL.shape[1], imgL.shape[0])
            print(f"[i] Image size: {image_size[0]}×{image_size[1]}")

        foundL, cornersL = find_corners(imgL, pattern_size, save_path=os.path.join(debug_dir, f"pair{i+1:03d}_L.jpg"))
        foundR, cornersR = find_corners(imgR, pattern_size, save_path=os.path.join(debug_dir, f"pair{i+1:03d}_R.jpg"))

        if foundL and foundR:
            objpoints.append(objp.copy())
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            print(f"[+] Pair {i+1:3d}/{len(pairs)}: OK")
        else:
            status = f"L={'OK' if foundL else 'NO'}  R={'OK' if foundR else 'NO'}"
            print(f"[-] Pair {i+1:3d}/{len(pairs)}: {status} — skipped")
            rejected += 1

    print(f"\n[i] {len(objpoints)} valid pairs, {rejected} rejected")

    if len(objpoints) < MIN_PAIRS:
        print(f"[!] Need at least {MIN_PAIRS} valid pairs. Capture more images and retry.")
        return

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
    save_yaml(yaml_out, out)
    print(f"[✓] Saved to '{yaml_out}'")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
