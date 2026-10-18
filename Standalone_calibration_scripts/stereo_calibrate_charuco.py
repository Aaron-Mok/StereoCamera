#!/usr/bin/env python3
"""
Offline stereo calibration from saved image pairs, using a ChArUco board.

Unlike a plain checkerboard, every corner here carries an id, so a board that
runs off the edge of one camera's frame still contributes everything visible.
That is what lets a wide-baseline rig reach the frame corners at close range,
where the two fields of view no longer fully overlap.

Board geometry and the detection gates live in charuco_common.py.

Usage:
  python3 stereo_calibrate_charuco.py [pairs_dir] [output_yaml]
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import charuco_common as cb
from stereo_calibrate_offline import load_pairs

# ─── Settings ──────────────────────────────────────────────────────────────────
PAIRS_DIR = "stereo_calib_pairs"
YAML_OUT  = "stereo_params.yml"

MIN_PAIRS = 8

# k1,k2 only. The rational model fits the training corners better and rectifies
# worse — extra distortion freedom lets K absorb noise and pose error that then
# shows up as epipolar error, which is the number block matching actually cares
# about. Both stages use the same model so no coefficient is ever frozen at a
# value optimised under a different one.
DIST_FLAGS = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

SAVE_DEBUG = True
# ───────────────────────────────────────────────────────────────────────────────


def detect_all(pairs, board, debug_dir=None):
    """Detect both images of every pair, keeping per-camera and common corners."""
    detector = cb.make_detector(board)

    monoL, monoR, stereo, image_size = [], [], [], None
    rejected = []

    for i, (lp, rp) in enumerate(pairs):
        imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            rejected.append((i + 1, "unreadable"))
            continue

        if image_size is None:
            image_size = (imgL.shape[1], imgL.shape[0])
            print(f"[i] Image size: {image_size[0]}x{image_size[1]}")

        cL, iL = cb.detect(imgL, detector)
        cR, iR = cb.detect(imgR, detector)
        nL = 0 if iL is None else len(iL)
        nR = 0 if iR is None else len(iR)

        # Per-camera calibration uses everything that camera can see; only the
        # stereo stage is restricted to corners both cameras share.
        if iL is not None:
            monoL.append((cb.object_points(iL, board), cL.reshape(-1, 1, 2)))
        if iR is not None:
            monoR.append((cb.object_points(iR, board), cR.reshape(-1, 1, 2)))

        matched = None
        if iL is not None and iR is not None:
            matched = cb.match_pair(cL, iL, cR, iR, board)

        if matched is not None:
            stereo.append(matched)
            print(f"[+] Pair {i+1:3d}/{len(pairs)}: L={nL:2d} R={nR:2d} "
                  f"common={len(matched[0]):2d}")
        else:
            why = "no detection" if (iL is None or iR is None) else "too few/collinear common"
            rejected.append((i + 1, why))
            print(f"[-] Pair {i+1:3d}/{len(pairs)}: L={nL:2d} R={nR:2d} — {why}")

        if debug_dir and iL is not None:
            vis = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedCornersCharuco(vis, cL.reshape(-1, 1, 2),
                                                 iL.reshape(-1, 1), (0, 255, 0))
            cv2.imwrite(os.path.join(debug_dir, f"pair{i+1:03d}_L.jpg"), vis)

    return monoL, monoR, stereo, image_size, rejected


def calibrate_mono(views, image_size, name):
    objp = [o for o, _ in views]
    imgp = [p for _, p in views]
    rms, K, D, _, _ = cv2.calibrateCamera(objp, imgp, image_size, None, None,
                                          flags=DIST_FLAGS)
    print(f"[i] {name}: RMS={rms:.4f}  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    return rms, K, D


def main():
    pairs_dir = sys.argv[1] if len(sys.argv) > 1 else PAIRS_DIR
    yaml_out  = sys.argv[2] if len(sys.argv) > 2 else YAML_OUT

    print(f"[i] Board: {cb.describe()}")
    if cb.MEASURED_SQUARE_M == cb.SQUARE_MM / 1000.0:
        print("[!] MEASURED_SQUARE_M is still the nominal value. Measure five "
              "squares with calipers and set it in charuco_common.py — print "
              "scale error becomes depth error with nothing to warn you.")

    pairs = load_pairs(pairs_dir)
    print(f"[i] Found {len(pairs)} pairs in '{pairs_dir}'")

    debug_dir = None
    if SAVE_DEBUG:
        debug_dir = os.path.join(pairs_dir, "debug_corners")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[i] Saving debug images to '{debug_dir}'")

    board = cb.board_for_calib()
    monoL, monoR, stereo, image_size, rejected = detect_all(pairs, board, debug_dir)

    print(f"\n[i] {len(stereo)} usable pairs, {len(rejected)} rejected")
    if stereo:
        counts = [len(o) for o, _, _ in stereo]
        print(f"[i] Common corners per pair: min={min(counts)} "
              f"median={int(np.median(counts))} max={max(counts)} "
              f"(board has {cb.N_CORNERS})")
    if len(stereo) < MIN_PAIRS:
        print(f"[!] Need at least {MIN_PAIRS} usable pairs. Capture more and retry.")
        return 1

    print("\n[i] Running per-camera calibration ...")
    retL, K1, D1 = calibrate_mono(monoL, image_size, "left ")
    retR, K2, D2 = calibrate_mono(monoR, image_size, "right")

    objpoints  = [o for o, _, _ in stereo]
    imgpointsL = [l for _, l, _ in stereo]
    imgpointsR = [r for _, _, r in stereo]

    print("[i] Running stereoCalibrate ...")
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, image_size,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | DIST_FLAGS)

    baseline = float(np.linalg.norm(T))
    print(f"[i] Stereo RMS : {rms:.4f}   (mono {retL:.4f} / {retR:.4f}, "
          f"ratio {rms/max(retL, retR):.2f})")
    print(f"[i] Baseline   : {baseline*100:.2f} cm")
    print(f"[i] T          : [{T[0,0]:+.4f} {T[1,0]:+.4f} {T[2,0]:+.4f}] m")
    if T[0, 0] > 0:
        print("[!] T[0] is positive — the left and right cameras are swapped. "
              "StereoBM only searches leftward, so disparity will be noise. "
              "Swap LEFT_DEVICE/RIGHT_DEVICE and recapture.")

    print("[i] Computing stereoRectify + remap tables ...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
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
        "board_type":    "charuco",
        "square_size_m": float(cb.MEASURED_SQUARE_M),
        "marker_size_m": float(cb.MEASURED_SQUARE_M * cb.MARKER_RATIO),
        "squares_x":     int(cb.SQUARES_X),
        "squares_y":     int(cb.SQUARES_Y),
        "aruco_dict":    cb.DICT_NAME,
    }
    fs = cv2.FileStorage(yaml_out, cv2.FILE_STORAGE_WRITE)
    for k, v in out.items():
        fs.write(k, v)
    fs.release()
    print(f"[✓] Saved to '{yaml_out}'")
    print(f"\n    Next: python3 Standalone_calibration_scripts/validate_calibration.py "
          f"{yaml_out} {pairs_dir}")
    print("    Do not use the calibration until rectified epipolar RMS < 0.3 px.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
