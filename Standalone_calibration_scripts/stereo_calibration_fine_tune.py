#!/usr/bin/env python3
"""
Stereo calibration fine-tuner.

Measures residual vertical epipolar misalignment after stereo rectification
using two independent methods:

  1. 2-D strip phase correlation  (primary — cv2.phaseCorrelate on horizontal strips;
                                    y-component only; each strip handles its own disparity)
  2. Checkerboard corner y-residuals  (secondary — subpixel, only if board is visible)

Image source (choose one):
  --live    Capture fresh pairs from both cameras interactively (c / s / q)
  --pairs   Folder with left_*/right_* (raw) or rectL_*/rectR_* (pre-rectified) PNGs

Saves corrected calibration as  <original>_fine_tuned.yml

Usage:
  python stereo_calibration_fine_tune.py --live
  python stereo_calibration_fine_tune.py --pairs captures/
  python stereo_calibration_fine_tune.py --pairs stereo_calib_pairs/ --pattern 8x5
"""
import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

# ─── Camera settings (--live mode) ───────────────────────────────────────────
LEFT_DEVICE  = "/dev/video0"
RIGHT_DEVICE = "/dev/video1"
W, H         = 3840, 2160
EXPOSURE     = 33000
GAIN         = 200
FRAME_RATE   = 30000000

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CALIB   = "Calibration_output/Stereo_params_20260614_Jetson.yml"
DEFAULT_PAIRS   = "stereo_calib_pairs"
DEFAULT_PATTERN = "8x5"
N_STRIPS        = 7     # horizontal strips for 2D phase correlation
MIN_LIVE_PAIRS  = 5
# ──────────────────────────────────────────────────────────────────────────────


def load_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")

    def mat(k):  return fs.getNode(k).mat()
    def ival(k): return int(fs.getNode(k).real())

    cal = {
        "image_width":  ival("image_width"),
        "image_height": ival("image_height"),
        "K1": mat("K1"), "D1": mat("D1"),
        "K2": mat("K2"), "D2": mat("D2"),
        "R":  mat("R"),  "T":  mat("T"),
        "E":  mat("E"),  "F":  mat("F"),
        "R1": mat("R1"), "R2": mat("R2"),
        "P1": mat("P1"), "P2": mat("P2"),
        "Q":  mat("Q"),
        "roi1": mat("roi1"), "roi2": mat("roi2"),
        "map1x": mat("map1x"), "map1y": mat("map1y"),
        "map2x": mat("map2x"), "map2y": mat("map2y"),
    }
    for key in ("square_size_m", "cb_cols", "cb_rows"):
        node = fs.getNode(key)
        if not node.empty():
            cal[key] = node.real()
    fs.release()
    return cal


def save_calibration(path, cal):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for k, v in cal.items():
        if isinstance(v, np.ndarray):
            fs.write(k, v)
        elif isinstance(v, int):
            fs.write(k, int(v))
        else:
            fs.write(k, float(v))
    fs.release()


def vertical_shift_2d_phase_corr(img1, img2, n_strips=N_STRIPS):
    """
    Measure subpixel vertical shift of img2 relative to img1.

    Divides both images into n_strips horizontal bands and runs cv2.phaseCorrelate
    on each band. Uses only the y-component of each result — the x-component is
    local disparity which varies by depth and is discarded.

    Why strips instead of the full image?
      Disparity is not a global translation — different objects at different depths
      shift by different amounts horizontally. Within a narrow horizontal strip the
      depth is approximately constant, so the strip behaves like a translated patch
      and phaseCorrelate gives a clean, reliable peak.

    A Hanning window per strip reduces spectral leakage at the strip edges.

    Convention: dy > 0 means img2 features appear BELOW img1.
    Correction:  map2y_new = map2y + dy
    """
    H_img, W_img = img1.shape
    strip_h = H_img // n_strips
    win = cv2.createHanningWindow((W_img, strip_h), cv2.CV_64F)

    estimates = []
    for i in range(n_strips):
        y0 = i * strip_h
        y1 = y0 + strip_h
        s1 = img1[y0:y1].astype(np.float64)
        s2 = img2[y0:y1].astype(np.float64)

        shift, response = cv2.phaseCorrelate(s1, s2, win)
        # shift = (dx, dy): dx = local disparity (ignored), dy = vertical error
        estimates.append((shift[1], response))

    # Drop strips with weak correlation (featureless / uniform regions)
    responses = np.array([r for _, r in estimates])
    threshold = np.percentile(responses, 25)   # keep the best 75%
    good = [(dy, r) for dy, r in estimates if r >= threshold] or estimates

    # Response-weighted average across strips
    total_r = sum(r for _, r in good)
    return sum(d * r for d, r in good) / total_r


def corner_y_residuals(rectL, rectR, pattern_size):
    """
    Detect checkerboard corners in both rectified images.
    Returns array of (y_right - y_left) per corner, or None if board not found.
    """
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE  |
             cv2.CALIB_CB_FAST_CHECK)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

    okL, cL = cv2.findChessboardCorners(rectL, pattern_size, flags)
    okR, cR = cv2.findChessboardCorners(rectR, pattern_size, flags)
    if not (okL and okR):
        return None

    cL = cv2.cornerSubPix(rectL, cL, (5, 5), (-1, -1), criteria).reshape(-1, 2)
    cR = cv2.cornerSubPix(rectR, cR, (5, 5), (-1, -1), criteria).reshape(-1, 2)
    return cR[:, 1] - cL[:, 1]


def make_overlay(rectL, rectR, label, line_step=20):
    bgr = cv2.cvtColor(np.hstack([rectL, rectR]), cv2.COLOR_GRAY2BGR)
    for y in range(0, bgr.shape[0], line_step):
        cv2.line(bgr, (0, y), (bgr.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.putText(bgr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 200, 255), 1, cv2.LINE_AA)
    return bgr


def run_live_capture(map1x, map1y, map2x, map2y):
    """
    Open both cameras and show a live rectified preview.
    Press 'c' to capture a pair, 's' to proceed, 'q' to abort.
    Returns list of (rectL, rectR) arrays — already rectified.
    """
    from ISP.binning import extract_green, bin_4x4
    from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
    from conversion_utils import bit16_to_bit8

    def raw_to_gray8(raw):
        return bit16_to_bit8(bin_4x4(extract_green(raw)))

    print("[i] Initialising cameras ...")
    pL, _ = initialize_camera_jetson(LEFT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)
    pR, _ = initialize_camera_jetson(RIGHT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)
    print(f"[i] c=capture  s=run fine-tune (min {MIN_LIVE_PAIRS} pairs)  q=abort")

    pairs = []
    while True:
        raw_L = capture_raw_frame_jetson(pL, W, H)
        raw_R = capture_raw_frame_jetson(pR, W, H)
        if raw_L is None or raw_R is None:
            print("[!] Frame grab failed.")
            break

        rectL = cv2.remap(raw_to_gray8(raw_L), map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(raw_to_gray8(raw_R), map2x, map2y, cv2.INTER_LINEAR)

        label = (f"Pairs: {len(pairs)}  |  c=capture   "
                 f"s=run (need {MIN_LIVE_PAIRS})   q=abort")
        cv2.imshow("Fine-tune — Live Capture", make_overlay(rectL, rectR, label))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            pairs.append((rectL.copy(), rectR.copy()))
            print(f"  [+] Pair #{len(pairs)} captured")
        elif key == ord('s'):
            if len(pairs) < MIN_LIVE_PAIRS:
                print(f"  [!] Need at least {MIN_LIVE_PAIRS} pairs; have {len(pairs)}.")
            else:
                break
        elif key == ord('q'):
            pairs = []
            break

    pL.terminate()
    pR.terminate()
    cv2.destroyAllWindows()
    return pairs   # empty = aborted


def fine_tuned_path(path):
    base, ext = os.path.splitext(path)
    return base + "_fine_tuned" + ext


def main():
    ap = argparse.ArgumentParser(description="Fine-tune stereo calibration via 2D phase correlation.")
    ap.add_argument("--calib",   default=DEFAULT_CALIB,   help="Input .yml calibration file")
    ap.add_argument("--pairs",   default=DEFAULT_PAIRS,   help="Folder with stereo pair images")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="Checkerboard inner corners COLSxROWS")
    ap.add_argument("--live",    action="store_true",     help="Capture fresh pairs from cameras")
    args = ap.parse_args()

    # ── Load calibration ──────────────────────────────────────────────────────
    calib_path = os.path.abspath(args.calib)
    print(f"[i] Loading calibration: {calib_path}")
    cal   = load_calibration(calib_path)
    map1x = cal["map1x"]; map1y = cal["map1y"]
    map2x = cal["map2x"]; map2y = cal["map2y"]

    # ── Collect rectified image pairs ─────────────────────────────────────────
    if args.live:
        image_pairs = run_live_capture(map1x, map1y, map2x, map2y)
        if not image_pairs:
            print("[!] No pairs captured — aborting.")
            sys.exit(1)
    else:
        pairs_dir  = os.path.abspath(args.pairs)
        lefts_raw  = sorted(glob.glob(os.path.join(pairs_dir, "left_*.png")))
        lefts_rect = sorted(glob.glob(os.path.join(pairs_dir, "rectL_*.png")))

        if lefts_raw:
            file_pairs    = [(l, l.replace("left_", "right_")) for l in lefts_raw
                             if os.path.exists(l.replace("left_", "right_"))]
            pre_rectified = False
            print(f"[i] Found {len(file_pairs)} raw pair(s) in '{pairs_dir}' — will remap")
        elif lefts_rect:
            file_pairs    = [(l, l.replace("rectL_", "rectR_")) for l in lefts_rect
                             if os.path.exists(l.replace("rectL_", "rectR_"))]
            pre_rectified = True
            print(f"[i] Found {len(file_pairs)} pre-rectified pair(s) in '{pairs_dir}'")
        else:
            print(f"[!] No left_*.png or rectL_*.png found in '{pairs_dir}'")
            sys.exit(1)

        image_pairs = []
        for lp, rp in file_pairs:
            imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
            if imgL is None or imgR is None:
                continue
            if pre_rectified:
                image_pairs.append((imgL, imgR))
            else:
                image_pairs.append((
                    cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR),
                    cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR),
                ))

    if not image_pairs:
        print("[!] No valid pairs loaded.")
        sys.exit(1)

    # ── Parse checkerboard pattern ────────────────────────────────────────────
    try:
        cb_cols, cb_rows = map(int, args.pattern.lower().split("x"))
        pattern_size = (cb_cols, cb_rows)
    except Exception:
        print(f"[!] Bad --pattern '{args.pattern}'; expected e.g. '8x5'")
        sys.exit(1)

    # ── Measure residual vertical shift across all pairs ─────────────────────
    phase_dy  = []
    corner_dy = []

    for idx, (rectL, rectR) in enumerate(image_pairs):
        dy_ph = vertical_shift_2d_phase_corr(rectL, rectR)
        phase_dy.append(dy_ph)

        residuals = corner_y_residuals(rectL, rectR, pattern_size)
        if residuals is not None:
            mean_c = float(np.mean(residuals))
            corner_dy.append(mean_c)
            print(f"  [{idx+1:02d}] 2D-xcorr dy={dy_ph:+.3f} px | "
                  f"corners dy={mean_c:+.3f} px (n={len(residuals)})")
        else:
            print(f"  [{idx+1:02d}] 2D-xcorr dy={dy_ph:+.3f} px | corners: not detected")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print()
    med_phase = float(np.median(phase_dy))
    print(f"  2D phase-corr  median dy : {med_phase:+.4f} px  (n={len(phase_dy)})")

    if corner_dy:
        med_corners = float(np.median(corner_dy))
        print(f"  Corner         median dy : {med_corners:+.4f} px  (n={len(corner_dy)})")
        final_dy = (med_phase + med_corners) / 2.0
        print(f"  Combined estimate        : {final_dy:+.4f} px  (average of both)")
    else:
        final_dy = med_phase
        print(f"  Using 2D phase-corr only (no checkerboard detections).")

    print(f"\n[→] Correction: right map2y += {final_dy:+.4f} px")
    print(f"    (positive dy = right features were BELOW left → shifts right image UP)")

    # ── Apply correction ──────────────────────────────────────────────────────
    # map2y_new[y,x] = map2y[y,x] + dy
    # The remap samples dy rows lower in the source → output feature moves UP by dy. ✓
    map2y_new = (map2y + np.float32(final_dy)).astype(map2y.dtype)
    map2x_new = map2x.copy()   # x not touched — horizontal offset is disparity

    # ── Save fine-tuned calibration ───────────────────────────────────────────
    out_path = fine_tuned_path(calib_path)
    cal["map2x"] = map2x_new
    cal["map2y"] = map2y_new
    save_calibration(out_path, cal)
    print(f"[✓] Saved → '{out_path}'")

    # ── Visual verification ───────────────────────────────────────────────────
    rectL_ref, rectR_ref = image_pairs[0]
    M = np.float32([[1, 0, 0], [0, 1, -final_dy]])   # shift right image UP by final_dy
    rR_after = cv2.warpAffine(rectR_ref, M, (rectR_ref.shape[1], rectR_ref.shape[0]))

    before = make_overlay(rectL_ref, rectR_ref,  f"BEFORE  (residual={final_dy:+.3f} px)")
    after  = make_overlay(rectL_ref, rR_after,   "AFTER fine-tune")

    cv2.imshow("Fine-tune verification (any key to close)", np.vstack([before, after]))
    print("[i] Features should sit on the same green epipolar line in both L and R halves.")
    print("[i] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
