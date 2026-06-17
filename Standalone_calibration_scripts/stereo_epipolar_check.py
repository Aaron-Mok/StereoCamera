#!/usr/bin/env python3
"""
Stereo epipolar residual error checker.

Rectifies stereo image pairs and measures the residual vertical (y) epipolar
misalignment spatially across the image using sparse Lucas-Kanade optical flow.

Outputs:
  1. Rectified side-by-side with epipolar lines + per-pair stats
  2. Spatial grid heatmap of local median y-shift per image region
  3. Histogram of per-feature dy values
  4. Diagnosis: uniform shift vs. spatially varying (edge) shift

Interpretation:
  Uniform grid  → physical camera offset or small R/T error; map2y += dy fixes it
  Varies at edges → distortion model overfitting; recalibrate with simpler flags

Usage:
  python stereo_epipolar_check.py --live
  python stereo_epipolar_check.py --pairs stereo_calib_pairs/
  python stereo_epipolar_check.py --pairs stereo_calib_pairs/ --no-plot
"""
import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

# ─── Camera settings (--live mode) ────────────────────────────────────────────
LEFT_DEVICE  = "/dev/video0"
RIGHT_DEVICE = "/dev/video1"
W, H         = 3840, 2160
EXPOSURE     = 33000
GAIN         = 200
FRAME_RATE   = 30000000

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CALIB = "Calibration_output/Stereo_params_20260614_Jetson.yml"
DEFAULT_PAIRS = "stereo_calib_pairs"

GRID_COLS          = 6     # spatial grid columns
GRID_ROWS          = 4     # spatial grid rows
MIN_CELL_MATCHES   = 5     # minimum features to trust a grid cell
MAX_FEATURES       = 800
OUTLIER_THRESHOLD  = 5.0   # px — discard gross optical flow errors
# ──────────────────────────────────────────────────────────────────────────────


def load_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration: {path}")
    def mat(k):  return fs.getNode(k).mat()
    def ival(k): return int(fs.getNode(k).real())
    cal = {
        "image_width":  ival("image_width"),
        "image_height": ival("image_height"),
        "map1x": mat("map1x"), "map1y": mat("map1y"),
        "map2x": mat("map2x"), "map2y": mat("map2y"),
    }
    fs.release()
    return cal


def measure_y_shifts(rectL, rectR):
    """
    Track Shi-Tomasi corners from rectL into rectR with Lucas-Kanade.
    Returns float32 array of shape (N, 3): [x_left, y_left, dy]
    where dy = y_right - y_left  (positive = right feature is below left).
    """
    pts = cv2.goodFeaturesToTrack(
        rectL, maxCorners=MAX_FEATURES,
        qualityLevel=0.01, minDistance=8, blockSize=5,
    )
    if pts is None or len(pts) < 10:
        return np.empty((0, 3), dtype=np.float32)

    pts_r, status, _ = cv2.calcOpticalFlowPyrLK(
        rectL, rectR, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    good = status.ravel() == 1
    pL = pts[good, 0]
    pR = pts_r[good, 0]
    dy = pR[:, 1] - pL[:, 1]

    inliers = np.abs(dy) < OUTLIER_THRESHOLD
    return np.column_stack([pL[inliers, 0], pL[inliers, 1], dy[inliers]]).astype(np.float32)


def build_spatial_grid(shifts, img_shape):
    """
    Bin shift measurements into a GRID_ROWS × GRID_COLS grid.
    Returns array of median dy per cell; NaN where fewer than MIN_CELL_MATCHES points.
    """
    H_img, W_img = img_shape
    grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    for r in range(GRID_ROWS):
        y0 = r       * H_img // GRID_ROWS
        y1 = (r + 1) * H_img // GRID_ROWS
        for c in range(GRID_COLS):
            x0 = c       * W_img // GRID_COLS
            x1 = (c + 1) * W_img // GRID_COLS
            mask = (
                (shifts[:, 0] >= x0) & (shifts[:, 0] < x1) &
                (shifts[:, 1] >= y0) & (shifts[:, 1] < y1)
            )
            cell = shifts[mask, 2]
            if len(cell) >= MIN_CELL_MATCHES:
                grid[r, c] = float(np.median(cell))
    return grid


def draw_epipolar_overlay(rectL, rectR, dy_vals, line_step=40):
    bgr = cv2.cvtColor(np.hstack([rectL, rectR]), cv2.COLOR_GRAY2BGR)
    for y in range(0, bgr.shape[0], line_step):
        cv2.line(bgr, (0, y), (bgr.shape[1] - 1, y), (0, 200, 0), 1)
    med = float(np.median(dy_vals)) if len(dy_vals) else 0.0
    std = float(np.std(dy_vals))    if len(dy_vals) else 0.0
    cv2.putText(bgr,
                f"median dy={med:+.3f} px   std={std:.3f} px   n={len(dy_vals)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
    return bgr


def draw_grid_overlay(rectL, grid):
    """
    Draw the spatial grid directly onto the left rectified image for display
    without requiring matplotlib.
    """
    vis = cv2.cvtColor(rectL, cv2.COLOR_GRAY2BGR)
    H_img, W_img = rectL.shape

    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        return vis
    vmax = max(abs(valid.max()), abs(valid.min()), 0.3)

    for r in range(GRID_ROWS):
        y0 = r       * H_img // GRID_ROWS
        y1 = (r + 1) * H_img // GRID_ROWS
        for c in range(GRID_COLS):
            x0 = c       * W_img // GRID_COLS
            x1 = (c + 1) * W_img // GRID_COLS
            val = grid[r, c]
            if np.isnan(val):
                color = (60, 60, 60)
                label = "N/A"
            else:
                # Blue = negative (right above left), Red = positive (right below left)
                t = np.clip(val / vmax, -1.0, 1.0)
                if t >= 0:
                    color = (0, int(80 * (1 - t)), int(220 * t))   # dark→red
                else:
                    color = (int(220 * -t), int(80 * (1 + t)), 0)  # dark→blue
                label = f"{val:+.2f}"
            cv2.rectangle(vis, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), color, -1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (180, 180, 180), 1)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.putText(vis, label, (cx - 28, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(vis, "Spatial y-shift grid (px)  blue=right above  red=right below",
                (6, H_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return vis


def plot_results(grid, all_dy, label, use_matplotlib):
    if not use_matplotlib:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not available — skipping plot (use --no-plot to suppress this warning)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Epipolar residual — {label}", fontsize=13)

    valid = grid[~np.isnan(grid)]
    vmax  = max(abs(valid.max()), abs(valid.min()), 0.3) if len(valid) else 1.0
    im = axes[0].imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title(
        "Spatial y-shift grid (px)\n"
        "Uniform → mount/R/T offset   |   Varies at edges → distortion overfit"
    )
    axes[0].set_xlabel("Image column region  →")
    axes[0].set_ylabel("Image row region  ↓")
    plt.colorbar(im, ax=axes[0], label="dy (px)")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if not np.isnan(grid[r, c]):
                axes[0].text(c, r, f"{grid[r, c]:+.2f}",
                             ha="center", va="center", fontsize=9)

    axes[1].hist(all_dy, bins=50, color="steelblue", edgecolor="white")
    axes[1].axvline(float(np.median(all_dy)), color="red", linestyle="--",
                    label=f"median = {np.median(all_dy):+.3f} px")
    axes[1].axvline(0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title("Per-feature dy distribution")
    axes[1].set_xlabel("dy (px)  [right − left]")
    axes[1].set_ylabel("Feature count")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def analyse_pair(rectL, rectR, pair_idx, use_matplotlib):
    shifts = measure_y_shifts(rectL, rectR)
    if len(shifts) < 10:
        print(f"  [{pair_idx+1:02d}] Too few features — skipping.")
        return None

    dy_all = shifts[:, 2]
    med    = float(np.median(dy_all))
    std    = float(np.std(dy_all))
    print(f"  [{pair_idx+1:02d}] features={len(dy_all):4d}  "
          f"median_dy={med:+.4f} px  std={std:.4f} px")

    grid    = build_spatial_grid(shifts, rectL.shape)
    overlay = draw_epipolar_overlay(rectL, rectR, dy_all)
    grid_vis = draw_grid_overlay(rectL, grid)

    display = np.vstack([
        overlay,
        np.hstack([
            grid_vis,
            cv2.cvtColor(rectR, cv2.COLOR_GRAY2BGR),
        ])
    ]) if overlay.shape[1] == grid_vis.shape[1] + rectR.shape[1] else overlay

    # Resize for display if too large
    dh, dw = overlay.shape[:2]
    if dw > 1800:
        scale  = 1800 / dw
        overlay   = cv2.resize(overlay,   (int(dw * scale), int(dh * scale)))
        grid_vis  = cv2.resize(grid_vis,  (int(rectL.shape[1] * scale), int(rectL.shape[0] * scale)))

    cv2.imshow(f"Epipolar pair {pair_idx+1}  — epipolar lines (any key to continue)", overlay)
    cv2.waitKey(0)
    cv2.imshow(f"Epipolar pair {pair_idx+1}  — spatial y-shift grid (any key to continue)", grid_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_results(grid, dy_all, f"pair {pair_idx+1}", use_matplotlib)
    return grid, med, dy_all


def diagnose(combined_grid, all_medians):
    print()
    print("─── Diagnosis ───────────────────────────────────────────────────────")
    overall_dy = float(np.median(all_medians))
    print(f"  Overall median dy     : {overall_dy:+.4f} px")

    valid = combined_grid[~np.isnan(combined_grid)]
    if len(valid) < 2:
        print("  Not enough grid cells to assess spatial variation.")
        return

    spatial_range = float(np.nanmax(combined_grid) - np.nanmin(combined_grid))
    spatial_std   = float(np.nanstd(combined_grid))
    print(f"  Spatial range (max-min): {spatial_range:.4f} px across grid")
    print(f"  Spatial std             : {spatial_std:.4f} px across grid")
    print()

    if spatial_range < 0.4:
        print("  ✓ Shift is UNIFORM across the image.")
        print("    → The fine-tune script (map2y offset) will fix this cleanly.")
        print(f"    → Expected correction: map2y += {overall_dy:+.4f} px")
    elif spatial_range < 1.0:
        print("  ~ Shift has mild spatial variation.")
        print("    → Fine-tuning will help but may not fully eliminate the error.")
        print("    → Consider recalibrating with:  CALIB_USE_INTRINSIC_GUESS | CALIB_FIX_K3")
        print("      (remove CALIB_RATIONAL_MODEL to avoid distortion overfitting)")
    else:
        print("  ✗ Shift varies SIGNIFICANTLY across the image.")
        print("    → This is a distortion model problem, not a simple offset.")
        print("    → Fine-tuning will not fix this properly.")
        print("    → Recalibrate stereo_calibrate_jetson.py without CALIB_RATIONAL_MODEL.")
    print("─────────────────────────────────────────────────────────────────────")


def run_live_capture(map1x, map1y, map2x, map2y, n_pairs):
    from ISP.binning import extract_green, bin_4x4
    from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
    from conversion_utils import bit16_to_bit8

    def raw_to_gray8(raw):
        return bit16_to_bit8(bin_4x4(extract_green(raw)))

    print("[i] Initialising cameras ...")
    pL, _ = initialize_camera_jetson(LEFT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)
    pR, _ = initialize_camera_jetson(RIGHT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)
    print(f"[i] c=capture (need {n_pairs})   q=abort")

    pairs = []
    while len(pairs) < n_pairs:
        rawL = capture_raw_frame_jetson(pL, W, H)
        rawR = capture_raw_frame_jetson(pR, W, H)
        if rawL is None or rawR is None:
            print("[!] Frame grab failed.")
            break

        rectL = cv2.remap(raw_to_gray8(rawL), map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(raw_to_gray8(rawR), map2x, map2y, cv2.INTER_LINEAR)

        preview = cv2.cvtColor(np.hstack([rectL, rectR]), cv2.COLOR_GRAY2BGR)
        dw = preview.shape[1]
        if dw > 1800:
            scale   = 1800 / dw
            preview = cv2.resize(preview, (1800, int(preview.shape[0] * scale)))
        for y in range(0, preview.shape[0], 40):
            cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 200, 0), 1)
        cv2.putText(preview,
                    f"Captured {len(pairs)}/{n_pairs}   c=capture   q=abort",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.imshow("Live rectified preview", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            pairs.append((rectL.copy(), rectR.copy()))
            print(f"  [+] Pair {len(pairs)}/{n_pairs} captured")
        elif key == ord('q'):
            break

    pL.terminate()
    pR.terminate()
    cv2.destroyAllWindows()
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Check stereo epipolar residual error.")
    ap.add_argument("--calib",    default=DEFAULT_CALIB, help="Calibration .yml file")
    ap.add_argument("--pairs",    default=DEFAULT_PAIRS, help="Folder with stereo pairs")
    ap.add_argument("--live",     action="store_true",   help="Capture live from cameras")
    ap.add_argument("--n",        type=int, default=3,   help="Pairs to capture in --live mode")
    ap.add_argument("--no-plot",  action="store_true",   help="Skip matplotlib plots")
    args = ap.parse_args()

    use_matplotlib = not args.no_plot

    # ── Load calibration ──────────────────────────────────────────────────────
    calib_path = os.path.abspath(args.calib)
    print(f"[i] Loading calibration: {calib_path}")
    cal   = load_calibration(calib_path)
    map1x = cal["map1x"]; map1y = cal["map1y"]
    map2x = cal["map2x"]; map2y = cal["map2y"]

    # ── Collect pairs ─────────────────────────────────────────────────────────
    if args.live:
        image_pairs = run_live_capture(map1x, map1y, map2x, map2y, args.n)
    else:
        pairs_dir   = os.path.abspath(args.pairs)
        lefts_rect  = sorted(glob.glob(os.path.join(pairs_dir, "rectL_*.png")))
        lefts_raw   = sorted(glob.glob(os.path.join(pairs_dir, "left_*.png")))

        if lefts_rect:
            file_pairs = [(l, l.replace("rectL_", "rectR_")) for l in lefts_rect
                          if os.path.exists(l.replace("rectL_", "rectR_"))]
            pre_rect   = True
            print(f"[i] Found {len(file_pairs)} pre-rectified pair(s) in '{pairs_dir}'")
        elif lefts_raw:
            file_pairs = [(l, l.replace("left_", "right_")) for l in lefts_raw
                          if os.path.exists(l.replace("left_", "right_"))]
            pre_rect   = False
            print(f"[i] Found {len(file_pairs)} raw pair(s) in '{pairs_dir}' — will remap")
        else:
            print(f"[!] No left_*.png or rectL_*.png found in '{pairs_dir}'")
            sys.exit(1)

        image_pairs = []
        for lp, rp in file_pairs:
            imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
            if imgL is None or imgR is None:
                continue
            if pre_rect:
                image_pairs.append((imgL, imgR))
            else:
                image_pairs.append((
                    cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR),
                    cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR),
                ))

    if not image_pairs:
        print("[!] No valid pairs — aborting.")
        sys.exit(1)

    # ── Analyse each pair ─────────────────────────────────────────────────────
    print(f"\n[i] Analysing {len(image_pairs)} pair(s) ...\n")
    all_grids   = []
    all_medians = []
    all_dy      = []

    for idx, (rectL, rectR) in enumerate(image_pairs):
        result = analyse_pair(rectL, rectR, idx, use_matplotlib)
        if result is not None:
            grid, med, dy_vals = result
            all_grids.append(grid)
            all_medians.append(med)
            all_dy.extend(dy_vals.tolist())

    if not all_grids:
        print("[!] No usable pairs.")
        sys.exit(1)

    # ── Combined summary ──────────────────────────────────────────────────────
    combined_grid = np.nanmedian(np.stack(all_grids), axis=0)

    if len(image_pairs) > 1:
        print(f"\n[i] Combined result across {len(image_pairs)} pairs:")
        plot_results(combined_grid, np.array(all_dy), "combined", use_matplotlib)

    diagnose(combined_grid, all_medians)


if __name__ == "__main__":
    main()
