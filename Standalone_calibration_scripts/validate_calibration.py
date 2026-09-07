#!/usr/bin/env python3
"""
Validate a stereo calibration before trusting it for depth.

Reports the numbers that actually predict whether block matching will work —
rectified epipolar error above all — plus the input-coverage stats that explain
a bad result. Run this on the pairs used for calibration, or (better) on a
held-out set captured separately.

Usage:
  python3 validate_calibration.py [calib_yaml] [pairs_dir]
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from stereo_calibrate_offline import load_pairs, make_object_points, find_corners

# ─── Settings ──────────────────────────────────────────────────────────────────
CB_COLS       = 8
CB_ROWS       = 5
SQUARE_SIZE_M = 0.025

CALIB_YAML = "Calibration_output/20260620_stereo_params.yml"
PAIRS_DIR  = "stereo_calib_pairs"

# Pass/warn thresholds
EPI_RMS_PASS    = 0.30   # px — rectified epipolar error; StereoBM needs this
EPI_RMS_WARN    = 0.50
EPI_MAX_PASS    = 1.00   # px — worst single corner
MONO_RMS_PASS   = 0.30   # px — per-camera reprojection
STEREO_RATIO_PASS = 1.5  # stereo RMS / mono RMS; >>1 means poses can't constrain extrinsics
HULL_PASS       = 60.0   # % of frame area spanned by board corners
GRID_PASS       = 70.0   # % of grid cells visited
DIST_RATIO_PASS = 2.5    # max/min board distance

GRID_COLS, GRID_ROWS = 12, 8   # coverage map resolution
# ───────────────────────────────────────────────────────────────────────────────


def read_calib(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open '{path}'")
    get = lambda k: fs.getNode(k).mat()
    c = {k: get(k) for k in ("K1", "D1", "K2", "D2", "R", "T",
                             "R1", "R2", "P1", "P2", "Q")}
    # Image size comes from the remap tables — always present, never ambiguous.
    c["size"] = (get("map1x").shape[1], get("map1x").shape[0])
    fs.release()
    missing = [k for k, v in c.items() if v is None]
    if missing:
        raise ValueError(f"'{path}' is missing: {', '.join(missing)}")
    return c


def collect_corners(pairs, pattern_size):
    """Detect the board in both views of every pair. Returns parallel lists."""
    objp = make_object_points(pattern_size, SQUARE_SIZE_M)
    obj, ptsL, ptsR, names = [], [], [], []
    rejected = []
    for lp, rp in pairs:
        gl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if gl is None or gr is None:
            rejected.append((os.path.basename(lp), "unreadable"))
            continue
        okL, cL = find_corners(gl, pattern_size)
        okR, cR = find_corners(gr, pattern_size)
        if not (okL and okR):
            side = "left" if not okL else "right"
            rejected.append((os.path.basename(lp), f"no corners ({side})"))
            continue
        obj.append(objp)
        ptsL.append(cL)
        ptsR.append(cR)
        names.append(os.path.basename(lp))
    return obj, ptsL, ptsR, names, rejected


def reprojection_errors(obj, ptsL, ptsR, c):
    """Per-camera reprojection RMS, and stereo RMS using the stored R,T.

    Pose is re-fit per view with solvePnP, so mono numbers are mildly
    optimistic. The stereo number reuses the left pose pushed through R,T,
    which is what makes it a direct test of the extrinsics.
    """
    K1, D1, K2, D2, R, T = c["K1"], c["D1"], c["K2"], c["D2"], c["R"], c["T"]
    eL, eR, eS, dists = [], [], [], []
    for o, pL, pR in zip(obj, ptsL, ptsR):
        okL, rvec1, tvec1 = cv2.solvePnP(o, pL, K1, D1)
        okR, rvec2, tvec2 = cv2.solvePnP(o, pR, K2, D2)
        if not (okL and okR):
            continue
        dists.append(float(tvec1[2]))

        proj, _ = cv2.projectPoints(o, rvec1, tvec1, K1, D1)
        eL.append((proj - pL).reshape(-1, 2))
        proj, _ = cv2.projectPoints(o, rvec2, tvec2, K2, D2)
        eR.append((proj - pR).reshape(-1, 2))

        # Left pose carried into the right camera by the stored extrinsics.
        rvec21, tvec21, *_ = cv2.composeRT(rvec1, tvec1,
                                           cv2.Rodrigues(R)[0].astype(np.float64), T)
        proj, _ = cv2.projectPoints(o, rvec21, tvec21, K2, D2)
        eS.append((proj - pR).reshape(-1, 2))

    rms = lambda e: float(np.sqrt((np.vstack(e) ** 2).sum(axis=1).mean()))
    return rms(eL), rms(eR), rms(eS), np.array(dists)


def epipolar_errors(ptsL, ptsR, c):
    """Rectified vertical disagreement, per corner.

    Corners are pushed through the rectifying transform analytically rather
    than re-detected on remapped images — no detector failures, no resampling
    blur, and every pair contributes.
    """
    K1, D1, K2, D2 = c["K1"], c["D1"], c["K2"], c["D2"]
    dy, dx = [], []
    for pL, pR in zip(ptsL, ptsR):
        rL = cv2.undistortPoints(pL, K1, D1, R=c["R1"], P=c["P1"]).reshape(-1, 2)
        rR = cv2.undistortPoints(pR, K2, D2, R=c["R2"], P=c["P2"]).reshape(-1, 2)
        dy.append(rL[:, 1] - rR[:, 1])
        dx.append(rL[:, 0] - rR[:, 0])
    return np.concatenate(dy), np.concatenate(dx)


def coverage(ptsL, size):
    """How much of the frame the board actually visited."""
    P = np.vstack([p.reshape(-1, 2) for p in ptsL]).astype(np.float32)
    W, H = size
    hull_pct = 100.0 * cv2.contourArea(cv2.convexHull(P)) / (W * H)

    grid = np.zeros((GRID_ROWS, GRID_COLS), bool)
    gx = np.clip((P[:, 0] / W * GRID_COLS).astype(int), 0, GRID_COLS - 1)
    gy = np.clip((P[:, 1] / H * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
    grid[gy, gx] = True
    return hull_pct, 100.0 * grid.mean(), grid, P


def verdict(ok, warn=False):
    return "\033[33mWARN\033[0m" if warn else ("\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m")


def main():
    calib_yaml = sys.argv[1] if len(sys.argv) > 1 else CALIB_YAML
    pairs_dir  = sys.argv[2] if len(sys.argv) > 2 else PAIRS_DIR
    pattern_size = (CB_COLS, CB_ROWS)

    print(f"[i] Calibration : {calib_yaml}")
    print(f"[i] Pairs       : {pairs_dir}")
    c = read_calib(calib_yaml)
    W, H = c["size"]

    pairs = load_pairs(pairs_dir)
    obj, ptsL, ptsR, names, rejected = collect_corners(pairs, pattern_size)
    print(f"[i] {len(obj)}/{len(pairs)} pairs detected in both views "
          f"({len(obj) * CB_COLS * CB_ROWS} corners)")
    for n, why in rejected:
        print(f"    rejected: {n} — {why}")
    if len(obj) < 3:
        print("[!] Too few usable pairs to validate.")
        return 1

    fails, warns = [], []

    # ── Geometry sanity ────────────────────────────────────────────────────
    Q = c["Q"]
    focal = float(Q[2, 3])
    baseline = 1.0 / abs(float(Q[3, 2]))
    Tx = float(c["T"][0])
    print(f"\n─── Geometry " + "─" * 54)
    print(f"  image size     : {W}×{H}")
    print(f"  focal          : {focal:.1f} px")
    print(f"  baseline       : {baseline * 100:.2f} cm")
    print(f"  T[0]           : {Tx:+.4f} m   {verdict(Tx < 0)}"
          f"   {'' if Tx < 0 else '<- left/right inputs are SWAPPED'}")
    if Tx >= 0:
        fails.append("T[0] positive — the 'left' camera is physically on the right; "
                     "StereoBM searches the wrong direction and can never match")

    # ── Reprojection ───────────────────────────────────────────────────────
    rmsL, rmsR, rmsS, dists = reprojection_errors(obj, ptsL, ptsR, c)
    mono = max(rmsL, rmsR)
    ratio = rmsS / mono if mono > 0 else float("inf")
    print(f"\n─── Reprojection " + "─" * 50)
    print(f"  mono RMS left  : {rmsL:.3f} px   {verdict(rmsL < MONO_RMS_PASS)}")
    print(f"  mono RMS right : {rmsR:.3f} px   {verdict(rmsR < MONO_RMS_PASS)}")
    print(f"  stereo RMS     : {rmsS:.3f} px   (stricter than the number "
          f"stereoCalibrate prints —\n"
          f"                                    poses are not re-optimised to hide "
          f"extrinsic error)")
    print(f"  stereo / mono  : {ratio:.2f}×      {verdict(ratio < STEREO_RATIO_PASS)}"
          f"   (>{STEREO_RATIO_PASS:.1f} means poses don't constrain the extrinsics)")
    if rmsL >= MONO_RMS_PASS or rmsR >= MONO_RMS_PASS:
        warns.append(f"per-camera reprojection above {MONO_RMS_PASS} px")
    if ratio >= STEREO_RATIO_PASS:
        fails.append(f"stereo RMS is {ratio:.1f}× the per-camera RMS — each camera fits "
                     "its own views but the pair cannot be reconciled; add pose diversity")

    # ── Epipolar error (the one that matters) ──────────────────────────────
    dy, dx = epipolar_errors(ptsL, ptsR, c)
    epi_rms = float(np.sqrt((dy ** 2).mean()))
    epi_max = float(np.abs(dy).max())
    print(f"\n─── Rectified epipolar error " + "─" * 38)
    print(f"  dy RMS         : {epi_rms:.3f} px   "
          f"{verdict(epi_rms < EPI_RMS_PASS, EPI_RMS_PASS <= epi_rms < EPI_RMS_WARN)}"
          f"   (target < {EPI_RMS_PASS})")
    print(f"  dy |max|       : {epi_max:.3f} px   {verdict(epi_max < EPI_MAX_PASS)}")
    print(f"  dy mean/bias   : {dy.mean():+.3f} px")
    if epi_rms >= EPI_RMS_WARN:
        fails.append(f"epipolar RMS {epi_rms:.2f} px — block matching needs "
                     f"< {EPI_RMS_PASS} px; correct matches fall outside the block")
    elif epi_rms >= EPI_RMS_PASS:
        warns.append(f"epipolar RMS {epi_rms:.2f} px is marginal")

    # Disparity sign and implied working range
    print(f"  disparity      : {dx.min():+.0f} … {dx.max():+.0f} px "
          f"({'correct sign' if dx.mean() > 0 else 'NEGATIVE — inputs swapped'})")
    if abs(dx).max() > 0:
        near = focal * baseline / abs(dx).max()
        far  = focal * baseline / max(abs(dx).min(), 1e-6)
        print(f"  board spanned  : {near:.2f} – {min(far, 99):.2f} m")

    # ── Input coverage ─────────────────────────────────────────────────────
    hull_pct, grid_pct, grid, P = coverage(ptsL, (W, H))
    dratio = dists.max() / dists.min() if len(dists) and dists.min() > 0 else 0.0
    print(f"\n─── Input coverage " + "─" * 48)
    print(f"  hull area      : {hull_pct:.1f} %     {verdict(hull_pct > HULL_PASS)}"
          f"   (target > {HULL_PASS:.0f})")
    print(f"  grid cells hit : {grid_pct:.1f} %     {verdict(grid_pct > GRID_PASS)}"
          f"   (target > {GRID_PASS:.0f})")
    print(f"  board distance : {dists.min():.2f} – {dists.max():.2f} m  "
          f"({dratio:.1f}× spread)   {verdict(dratio > DIST_RATIO_PASS)}")
    if hull_pct <= HULL_PASS or grid_pct <= GRID_PASS:
        fails.append("board never visited large parts of the frame — distortion there "
                     "is extrapolated, so rectification drifts off-centre")
    if dratio <= DIST_RATIO_PASS:
        fails.append(f"board distance spans only {dratio:.1f}× — too shallow to "
                     "separate focal length from board depth")

    print("\n  where the board was seen (# = sampled, . = never):")
    for row in grid:
        print("    " + " ".join("#" if v else "." for v in row))

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n─── Verdict " + "─" * 55)
    if not fails and not warns:
        print("  \033[32mPASS\033[0m — calibration is good enough for block matching.")
        return 0
    for f in fails:
        print(f"  \033[31m✗\033[0m {f}")
    for w in warns:
        print(f"  \033[33m!\033[0m {w}")
    if fails:
        print("\n  \033[31mDo not use this calibration for depth.\033[0m Recapture with the "
              "board\n  at varied distances and pushed into every corner of the frame.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
