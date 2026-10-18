#!/usr/bin/env python3
"""
Single source of truth for the ChArUco board, plus stereo-aware detection.

Both the PDF generator (make_charuco_board.py) and the calibrators import the
spec from here, so the printed board and the detector can never drift apart.

The one number you must edit after printing is MEASURED_SQUARE_M — printers
scale silently, and that value multiplies straight into the baseline and every
depth reading.
"""
import cv2
import numpy as np

# ─── Board geometry (must match the printed PDF) ───────────────────────────────
SQUARES_X    = 10          # squares across
SQUARES_Y    = 6           # squares down
SQUARE_MM    = 80.0        # nominal square size used to lay out the PDF
MARKER_RATIO = 0.75        # marker size as a fraction of the square
DICT_NAME    = "DICT_4X4_50"

# Measure five squares with calipers and divide by five. Do not leave this at
# the nominal value — a 2% print scale error is a silent 2% depth error.
MEASURED_SQUARE_M = 0.0795  # measured printed square: 79.5 mm

# ─── Detection quality gates ───────────────────────────────────────────────────
# Each view costs 6 pose parameters. A view with too few corners, or corners
# strung along a line, adds noise rather than information.
MIN_CORNERS_MONO   = 8     # per camera, for the per-camera calibration
MIN_CORNERS_STEREO = 12    # corners seen by BOTH cameras
MIN_SPREAD_RATIO   = 0.15  # 2nd/1st singular value of the corner cloud
# ───────────────────────────────────────────────────────────────────────────────

N_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)


def dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def make_board(square_len):
    """CharucoBoard at an arbitrary unit scale.

    Pass millimetres to lay out a print, metres to calibrate — object points
    come back in whatever unit is given here.
    """
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), square_len, square_len * MARKER_RATIO, dictionary())


def board_for_print():
    return make_board(SQUARE_MM)


def board_for_calib():
    return make_board(MEASURED_SQUARE_M)


def make_detector(board):
    return cv2.aruco.CharucoDetector(board)


def detect(gray, detector):
    """Detected ChArUco corners and their ids, or (None, None).

    Corner ids are what make partial views usable: a corner is identified by
    its neighbouring marker, not by counting in from a complete grid, so a
    board hanging off the frame edge still contributes everything visible.
    """
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS_MONO:
        return None, None
    return corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1)


def spread_ratio(pts):
    """Second/first singular value of the point cloud — 0 means collinear.

    Corners strung along a single row give a pose that is unconstrained about
    that axis, which poisons the fit even when there are plenty of them.
    """
    if len(pts) < 3:
        return 0.0
    p = pts[:, :2] - pts[:, :2].mean(axis=0)
    s = np.linalg.svd(p, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def match_pair(cL, iL, cR, iR, board):
    """Corners seen by both cameras, as (objp, ptsL, ptsR).

    stereoCalibrate needs object and image points to correspond 1:1, so only
    the intersection of the two id sets is usable — a corner one camera cannot
    see has nothing to match against. Returns None if the common set is too
    small or too collinear to give a stable pose.
    """
    common, ia, ib = np.intersect1d(iL, iR, return_indices=True)
    if len(common) < MIN_CORNERS_STEREO:
        return None
    objp = board.getChessboardCorners()[common].astype(np.float32)
    if spread_ratio(objp) < MIN_SPREAD_RATIO:
        return None
    return (objp.reshape(-1, 1, 3),
            cL[ia].reshape(-1, 1, 2),
            cR[ib].reshape(-1, 1, 2))


def object_points(ids, board):
    """Object points for one camera's own detections (mono calibration)."""
    return board.getChessboardCorners()[ids].astype(np.float32).reshape(-1, 1, 3)


def describe():
    d = dictionary()
    bits = d.markerSize + 2
    return (f"ChArUco {SQUARES_X}x{SQUARES_Y} squares, {N_CORNERS} corners, "
            f"square {MEASURED_SQUARE_M*1000:.2f} mm, "
            f"marker {MEASURED_SQUARE_M*MARKER_RATIO*1000:.2f} mm, "
            f"{DICT_NAME} ({bits} bits)")
