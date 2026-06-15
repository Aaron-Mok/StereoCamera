import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import cv2
from ISP.binning import extract_green, bin_4x4
from camera_utils import initialize_camera_jetson, capture_raw_frame_jetson
from conversion_utils import bit16_to_bit8

# ─── Settings ──────────────────────────────────────────────────────────────────
LEFT_DEVICE  = "/dev/video0"
RIGHT_DEVICE = "/dev/video1"

W, H       = 3840, 2160
EXPOSURE   = 33000
GAIN       = 200
FRAME_RATE = 30000000

CALIB_FILE  = "Calibration_output/Stereo_params_20260614_Jetson.yml"
CAPTURE_DIR = "captures"

NUM_DISPARITIES = 16 * 10   # must be multiple of 16
BLOCK_SIZE      = 9         # 5-11 recommended; smaller = more detail, noisier
WLS_LAMBDA      = 8000      # WLS filter smoothness (higher = smoother)
WLS_SIGMA       = 1.5       # WLS filter edge sensitivity
# ───────────────────────────────────────────────────────────────────────────────


def raw_to_gray8(raw_u16):
    """Raw 4K Bayer → 480×270 uint8 green channel (matches calibration resolution)."""
    return bit16_to_bit8(bin_4x4(extract_green(raw_u16)))


def load_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    map1x = fs.getNode("map1x").mat()
    map1y = fs.getNode("map1y").mat()
    map2x = fs.getNode("map2x").mat()
    map2y = fs.getNode("map2y").mat()
    Q     = fs.getNode("Q").mat()
    fs.release()
    return map1x, map1y, map2x, map2y, Q


os.makedirs(CAPTURE_DIR, exist_ok=True)

print(f"[i] Loading calibration from '{CALIB_FILE}' ...")
map1x, map1y, map2x, map2y, Q = load_calibration(CALIB_FILE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=8  * 3 * BLOCK_SIZE ** 2,
    P2=32 * 3 * BLOCK_SIZE ** 2,
    disp12MaxDiff=2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# WLS filter fills holes left by SGBM (requires opencv-contrib)
try:
    right_matcher = cv2.ximgproc.createRightMatcher(matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
    wls_filter.setLambda(WLS_LAMBDA)
    wls_filter.setSigmaColor(WLS_SIGMA)
    USE_WLS = True
    print("[i] WLS filter enabled.")
except AttributeError:
    USE_WLS = False
    print("[!] cv2.ximgproc not available — WLS filter disabled.")

print("[i] Initialising cameras ...")
pL, _ = initialize_camera_jetson(LEFT_DEVICE,  W, H, EXPOSURE, GAIN, FRAME_RATE)
pR, _ = initialize_camera_jetson(RIGHT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)

print("[i] Running — press 'q' to quit, 'c' to capture.")

capture_count = 1

while True:
    raw_L = capture_raw_frame_jetson(pL, W, H)
    raw_R = capture_raw_frame_jetson(pR, W, H)
    if raw_L is None or raw_R is None:
        print("[!] Frame grab failed.")
        break

    grayL = raw_to_gray8(raw_L)
    grayR = raw_to_gray8(raw_R)

    rectL = cv2.remap(grayL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, map2x, map2y, cv2.INTER_LINEAR)

    # ── Rectified view — horizontal lines verify alignment ──────────────────
    both_bgr = cv2.cvtColor(np.hstack([rectL, rectR]), cv2.COLOR_GRAY2BGR)
    for y in range(0, both_bgr.shape[0], 30):
        cv2.line(both_bgr, (0, y), (both_bgr.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.imshow("Rectified", both_bgr)

    # ── Disparity / depth view ──────────────────────────────────────────────
    dispL = matcher.compute(rectL, rectR)

    if USE_WLS:
        dispR = right_matcher.compute(rectR, rectL)
        disp = wls_filter.filter(dispL, rectL, disparity_map_right=dispR)
    else:
        disp = dispL

    # SGBM returns 16x fixed-point; divide to get real pixels, mask invalids
    disp_f = disp.astype(np.float32) / 16.0
    valid = disp_f >= 1.0
    disp_vis = np.zeros(disp_f.shape, dtype=np.uint8)
    if valid.any():
        mn, mx = disp_f[valid].min(), disp_f[valid].max()
        if mx > mn:
            disp_vis[valid] = ((disp_f[valid] - mn) / (mx - mn) * 255).astype(np.uint8)

    cv2.imshow("Depth", cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"rectL_{ts}.png"), rectL)
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"rectR_{ts}.png"), rectR)
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"depth_{ts}.png"), disp_vis)
        print(f"[+] Captured pair #{capture_count} → captures/")
        capture_count += 1

pL.terminate()
pR.terminate()
cv2.destroyAllWindows()
