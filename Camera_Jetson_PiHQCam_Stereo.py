import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import cv2
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

CALIB_FILE  = "Calibration_output/20260620_stereo_params.yml"
CAPTURE_DIR = "captures"

NUM_DISPARITIES  = 16 * 16  # must be multiple of 16; CUDA StereoBM max is 256; covers ~0.8 m to 5 m
BLOCK_SIZE       = 5       # must be odd; 5-21 for StereoBM
TEXTURE_THRESH   = 20       # mask pixels with too little texture (raise to kill more ghosts)
UNIQUENESS_RATIO = 20       # reject match if 2nd-best is within this % of best (0=off)
PREFILTER_CAP    = 31       # clamp prefilter response; 1-63
SPECKLE_WINDOW   = 0      # min area (px) to keep a disparity region; 0=off
SPECKLE_RANGE    = 32       # max disparity variation within a speckle region
# ───────────────────────────────────────────────────────────────────────────────


def raw_to_gray8(raw_u16):
    """Raw 4K Bayer → 1920×1080 uint8 green channel (matches calibration resolution)."""
    return bit16_to_bit8(extract_green(raw_u16))


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

# Upload rectification maps to GPU once at startup
gpu_map1x = cv2.cuda_GpuMat(); gpu_map1x.upload(map1x)
gpu_map1y = cv2.cuda_GpuMat(); gpu_map1y.upload(map1y)
gpu_map2x = cv2.cuda_GpuMat(); gpu_map2x.upload(map2x)
gpu_map2y = cv2.cuda_GpuMat(); gpu_map2y.upload(map2y)

# CUDA StereoBM — faster than SGBM on GPU; WLS not compatible with CUDA matcher
matcher = cv2.cuda.createStereoBM(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
matcher.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)  # edge filter reduces horizontal ghosting
matcher.setPreFilterCap(PREFILTER_CAP)
matcher.setTextureThreshold(TEXTURE_THRESH)
matcher.setUniquenessRatio(UNIQUENESS_RATIO)
# matcher.setSpeckleWindowSize(SPECKLE_WINDOW)
# matcher.setSpeckleRange(SPECKLE_RANGE)
stream = cv2.cuda.Stream()
print("[i] Using CUDA StereoBM (WLS disabled).")

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

    # Upload to GPU and remap
    gpu_grayL = cv2.cuda_GpuMat(); gpu_grayL.upload(grayL)
    gpu_grayR = cv2.cuda_GpuMat(); gpu_grayR.upload(grayR)

    gpu_rectL = cv2.cuda.remap(gpu_grayL, gpu_map1x, gpu_map1y, cv2.INTER_LINEAR)
    gpu_rectR = cv2.cuda.remap(gpu_grayR, gpu_map2x, gpu_map2y, cv2.INTER_LINEAR)

    # Download rectified frames for capture
    rectL = gpu_rectL.download()
    rectR = gpu_rectR.download()

    # ── Rectified view — resize to 960×540 each on GPU → 1920×540 side-by-side
    bgrL = cv2.cuda.cvtColor(cv2.cuda.resize(gpu_rectL, (960, 540)), cv2.COLOR_GRAY2BGR).download()
    bgrR = cv2.cuda.cvtColor(cv2.cuda.resize(gpu_rectR, (960, 540)), cv2.COLOR_GRAY2BGR).download()
    both_bgr = cv2.rotate(np.hstack([bgrL, bgrR]), cv2.ROTATE_180)
    for y in range(0, both_bgr.shape[0], 30):
        cv2.line(both_bgr, (0, y), (both_bgr.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.imshow("Rectified", both_bgr)

    # ── Disparity on GPU, download for post-processing ──────────────────────
    gpu_disp = matcher.compute(gpu_rectL, gpu_rectR, stream)
    disp = gpu_disp.download()

    # StereoBM returns 16x fixed-point; divide to get real pixels, mask invalids
    # Normalize against fixed range so colors are physically consistent across frames:
    #   disp=1 (far, ~5 m) → 0 (blue in TURBO)
    #   disp=NUM_DISPARITIES (near, ~0.8 m) → 255 (red in TURBO)
    # disp_f = disp.astype(np.float32) / 16.0
    disp_f = disp
    valid = disp_f >= 0.0
    disp_vis = np.zeros(disp_f.shape, dtype=np.uint8)
    disp_vis[valid] = disp_f[valid]

    depth_display = cv2.rotate(cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO), (960, 540)), cv2.ROTATE_180)
    cv2.imshow("Depth", depth_display)

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
