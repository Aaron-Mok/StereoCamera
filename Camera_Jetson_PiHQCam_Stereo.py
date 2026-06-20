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

NUM_DISPARITIES = 16 * 5   # must be multiple of 16
BLOCK_SIZE      = 5         # must be odd; 5-21 for StereoBM
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

# Upload rectification maps to GPU once at startup
gpu_map1x = cv2.cuda_GpuMat(); gpu_map1x.upload(map1x)
gpu_map1y = cv2.cuda_GpuMat(); gpu_map1y.upload(map1y)
gpu_map2x = cv2.cuda_GpuMat(); gpu_map2x.upload(map2x)
gpu_map2y = cv2.cuda_GpuMat(); gpu_map2y.upload(map2y)

# CUDA StereoBM — faster than SGBM on GPU; WLS not compatible with CUDA matcher
matcher = cv2.cuda.createStereoBM(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
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

    # Download rectified frames for display and capture
    rectL = gpu_rectL.download()
    rectR = gpu_rectR.download()

    # ── Rectified view — horizontal lines verify alignment ──────────────────
    both_bgr = cv2.cvtColor(np.hstack([rectL, rectR]), cv2.COLOR_GRAY2BGR)
    for y in range(0, both_bgr.shape[0], 30):
        cv2.line(both_bgr, (0, y), (both_bgr.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.imshow("Rectified", both_bgr)

    # ── Disparity on GPU, download for post-processing ──────────────────────
    gpu_disp = matcher.compute(gpu_rectL, gpu_rectR, stream)
    disp = gpu_disp.download()

    # StereoBM returns 16x fixed-point; divide to get real pixels, mask invalids
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
