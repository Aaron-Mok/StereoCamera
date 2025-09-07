#!/usr/bin/env python3
import cv2, numpy as np, os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from StereoCamera import camera_utils  # your helper

# --- Settings ---
LEFT_CAM_INDEX  = 1
RIGHT_CAM_INDEX = 0
RESOLUTION      = (800, 600)  # must match calibration size
FLIP_FRAMES     = False
YAML_IN         = "stereo_params.yml"

# --- Load calibration/rectification from YAML ---
fs = cv2.FileStorage(YAML_IN, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise FileNotFoundError(f"Cannot open {YAML_IN}")
Q      = fs.getNode("Q").mat()
map1x  = fs.getNode("map1x").mat(); map1y = fs.getNode("map1y").mat()
map2x  = fs.getNode("map2x").mat(); map2y = fs.getNode("map2y").mat()
img_w  = int(fs.getNode("image_width").real())
img_h  = int(fs.getNode("image_height").real())
fs.release()

if (img_w, img_h) != RESOLUTION:
    print(f"[!] Warning: camera RESOLUTION {RESOLUTION} != YAML size {(img_w, img_h)}.")
    print("    For best results, set RESOLUTION to the YAML size or regenerate maps.")

def grab_frame(picam2):
    frame = picam2.capture_array()
    if frame is None: return None
    return cv2.flip(frame, -1) if FLIP_FRAMES else frame

# --- Init cameras (Picamera2 via your utils) ---
left_picam2,  _ = camera_utils.initialize_camera_with_ISP(LEFT_CAM_INDEX,  RESOLUTION)
right_picam2, _ = camera_utils.initialize_camera_with_ISP(RIGHT_CAM_INDEX, RESOLUTION)

# Optional: lock exposure/white balance (copy left -> right)
time.sleep(1.0)
meta = left_picam2.capture_metadata()
exp_time = meta.get("ExposureTime", 10000)
gain     = meta.get("AnalogueGain", 1.0)
col_g    = meta.get("ColourGains", (1.0, 1.0))
for cam in (left_picam2, right_picam2):
    cam.set_controls({"AeEnable": False, "AwbEnable": False,
                      "ExposureTime": exp_time, "AnalogueGain": gain, "ColourGains": col_g})

# --- Create SGBM + optional WLS filter ---
win = 5                       # blockSize
num_disp = 160                # must be multiple of 16; increase if objects are very close
min_disp = 0
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=win,
    P1=8 * 1 * win * win,
    P2=32 * 1 * win * win,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
)

use_wls = False
try:
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls.setLambda(8000.0); wls.setSigmaColor(1.2)
    use_wls = True
except Exception:
    pass

print("[i] q = quit,  s = save PLY point cloud (depth)")

while True:
    L = grab_frame(left_picam2)
    R = grab_frame(right_picam2)
    if L is None or R is None:
        print("[!] Frame grab failed"); break

    # Rectify
    rectL = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)

    # Grayscale
    gL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Disparity
    dispL = stereo.compute(gL, gR).astype(np.int16)
    if use_wls:
        dispR = right_matcher.compute(gR, gL).astype(np.int16)
        disp = wls.filter(dispL, gL, disparity_map_right=dispR)
    else:
        disp = dispL

    # SGBM returns disparity*16; convert to float px
    disp_f = disp.astype(np.float32) / 16.0  # disparity in pixels
    valid = disp_f > 0.5

    # scale disparity into 0..255 based on known range
    disp_vis = np.zeros_like(disp_f, dtype=np.uint8)
    disp_vis[valid] = np.clip(
        (disp_f[valid] - min_disp) / num_disp * 255, 0, 255
    ).astype(np.uint8)

    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)
    cv2.imshow("Disparity", disp_color)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        # Depth via Q
        pts3d = cv2.reprojectImageTo3D(disp_f, Q)  # meters if baseline in meters in your YAML
        mask = valid & np.isfinite(pts3d[...,2])
        # Save simple PLY
        pts = pts3d[mask].reshape(-1,3)
        cols = rectL[mask].reshape(-1,3)[:, ::-1]  # BGR->RGB
        ply_path = f"pointcloud_{int(time.time())}.ply"
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (x,y,z),(r,g,b) in zip(pts, cols):
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        print(f"[✓] Saved {ply_path}")

# Cleanup
cv2.destroyAllWindows()
try: left_picam2.stop()
except: pass
try: right_picam2.stop()
except: pass
