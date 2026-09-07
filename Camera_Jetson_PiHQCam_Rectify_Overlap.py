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

DISP_W, DISP_H = 960, 540   # display size of the overlap view
FLICKER_FRAMES = 8          # frames per side in flicker mode
MEASURE_EVERY  = 30         # run the alignment measurement every N frames (~56 ms each)
ORB_FEATURES   = 800        # features per view for the measurement
MIN_MATCHES    = 20         # below this the measurement is reported as unavailable
DY_SPAN_PX     = 100        # display px; row errors are only searched within +-this
DY_BIN_PX      = 2          # vote bin width when looking for the dominant row error
DY_INLIER_PX   = 3          # display px around the winning bin that counts as an inlier

# Disparity matcher — same settings as Camera_Jetson_PiHQCam_Stereo.py
NUM_DISPARITIES  = 16 * 16  # must be multiple of 16; CUDA StereoBM max is 256
BLOCK_SIZE       = 5        # must be odd; 5-21 for StereoBM
TEXTURE_THRESH   = 20       # mask pixels with too little texture
UNIQUENESS_RATIO = 20       # reject match if 2nd-best is within this % of best (0=off)
PREFILTER_CAP    = 31       # clamp prefilter response; 1-63

# The calibration solves /dev/video0 as camera 1, but the recovered geometry puts
# camera 1 to the *right* of camera 2 (P2[0,3] > 0, T_x > 0, and rectified
# chessboard corners give xL - xR < 0) — the cameras are rolled 180 deg, same
# reason the preview needs ROTATE_180. Swap the eyes after rectification so the
# overlay is a true left/right pair. Set False if you recalibrate in device order.
SWAP_EYES = True
# ───────────────────────────────────────────────────────────────────────────────

MODES = {
    ord('1'): "anaglyph",     # left = red, right = cyan
    ord('2'): "blend",        # 50/50 average
    ord('3'): "difference",   # |L - R|, gain x2
    ord('4'): "flicker",      # alternate L/R in place
    ord('5'): "side-by-side", # the classic rectified pair
}


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
    P2    = fs.getNode("P2").mat()
    fs.release()
    focal    = Q[2, 3]             # focal length in pixels (from rectified projection)
    baseline = 1.0 / abs(Q[3, 2])  # baseline in metres
    return map1x, map1y, map2x, map2y, Q, P2, focal, baseline


def valid_mask(mapx, mapy, src_w, src_h):
    """True where a rectified pixel actually samples inside the source image."""
    return ((mapx >= 0) & (mapx <= src_w - 1) &
            (mapy >= 0) & (mapy <= src_h - 1))


def shift_x(img, dx):
    """Shift horizontally by dx px, filling the vacated edge with black."""
    if dx == 0:
        return img
    out = np.zeros_like(img)
    if dx > 0:
        out[:, dx:] = img[:, :-dx]
    else:
        out[:, :dx] = img[:, -dx:]
    return out


def disparity_warp(right, disp_disp_px, grid_x, grid_y):
    """Pull the right view onto the left one using a per-pixel x offset.

    StereoBM gives disparity in the *left* frame, so the pixel matching left
    (x, y) sits at (x - d, y) in the right view — sampling there lands every
    depth on top of its match at once, where the manual [/] shift can only ever
    align one plane. Pixels with no disparity keep d = 0 and stay misaligned;
    they are dimmed by the caller rather than hidden, so a bad match looks bad.
    """
    return cv2.remap(right, grid_x - disp_disp_px, grid_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compose(left, right, mode, frame_idx):
    """Combine the two rectified views into one BGR image."""
    if mode == "anaglyph":
        return np.dstack([right, right, left])          # L→red, R→cyan
    if mode == "blend":
        return cv2.cvtColor(cv2.addWeighted(left, 0.5, right, 0.5, 0), cv2.COLOR_GRAY2BGR)
    if mode == "difference":
        diff = cv2.convertScaleAbs(cv2.absdiff(left, right), alpha=2.0)
        return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    if mode == "flicker":
        side = left if (frame_idx // FLICKER_FRAMES) % 2 == 0 else right
        return cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    # side-by-side
    return cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_GRAY2BGR)


ORB = cv2.ORB_create(ORB_FEATURES)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def estimate_alignment(left, right):
    """Median row error and disparity from ORB matches, in *display* px.

    dy near 0 means the pair really is rectified; disparity must be positive —
    a negative value means the two eyes are the wrong way round. Phase
    correlation was tried here first and is not usable: on a scene with mixed
    depths it locks onto whichever plane dominates and reports nonsense dy at
    low confidence. Returns None when there are too few matches to trust.
    """
    kpL, desL = ORB.detectAndCompute(left, None)
    kpR, desR = ORB.detectAndCompute(right, None)
    if desL is None or desR is None:
        return None

    matches = BF.match(desL, desR)
    if len(matches) < MIN_MATCHES:
        return None

    ptsL = np.float32([kpL[m.queryIdx].pt for m in matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in matches])
    dy   = ptsR[:, 1] - ptsL[:, 1]
    dx   = ptsL[:, 0] - ptsR[:, 0]   # disparity: positive for a correctly ordered pair

    # Vote for the dominant row error rather than taking a median: on repetitive
    # texture most matches are wrong, and a median of mostly-wrong matches is a
    # confident wrong answer. A vote lets the correct matches win, or fail loudly.
    in_range = np.abs(dy) < DY_SPAN_PX
    if in_range.sum() < MIN_MATCHES:
        return None
    votes, edges = np.histogram(dy[in_range], bins=int(2 * DY_SPAN_PX / DY_BIN_PX),
                                range=(-DY_SPAN_PX, DY_SPAN_PX))
    peak = edges[votes.argmax()] + DY_BIN_PX / 2

    inliers = np.abs(dy - peak) < DY_INLIER_PX
    if inliers.sum() < MIN_MATCHES:
        return None
    return np.median(dy[inliers]), np.median(dx[inliers]), int(inliers.sum())


def alignment_text(alignment, scale):
    """One-line summary of an estimate_alignment() result, in rectified px."""
    if alignment is None:
        return "row error: too few matches"
    dy, disparity, n = alignment
    return (f"row error dy={dy * scale:+.2f} px   "
            f"disparity={disparity * scale:+.1f} px   n={n}")


def draw_hud(view, lines):
    for i, text in enumerate(lines):
        cv2.putText(view, text, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 255), 1, cv2.LINE_AA)
    return view


os.makedirs(CAPTURE_DIR, exist_ok=True)

print(f"[i] Loading calibration from '{CALIB_FILE}' ...")
map1x, map1y, map2x, map2y, Q, P2, FOCAL_PX, BASELINE_M = load_calibration(CALIB_FILE)
print(f"[i] focal={FOCAL_PX:.1f} px  baseline={BASELINE_M*100:.1f} cm")

# OpenCV writes P2[0,3] = -fx * baseline, so a positive value means camera 1
# (LEFT_DEVICE) is the right-eye view and the pair has to be swapped.
calib_swapped = P2[0, 3] > 0
if calib_swapped != SWAP_EYES:
    print(f"[!] SWAP_EYES={SWAP_EYES} but P2[0,3]={P2[0,3]:+.1f} says the eyes are "
          f"{'swapped' if calib_swapped else 'in device order'} — disparity will come out negative.")
print(f"[i] left eye = {RIGHT_DEVICE if SWAP_EYES else LEFT_DEVICE}, "
      f"right eye = {LEFT_DEVICE if SWAP_EYES else RIGHT_DEVICE}")

RECT_H, RECT_W = map1x.shape[:2]
SRC_W, SRC_H   = W // 2, H // 2   # extract_green halves both dimensions
SCALE          = RECT_W / DISP_W  # display px → rectified px

# Overlap = pixels that are valid in *both* rectified views
maskL   = valid_mask(map1x, map1y, SRC_W, SRC_H)
maskR   = valid_mask(map2x, map2y, SRC_W, SRC_H)
common  = (maskL & maskR)
ys, xs  = np.where(common)
if ys.size == 0:
    raise RuntimeError("Rectified views do not overlap — check the calibration file.")
OVERLAP_BOX = (xs.min(), ys.min(), xs.max(), ys.max())
overlap_pct = 100.0 * common.sum() / common.size
print(f"[i] rectified size {RECT_W}x{RECT_H} — common FOV {overlap_pct:.1f}% of frame, "
      f"bbox x[{OVERLAP_BOX[0]}..{OVERLAP_BOX[2]}] y[{OVERLAP_BOX[1]}..{OVERLAP_BOX[3]}]")

common_small = cv2.resize(common.astype(np.uint8) * 255, (DISP_W, DISP_H),
                          interpolation=cv2.INTER_NEAREST)
inside_small  = common_small > 0                              # unrotated, for valid-% stats
outside_small = cv2.rotate(common_small, cv2.ROTATE_180) == 0  # preview is rotated 180

# Sampling grid for the disparity warp, in display coordinates
GRID_X, GRID_Y = np.meshgrid(np.arange(DISP_W, dtype=np.float32),
                             np.arange(DISP_H, dtype=np.float32))

# Upload rectification maps to GPU once at startup
gpu_map1x = cv2.cuda_GpuMat(); gpu_map1x.upload(map1x)
gpu_map1y = cv2.cuda_GpuMat(); gpu_map1y.upload(map1y)
gpu_map2x = cv2.cuda_GpuMat(); gpu_map2x.upload(map2x)
gpu_map2y = cv2.cuda_GpuMat(); gpu_map2y.upload(map2y)

# CUDA StereoBM — same matcher as the depth script, on the rectified pair
try:
    matcher = cv2.cuda.createStereoBM(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
    matcher.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)
    matcher.setPreFilterCap(PREFILTER_CAP)
    matcher.setTextureThreshold(TEXTURE_THRESH)
    matcher.setUniquenessRatio(UNIQUENESS_RATIO)
    stream = cv2.cuda.Stream()
except (cv2.error, AttributeError) as exc:
    matcher = None
    print(f"[!] CUDA StereoBM unavailable ({exc}) — disparity warp disabled.")

print("[i] Initialising cameras ...")
p0, _ = initialize_camera_jetson(LEFT_DEVICE,  W, H, EXPOSURE, GAIN, FRAME_RATE)
p1, _ = initialize_camera_jetson(RIGHT_DEVICE, W, H, EXPOSURE, GAIN, FRAME_RATE)

WIN_BEFORE = "Before rectification"
WIN_AFTER  = "After rectification"
WIN_WARP   = "After disparity warp"
cv2.namedWindow(WIN_BEFORE); cv2.moveWindow(WIN_BEFORE, 0, 0)
cv2.namedWindow(WIN_AFTER);  cv2.moveWindow(WIN_AFTER, DISP_W + 10, 0)
cv2.namedWindow(WIN_WARP);   cv2.moveWindow(WIN_WARP, DISP_W + 10, DISP_H + 40)

print("[i] Running — 1-5 view mode, m=mask non-overlap, [/]=shift right view, "
      "0=reset shift, d=disparity warp, a=snap shift to median disparity, c=capture, q=quit.")

mode          = "anaglyph"
show_mask     = True
shift_disp    = 0      # horizontal shift of the right view, in display px
frame_idx     = 0
capture_count = 1
align_before  = None
align_after   = None
align_warp    = None
show_warp     = matcher is not None
warp          = np.zeros((DISP_H, DISP_W, 3), np.uint8)
med_disp_rect = None   # median valid disparity, in rectified px
valid_pct     = 0.0

while True:
    raw_0 = capture_raw_frame_jetson(p0, W, H)
    raw_1 = capture_raw_frame_jetson(p1, W, H)
    if raw_0 is None or raw_1 is None:
        print("[!] Frame grab failed.")
        break

    gray0 = raw_to_gray8(raw_0)
    gray1 = raw_to_gray8(raw_1)

    # Upload to GPU and remap — map1* belongs to video0, map2* to video1
    gpu_gray0 = cv2.cuda_GpuMat(); gpu_gray0.upload(gray0)
    gpu_gray1 = cv2.cuda_GpuMat(); gpu_gray1.upload(gray1)

    gpu_rect0 = cv2.cuda.remap(gpu_gray0, gpu_map1x, gpu_map1y, cv2.INTER_LINEAR)
    gpu_rect1 = cv2.cuda.remap(gpu_gray1, gpu_map2x, gpu_map2y, cv2.INTER_LINEAR)

    # Assign eyes only after rectification; the maps stay bound to their device
    gpu_rectL, gpu_rectR = (gpu_rect1, gpu_rect0) if SWAP_EYES else (gpu_rect0, gpu_rect1)

    # Downscale on GPU; measure and shift in rectified coordinates ...
    rawL_small,  rawR_small  = [cv2.cuda.resize(g, (DISP_W, DISP_H)).download()
                                for g in ((gpu_gray1, gpu_gray0) if SWAP_EYES else
                                          (gpu_gray0, gpu_gray1))]
    rectL_small, rectR_small = [cv2.cuda.resize(g, (DISP_W, DISP_H)).download()
                                for g in (gpu_rectL, gpu_rectR)]

    # ── Dense disparity → per-pixel x offset → right view pulled onto the left ──
    warpR_small = None
    if show_warp:
        # Matched in rectified orientation, where the epipolar lines are rows;
        # rotating first would flip the sign of every disparity.
        gpu_disp = matcher.compute(gpu_rectL, gpu_rectR, stream)
        stream.waitForCompletion()
        disp_small = cv2.cuda.resize(gpu_disp, (DISP_W, DISP_H),
                                     interpolation=cv2.INTER_NEAREST).download()

        # CUDA StereoBM returns whole-pixel CV_8U disparity (no 1/16 fixed point),
        # in rectified px — rescale to display px before warping the small view.
        disp_valid = disp_small > 0
        warpR_small = disparity_warp(rectR_small, disp_small.astype(np.float32) / SCALE,
                                     GRID_X, GRID_Y)

        matched   = disp_valid & inside_small
        valid_pct = 100.0 * matched.sum() / max(inside_small.sum(), 1)
        med_disp_rect = float(np.median(disp_small[disp_valid])) if disp_valid.any() else None
        no_disp_small = cv2.rotate((~disp_valid).astype(np.uint8), cv2.ROTATE_180) > 0

    # One measurement per frame, staggered — running both costs ~110 ms and stutters
    if frame_idx % MEASURE_EVERY == 0:
        align_after = estimate_alignment(rectL_small, rectR_small)
    elif frame_idx % MEASURE_EVERY == 1:
        align_before = estimate_alignment(rawL_small, rawR_small)
    elif frame_idx % MEASURE_EVERY == 2 and warpR_small is not None:
        # Residual after the warp: dx should collapse to ~0 where the match is good
        align_warp = estimate_alignment(rectL_small, warpR_small)

    # ... then rotate into viewing orientation (the cameras are mounted upside down)
    before = compose(cv2.rotate(rawL_small, cv2.ROTATE_180),
                     cv2.rotate(rawR_small, cv2.ROTATE_180), mode, frame_idx)
    after  = compose(cv2.rotate(rectL_small, cv2.ROTATE_180),
                     cv2.rotate(shift_x(rectR_small, shift_disp), cv2.ROTATE_180),
                     mode, frame_idx)

    if warpR_small is not None:
        warp = compose(cv2.rotate(rectL_small, cv2.ROTATE_180),
                       cv2.rotate(warpR_small, cv2.ROTATE_180), mode, frame_idx)

    # Dim whatever falls outside the common field of view (rectified view only)
    if show_mask and mode != "side-by-side":
        after[outside_small] = (after[outside_small] * 0.35).astype(np.uint8)
        after[outside_small, 2] = np.clip(after[outside_small, 2] + 40, 0, 255)
        if warpR_small is not None:
            # Dim both the non-overlap and the pixels the matcher gave up on, so
            # what stays bright is exactly what the disparity actually aligned
            dim = outside_small | no_disp_small
            warp[dim] = (warp[dim] * 0.35).astype(np.uint8)
            warp[dim, 2] = np.clip(warp[dim, 2] + 40, 0, 255)

    # ── Readout: shift in rectified px, and the depth that shift aligns ──────
    shift_rect = shift_disp * SCALE
    depth_txt  = "-"
    if abs(shift_rect) > 0.5:
        depth_txt = f"{(FOCAL_PX * BASELINE_M) / abs(shift_rect):.2f} m"

    draw_hud(before, [
        f"BEFORE rectification (raw green, eyes assigned)  |  {mode}",
        alignment_text(align_before, SCALE),
    ])
    draw_hud(after, [
        f"AFTER rectification  |  overlap {overlap_pct:.1f}%  |  1-5 modes  m=mask  q=quit",
        f"shift {shift_rect:+.0f} px (aligns ~{depth_txt})   [ ] to adjust, 0 to reset",
        alignment_text(align_after, SCALE),
    ])

    if show_warp:
        med_txt = "-"
        if med_disp_rect:
            med_txt = (f"{med_disp_rect:.1f} px (~{(FOCAL_PX * BASELINE_M) / med_disp_rect:.2f} m)")
        draw_hud(warp, [
            f"AFTER disparity warp  |  right eye pulled onto left, per pixel",
            f"matched {valid_pct:.0f}% of overlap   median disp {med_txt}   a=snap shift, d=off",
            alignment_text(align_warp, SCALE),
        ])
    else:
        warp = np.zeros((DISP_H, DISP_W, 3), np.uint8)
        reason = "press d to enable" if matcher is not None else "CUDA StereoBM unavailable"
        draw_hud(warp, ["AFTER disparity warp — off", reason])

    cv2.imshow(WIN_BEFORE, before)
    cv2.imshow(WIN_AFTER, after)
    cv2.imshow(WIN_WARP, warp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in MODES:
        mode = MODES[key]
    elif key == ord('m'):
        show_mask = not show_mask
    elif key == ord('['):
        shift_disp -= 1
    elif key == ord(']'):
        shift_disp += 1
    elif key == ord('0'):
        shift_disp = 0
    elif key == ord('d'):
        show_warp = (not show_warp) and matcher is not None
        if not show_warp:
            med_disp_rect, align_warp = None, None
    elif key == ord('a') and med_disp_rect:
        # The single shift that best matches the dominant depth in view — what
        # the manual [/] shift converges to by hand
        shift_disp = int(round(med_disp_rect / SCALE))
    elif key == ord('c'):
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"rectL_{ts}.png"), gpu_rectL.download())
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"rectR_{ts}.png"), gpu_rectR.download())
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"overlap_{mode}_before_{ts}.png"), before)
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"overlap_{mode}_after_{ts}.png"), after)
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"overlap_{mode}_warp_{ts}.png"), warp)
        print(f"[+] Captured overlap #{capture_count} → {CAPTURE_DIR}/")
        capture_count += 1

    frame_idx += 1

p0.terminate()
p1.terminate()
cv2.destroyAllWindows()
