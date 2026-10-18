import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import argparse
import json
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

NUM_DISPARITIES  = 16 * 16  # full-resolution search range; nearest depth is about focal*baseline/255
BLOCK_SIZE       = 5       # must be odd; 5-21 for StereoBM
TEXTURE_THRESH   = 20       # mask pixels with too little texture (raise to kill more ghosts)
UNIQUENESS_RATIO = 20       # reject match if 2nd-best is within this % of best (0=off)
PREFILTER_CAP    = 31       # clamp prefilter response; 1-63
SPECKLE_WINDOW   = 0      # min area (px) to keep a disparity region; 0=off
SPECKLE_RANGE    = 32       # max disparity variation within a speckle region
DEPTH_MIN_M      = 0.3      # clip range for depth colormap (metres)
DEPTH_MAX_M      = 5.0
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
    focal    = Q[2, 3]           # focal length in pixels (from rectified projection)
    baseline = 1.0 / abs(Q[3, 2])  # baseline in metres
    return map1x, map1y, map2x, map2y, Q, focal, baseline


def make_sgbm(scale, num_disparities=NUM_DISPARITIES):
    """CPU matcher with smoothness and speckle rejection for a denser preview."""
    if num_disparities <= 0 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    num_disparities = int(np.ceil(num_disparities * scale / 16)) * 16
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=5,
        P1=8 * 25, P2=32 * 25, preFilterCap=31,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def make_wls(matcher):
    """Create a reverse matcher and edge-aware filter, including consistency confidence."""
    if not (hasattr(cv2, "ximgproc") and
            hasattr(cv2.ximgproc, "createDisparityWLSFilter") and
            hasattr(cv2.ximgproc, "createRightMatcher")):
        raise RuntimeError("--wls requires an OpenCV build with ximgproc. Omit --wls to run without it.")
    # OpenCV adjusts the left matcher for WLS; make the reverse matcher afterwards.
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
    wls.setLambda(8000)
    wls.setSigmaColor(1.5)
    return cv2.ximgproc.createRightMatcher(matcher), wls


def compute_sgbm(matcher, first, second, scale, postfilter=None):
    """Return disparity in ORIGINAL image pixels, even when matching at half size."""
    height, width = first.shape
    if scale != 1.0:
        first = cv2.resize(first, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        second = cv2.resize(second, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    actual_scale = first.shape[1] / width
    if first.shape[1] <= matcher.getNumDisparities() + matcher.getBlockSize():
        raise ValueError("Disparity search is too wide for the matching image; reduce --num-disparities")
    raw = matcher.compute(first, second)
    if postfilter is not None:
        right_matcher, wls = postfilter
        reverse_disparity = right_matcher.compute(second, first)
        raw = wls.filter(raw, first, disparity_map_right=reverse_disparity)
        # Confidence is 0..255. Suppress weak estimates even if the filter fills them.
        raw[wls.getConfidenceMap() < 128] = -16
    # CPU SGBM uses 16x fixed point. Also undo image scaling before using Q.
    disparity = raw.astype(np.float32) / (16 * actual_scale)
    if scale != 1.0:
        # Nearest-neighbour preserves invalid pixels and sharp depth boundaries.
        disparity = cv2.resize(disparity, (width, height), interpolation=cv2.INTER_NEAREST)
    return disparity


def cuda_sgm_config(image_size, num_disparities):
    """Fit a full-resolution search range into CUDA SGM's 64/128/256 choices."""
    if num_disparities <= 0 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Calibration image dimensions must be positive")
    # Prefer half resolution; reduce further when necessary to retain near coverage.
    scale = min(0.5, 256.0 / num_disparities)
    match_width = max(1, int(np.floor(width * scale)))
    scale = match_width / width
    match_height = max(1, round(height * scale))
    search = next(n for n in (64, 128, 256) if n >= num_disparities * scale - 1e-6)
    # The CUDA census transform uses a 9x7 neighbourhood.
    if match_width <= search + 9 or match_height < 9:
        raise ValueError("Disparity search is too wide for CUDA SGM at this image size; reduce --num-disparities")
    return {"image_size": (width, height), "size": (match_width, match_height),
            "scale": scale, "num_disparities": search}


def make_cuda_sgm(config):
    if not hasattr(cv2.cuda, "createStereoSGM"):
        raise RuntimeError("This OpenCV build does not expose CUDA StereoSGM; use --matcher sgbm")
    # Census matching has a different cost scale from CPU SGBM. Use CUDA's
    # starting penalties, rather than the CPU block-cost penalties of 200/800.
    return cv2.cuda.createStereoSGM(
        minDisparity=0, numDisparities=config["num_disparities"],
        P1=10, P2=120, uniquenessRatio=10, mode=cv2.STEREO_SGBM_MODE_HH4)


def compute_cuda_sgm(matcher, first, second, config):
    """Match rectified GPU images and return full-resolution pixel disparities."""
    if first.size() != config["image_size"] or second.size() != config["image_size"]:
        raise ValueError("Rectified image dimensions do not match the CUDA SGM configuration")
    small_first = cv2.cuda.resize(first, config["size"], interpolation=cv2.INTER_LINEAR)
    small_second = cv2.cuda.resize(second, config["size"], interpolation=cv2.INTER_LINEAR)
    raw = matcher.compute(small_first, small_second).download()
    # Unlike CUDA BM's uint8 output, CUDA SGM returns signed 16x fixed point.
    disparity = raw.astype(np.float32) / (16 * config["scale"])
    return cv2.resize(disparity, config["image_size"], interpolation=cv2.INTER_NEAREST)


def depth_from_disparity(disparity, Q, reverse_match):
    """Positive matcher disparity -> metres using the calibration's signed geometry."""
    signed_disp = -disparity if reverse_match else disparity
    denominator = Q[3, 2] * signed_disp + Q[3, 3]
    valid = (disparity > 0) & np.isfinite(disparity) & (denominator > 0)
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[valid] = Q[2, 3] / denominator[valid]
    return depth, valid


def colorize_depth(depth, valid):
    """Near is red, far is blue; missing depth is black."""
    clipped = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)
    norm = ((DEPTH_MAX_M - clipped) / (DEPTH_MAX_M - DEPTH_MIN_M) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def disparity_preview(disparity, valid, max_disparity):
    """Map the configured pixel range to uint8 without saturating close objects at 255 px."""
    vis = np.zeros(disparity.shape, dtype=np.uint8)
    vis[valid] = np.clip(disparity[valid] / max_disparity * 255, 0, 255).astype(np.uint8)
    return vis


def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Live rectified stereo and depth preview")
    parser.add_argument("--calibration", default=CALIB_FILE, help="Calibration YAML file")
    parser.add_argument("--matcher", choices=("bm", "sgbm", "sgm"), default="bm",
                        help="bm: CUDA block matching; sgbm: CPU SGBM; sgm: CUDA semi-global matching")
    parser.add_argument("--sgbm-scale", type=float, choices=(0.5, 1.0), default=0.5,
                        help="SGBM image scale (default: 0.5 for speed and less noise)")
    parser.add_argument("--num-disparities", type=int, default=NUM_DISPARITIES,
                        help="Full-resolution search range, a multiple of 16 (default: 256; try 640 for close objects)")
    parser.add_argument("--wls", action="store_true",
                        help="Filter SGBM streaks using image edges and reverse-match confidence (slower)")
    args = parser.parse_args()
    if args.num_disparities <= 0 or args.num_disparities % 16:
        parser.error("--num-disparities must be a positive multiple of 16")
    if args.matcher == "bm" and args.num_disparities > 256:
        parser.error("CUDA BM supports at most 256 disparities; use --matcher sgbm or sgm for a wider range")
    if args.wls and args.matcher != "sgbm":
        parser.error("--wls currently requires CPU --matcher sgbm; omit --wls for CUDA --matcher sgm")
    calib_file = args.calibration

    print(f"[i] Loading calibration from '{calib_file}' ...")
    map1x, map1y, map2x, map2y, Q, FOCAL_PX, BASELINE_M = load_calibration(calib_file)
    print(f"[i] focal={FOCAL_PX:.1f} px  baseline={BASELINE_M*100:.1f} cm")
    reverse_match = Q[3, 2] < 0
    if reverse_match:
        print("[i] Negative disparity geometry: matching right-to-left; depth is shown in the right view.")

    postfilter = None
    if args.matcher == "sgbm":
        matcher = make_sgbm(args.sgbm_scale, args.num_disparities)
        if args.wls:
            try:
                postfilter = make_wls(matcher)
            except RuntimeError as exc:
                parser.error(str(exc))
        matching_scale = args.sgbm_scale
        print(f"[i] Using CPU SGBM at {args.sgbm_scale:g}x resolution; missing depth is black.")
        if args.wls:
            print("[i] WLS filtering enabled; low-confidence estimates remain black.")
    elif args.matcher == "sgm":
        try:
            sgm_config = cuda_sgm_config((map1x.shape[1], map1x.shape[0]), args.num_disparities)
            matcher = make_cuda_sgm(sgm_config)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        matching_scale = sgm_config["scale"]
        mw, mh = sgm_config["size"]
        print(f"[i] Using CUDA StereoSGM at {mw}x{mh} ({matching_scale:g}x), "
              f"{sgm_config['num_disparities']} matching disparities; WLS off.")
    else:
        matcher = cv2.cuda.createStereoBM(numDisparities=args.num_disparities, blockSize=BLOCK_SIZE)
        matcher.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)
        matcher.setPreFilterCap(PREFILTER_CAP)
        matcher.setTextureThreshold(TEXTURE_THRESH)
        matcher.setUniquenessRatio(UNIQUENESS_RATIO)
        stream = cv2.cuda.Stream()
        matching_scale = 1.0
        print("[i] Using CUDA StereoBM; missing depth is black.")

    if round(map1x.shape[1] * matching_scale) <= matcher.getNumDisparities() + matcher.getBlockSize():
        parser.error("Disparity search is too wide for this calibration's image width")
    max_disparity = (matcher.getNumDisparities() - 1) / matching_scale
    denominator = abs(Q[3, 2]) * max_disparity + Q[3, 3]
    if denominator > 0:
        print(f"[i] Search ceiling ~{max_disparity:.0f} full-resolution pixels; "
              f"nearest depth from search range alone ~{Q[2, 3]/denominator:.2f} m.")

    # Upload rectification maps to GPU once at startup
    gpu_map1x = cv2.cuda_GpuMat(); gpu_map1x.upload(map1x)
    gpu_map1y = cv2.cuda_GpuMat(); gpu_map1y.upload(map1y)
    gpu_map2x = cv2.cuda_GpuMat(); gpu_map2x.upload(map2x)
    gpu_map2y = cv2.cuda_GpuMat(); gpu_map2y.upload(map2y)

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

        # Only the CPU matcher needs full-resolution frames on the host each loop.
        if args.matcher == "sgbm":
            rectL = gpu_rectL.download()
            rectR = gpu_rectR.download()

        # ── Rectified view — resize to 960×540 each on GPU → 1920×540 side-by-side
        bgrL = cv2.cuda.cvtColor(cv2.cuda.resize(gpu_rectL, (960, 540)), cv2.COLOR_GRAY2BGR).download()
        bgrR = cv2.cuda.cvtColor(cv2.cuda.resize(gpu_rectR, (960, 540)), cv2.COLOR_GRAY2BGR).download()
        both_bgr = np.hstack([cv2.rotate(bgrL, cv2.ROTATE_180),
                             cv2.rotate(bgrR, cv2.ROTATE_180)])
        for y in range(0, both_bgr.shape[0], 30):
            cv2.line(both_bgr, (0, y), (both_bgr.shape[1] - 1, y), (0, 255, 0), 1)
        cv2.imshow("Rectified", both_bgr)

        if args.matcher == "sgbm":
            first, second = (rectR, rectL) if reverse_match else (rectL, rectR)
            disp_f = compute_sgbm(matcher, first, second, args.sgbm_scale, postfilter)
        elif args.matcher == "sgm":
            first, second = ((gpu_rectR, gpu_rectL) if reverse_match
                             else (gpu_rectL, gpu_rectR))
            disp_f = compute_cuda_sgm(matcher, first, second, sgm_config)
        else:
            match_first, match_second = ((gpu_rectR, gpu_rectL) if reverse_match
                                         else (gpu_rectL, gpu_rectR))
            gpu_disp = matcher.compute(match_first, match_second, stream)
            stream.waitForCompletion()
            # CUDA BM returns uint8 pixel disparities, without the CPU 16x scale.
            disp_f = gpu_disp.download().astype(np.float32)
        depth_m, valid = depth_from_disparity(disp_f, Q, reverse_match)

        # ── Disparity visualisation (scaled to the configured search range) ─────
        disp_vis = disparity_preview(disp_f, valid, max_disparity)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
        disp_color[~valid] = 0
        disp_display = cv2.rotate(cv2.resize(disp_color, (960, 540), interpolation=cv2.INTER_NEAREST), cv2.ROTATE_180)
        cv2.imshow("Disparity", disp_display)

        depth_display = cv2.rotate(cv2.resize(colorize_depth(depth_m, valid), (960, 540), interpolation=cv2.INTER_NEAREST), cv2.ROTATE_180)
        cv2.imshow("Depth (m)", depth_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            captured_at = datetime.now().astimezone()
            # Local date/time; microseconds distinguish captures in the same second.
            ts = captured_at.strftime("%Y%m%d_%H%M%S_%f")
            if args.matcher != "sgbm":
                rectL = gpu_rectL.download()
                rectR = gpu_rectR.download()
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"rectL_{ts}.png"), rectL)
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"rectR_{ts}.png"), rectR)
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"depth_{ts}.png"), disp_vis)
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"depth_color_{ts}.png"), depth_display)
            np.save(os.path.join(CAPTURE_DIR, f"depth_m_{ts}.npy"), depth_m)
            np.save(os.path.join(CAPTURE_DIR, f"disparity_px_{ts}.npy"), disp_f)
            with open(os.path.join(CAPTURE_DIR, f"settings_{ts}.json"), "w") as f:
                json.dump({"captured_at": captured_at.isoformat(),
                           "calibration": calib_file, "matcher": args.matcher,
                           "matching_scale": matching_scale,
                           "num_disparities_matching": matcher.getNumDisparities(),
                           "num_disparities_full_resolution": args.num_disparities,
                           "disparity_preview_max_px": max_disparity,
                           "wls": args.wls,
                           "reference_camera": "right" if reverse_match else "left",
                           "depth_m_orientation": "sensor", "invalid_depth_m": 0,
                           "depth_png_contents": "disparity preview scaled to 0..255",
                           "disparity_px_orientation": "sensor",
                           "disparity_px_units": "full-resolution pixels; nonpositive is invalid"}, f, indent=2)
            print(f"[+] Captured pair #{capture_count} → captures/")
            capture_count += 1

    pL.terminate()
    pR.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
