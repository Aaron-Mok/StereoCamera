# CUDA StereoSGM trial

Run from the repository root with the new ChArUco calibration:

```bash
python3 Camera_Jetson_PiHQCam_Stereo.py \
  --calibration stereo_params_charuco.yml \
  --matcher sgm --num-disparities 640
```

Matching now runs on the GPU. Depth conversion and color rendering still use
the CPU. Omit `--wls`: the viewer currently supports that filter only with CPU
`--matcher sgbm`. Press `c` to save the same image, disparity, depth, and settings
files as before, with local `YYYYMMDD_HHMMSS_microseconds` timestamps.

## Search range and units

CUDA SGM supports 64, 128, or 256 disparities at its working resolution. The
viewer uses half resolution where possible and reduces further for wider search
ranges. A 640-pixel request on this 1920 x 1080 calibration uses 768 x 432 images
and 256 matching disparities. The resulting search ceiling is about 638 original
pixels, corresponding to approximately 0.40 m on this rig. This is a search limit,
not a guarantee of usable depth at that distance.

The matcher uses four aggregation directions, P1=10, P2=120, and uniqueness=10.
Its signed, 16-bit output is divided by 16 and by the actual horizontal image
scale before depth conversion. Nearest-neighbour upsampling preserves invalid
pixels. The existing Q-sign handling selects the right camera as the reference
for this calibration. Capture settings record the actual scale and disparity
count used by the matcher.

See the [OpenCV CUDA SGM API](https://docs.opencv.org/4.x/d1/ddc/classcv_1_1cuda_1_1StereoSGM.html)
and [implementation](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudastereo/src/stereosgm.cpp)
for the supported ranges, output representation, and matching parameters.

## Saved-image comparison

Both modes used the same rectified images, calibration, 640-pixel requested search
range, and no WLS. CPU SGBM used half resolution; CUDA SGM used 0.4 resolution.
Timings are the median of five runs after one warm-up. They include resizing,
matching, unit conversion, full-size upsampling, and the CUDA disparity download.
They exclude camera capture, rectification, initial input upload, depth coloring,
and GUI display, so they are not live frame rates.

| Scene | CPU SGBM | CUDA SGM | CPU positive coverage | CUDA positive coverage |
|---|---:|---:|---:|---:|
| Room (`1792283796168`) | 187.33 ms | 43.78 ms | 64.47% | 47.89% |
| Hand (`1792347435584`) | 199.66 ms | 27.31 ms | 51.64% | 40.95% |

Coverage uses the common area excluding the first 640 sensor columns. Positive
disparity can still be wrong; coverage does not measure accuracy. CUDA matching
was faster in these trials but produced fewer positive estimates in this area.
Holes and streaks remain, and calibration errors and occlusions still affect both
modes.

- [Room comparison](cuda_sgm_room.png)
- [Hand comparison](cuda_sgm_hand.png)
- [Measurements](cuda_sgm_metrics.json)

## Verification

All nine tests passed with actual CUDA execution enabled:

```bash
STEREO_TEST_CUDA=1 python3 -m unittest discover -s tests -v
```

Synthetic shifted pairs check full-resolution disparity units and metric depth,
including a roughly 0.54 m surface. A viewer integration test exercises GPU
rectification, SGM, depth conversion, capture filenames, metadata, and saved
arrays using synthetic frames and mocked camera/GUI interfaces. The live camera
feed and GUI have not been exercised for this change. Without the environment
variable, the two GPU tests are skipped.
