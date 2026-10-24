# CUDA StereoSGM trial

Run from the repository root with the new ChArUco calibration:

```bash
python3 Camera_Jetson_PiHQCam_Stereo.py \
  --calibration Calibration_output/stereo_params_charuco.yml \
  --matcher sgm --num-disparities 640
```

Matching now runs on the GPU. Depth conversion and color rendering still use
the CPU. Omit `--wls`: the viewer currently supports that filter only with CPU
`--matcher sgbm`. Press `c` to save the same image, disparity, depth, and settings
files as before, with local `YYYYMMDD_HHMMSS_microseconds` timestamps.

For faster viewing with only the depth window, add `--depth-only`:

```bash
python3 Camera_Jetson_PiHQCam_Stereo.py --matcher sgm --num-disparities 640 --depth-only
```

This skips generating both rectified-camera previews, their GPU-to-CPU downloads,
alignment lines, and the disparity color preview. Stereo matching and depth
calculation still run normally. Pressing `c` saves the complete capture set,
including the rectified pair and grayscale disparity preview, which are prepared
on demand. The settings JSON records `depth_only`. Omit the flag to show all
three windows again.

The depth window shows FPS in its upper-left corner, averaged over completed
capture-to-display loops for at least half a second. It includes camera reads,
processing, GUI event handling, and capture-file saving when requested. The
counter starts at `FPS: --` until the first measurement is ready. It also appears
in saved `depth_color` previews; numerical depth and disparity arrays contain
only their original measurement values.

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

During live CUDA SGM preview, depth conversion, invalid-pixel masking, and color
mapping stay on the matching grid (768 x 432 for the command above). Only the
finished color previews are resized to 960 x 540 and rotated for display. This
avoids expanding the disparity to 1920 x 1080 for CPU postprocessing every frame.

Pressing `c` expands disparity, metric depth, and the grayscale disparity preview
to the original 1920 x 1080 dimensions with nearest-neighbour sampling. Numeric
units, sensor orientation, and invalid-depth zeros are preserved. The saved
`depth_color` image remains the displayed upright 960 x 540 preview. Capture
settings also record `depth_processing_size` as `[width, height]`.

See the [OpenCV CUDA SGM API](https://docs.opencv.org/4.x/d1/ddc/classcv_1_1cuda_1_1StereoSGM.html)
and [implementation](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudastereo/src/stereosgm.cpp)
for the supported ranges, output representation, and matching parameters.

## Saved-image comparison

These original matcher timings precede the live preview optimization below.
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

## Preview optimization measurements

For each scene, CUDA SGM was run on the saved rectified pair using the settings
above. The same resulting 768 x 432 disparity array was processed by both paths:
the old path expanded it to 1920 x 1080 before depth conversion and coloring;
the new path converted depth and colors at matching size. Both produced upright
960 x 540 color previews for depth and disparity.

| Scene | Previous CPU postprocessing | Matching-size CPU postprocessing |
|---|---:|---:|
| Room (`1792283796168`) | 162.54 ms | 31.59 ms |
| Hand (`1792347435584`) | 173.22 ms | 33.11 ms |

These are medians of ten alternating measurements after three warm-ups per path.
They include depth conversion, masking, color mapping, and preview resizing and
rotation. They exclude camera I/O, GPU processing, transfers, rectified-camera
previews, GUI display, and file saving. The approximately 5x improvement applies
to this stage, not the complete live frame rate. Expanding the new depth and
grayscale preview arrays reproduced the old saved-array calculation exactly on
both scenes. Directly resizing finished colors can change the display's pixel
sampling slightly compared with the previous two-step resize.

## Verification

All ten tests passed with actual CUDA execution enabled:

```bash
STEREO_TEST_CUDA=1 python3 -m unittest discover -s tests -v
```

Synthetic shifted pairs check full-resolution disparity units and metric depth,
including a roughly 0.54 m surface. A viewer integration test exercises GPU
rectification, SGM, depth conversion, capture filenames, metadata, and saved
arrays using synthetic frames and mocked camera/GUI interfaces. The live camera
feed and GUI have not been exercised for this change. Without the environment
variable, the three GPU tests are skipped.

The integration test also checks that depth and coloring receive matching-size
arrays on every frame, while saved disparity, metric depth, and grayscale
previews retain the original dimensions and values. Synthetic known-shift tests
cover both matching-size and full-size helper outputs.

The depth-only test exercises an ordinary frame, a captured frame, and quitting.
It checks that only the depth window is displayed, the extra camera-preview
conversion is skipped, and the grayscale disparity preview is generated only
for capture. The complete saved capture is checked in both display modes.
