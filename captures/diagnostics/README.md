# Edge-only depth investigation

The current CUDA block matcher reproduces the saved sparse output. On the same
new-calibration image pair, half-resolution CPU SGBM produces substantially more
continuous surfaces. This is an improvement in preview coverage, not a measurement
of absolute depth accuracy.

## Inputs and measurements

- New calibration: `stereo_params_charuco.yml`, capture `1792283796168`.
- Previous calibration: `Calibration_output/20260620_stereo_params.yml`, capture
  `1792283838691`.
- Matching order for both calibrations is rectified right first, left second,
  because their Q matrices encode negative left-minus-right disparity.
- The recomputed CUDA result exactly matched the saved disparity image.

| Matcher on the new pair | Positive-disparity coverage | Matched feature locations / 97 | Median disparity error at matched features |
|---|---:|---:|---:|
| Existing CUDA BM, block 5, texture 20, uniqueness 20 | 24.40% | 48 | 0.99 px |
| CUDA BM, block 15, texture 3, uniqueness 10 | 29.46% | 79 | 0.54 px |
| CPU SGBM, full resolution | 43.59% | 90 | 0.45 px |
| CPU SGBM, half resolution | 68.85% | 97 | 0.52 px |

Coverage excludes the first 256 full-resolution columns, which cannot be searched
over the whole configured disparity range. Positive disparity can still be a false
match. The feature check uses independent SIFT descriptor matches, a RANSAC
fundamental-matrix filter, vertical separation below 3 px, and positive disparity
inside the searchable area. These sparse, textured locations are not ground truth
and do not validate plain wall regions. Half-resolution disparity is converted
back to full-resolution pixel units before comparison or depth conversion.

For the saved room pairs, vertical feature mismatch RMS was 1.01 px for the new
calibration and 11.55 px for the old calibration (107 and 114 RANSAC inliers,
respectively). These are separate captures and sparse scene measurements; they
do not replace the user's 2.50 px ChArUco validation result.

The new pair has low-contrast areas and different brightness between cameras
(median intensity 42 left, 56 right, on a 0–255 scale). Lowering rejection
thresholds mainly adds speckles. SGBM's smoothness constraints and half-resolution
matching help this preview, while some holes and likely incorrect patches remain.
See the [OpenCV SGBM documentation](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
for the smoothness and rejection parameters. CUDA BM has its own texture filter
and output representation, as shown in the
[OpenCV CUDA source](https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudastereo/src/stereobm.cpp).
The earlier conversation's 16% result was CPU BM; the directly reproduced GPU
baseline is 24.40%.

## Viewer changes and use

```bash
python3 Camera_Jetson_PiHQCam_Stereo.py --calibration stereo_params_charuco.yml --matcher sgbm
```

SGBM defaults to half resolution; use `--sgbm-scale 1` for full resolution.
Use `--matcher bm` for the original CUDA matcher. Black now means missing depth;
valid depth keeps the same 0.3–5 m color range. Rectified previews retain left/right
window order when rotated upright. The full-resolution search range corresponds
to a closest distance of approximately 1.01 m for the new calibration; objects
closer than that may have no usable match.

Pressing `c` also saves an upright `depth_color_<timestamp>.png`, unrotated
`depth_m_<timestamp>.npy` (float metres, zero for missing), and
`settings_<timestamp>.json`. The legacy `depth_<timestamp>.png` remains an 8-bit
disparity image.

## Verification

The viewer's SGBM function produces exactly the same disparity as the experiment.
Three offline tests pass: known 64 px synthetic shift at both matching scales,
depth conversion with both disparity signs and principal-point offsets, and black
rendering of missing values. Both actual CUDA BM and CPU SGBM were exercised on
saved images. The live camera/GUI loop has not been tested during this change.

`bm_vs_sgbm.png` uses the same image pair, calibration, orientation, and depth color
range for both panels. Full matcher measurements are in `matcher_metrics.json`.
