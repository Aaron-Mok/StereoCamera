# Close-range depth and horizontal streaks

The ChArUco calibration has focal × baseline = 257.18 pixel-metres. A point at
1 m needs about 257 px disparity; at 0.5 m it needs about 514 px. The initial
256 px search could not cover these points. At half resolution its actual
search was 0–127 matching pixels, or approximately 0–254 original pixels.

The viewer now accepts a wider search range and optional edge-aware WLS filtering:

```bash
python3 Camera_Jetson_PiHQCam_Stereo.py --calibration stereo_params_charuco.yml --matcher sgbm --num-disparities 640 --wls
```

At half resolution this searches 320 disparity values, reaching approximately
638 original pixels. The theoretical near limit from search range alone is
0.403 m, leaving headroom for testing around 0.5–0.6 m. This does not guarantee
depth at that distance: overlap, occlusion, texture, focus, alignment, and camera
timing still matter. Wider searches also make the unmatched border larger and
increase computation. CUDA BM remains limited to 256 disparity values; use SGBM
for the wider range.

WLS uses an additional reverse disparity map and the reference camera image to
smooth disparity while respecting visible edges. Lambda is 8000 and sigmaColor
is 1.5. Estimates with WLS confidence below 128/255 are suppressed. This reduces
streaks in the saved room example but also produces more missing regions.
Filtering can propagate incorrect estimates and does not repair the remaining
calibration error. See the [OpenCV filtering tutorial](https://docs.opencv.org/4.x/d3/d14/tutorial_ximgproc_disparity_filtering.html).

Omit `--wls` to compare the raw SGBM result at the same search range. No default
matcher or calibration selection changed.

`wls_comparison.png` shows the same saved new-calibration pair before and after
filtering, using the same depth color range. On this pair, half-resolution
matching/filtering took approximately 0.28 s without WLS and 0.50 s with WLS
in one offline run; these are not live camera frame rates. Positive coverage in
the common search region was 64.5% without WLS and 48.3% after confidence masking.
There is no ground-truth depth for this scene.

The disparity preview now scales to the configured range, avoiding saturation
above 255 px. Full pixel disparities are saved in `disparity_px_<timestamp>.npy`;
depth in metres is saved separately as before. The color scale alone does not
change the matcher's near limit.

Six offline tests pass. New checks recover a synthetic 448 px shift (approximately
0.574 m with this calibration) at both matching scales, verify WLS preserves a
known shift while masking the unsearchable border, and verify close disparities
remain distinguishable in the preview. Both viewer processing paths reproduce
the saved-image experiment. A live close-range scene has not been tested here.
