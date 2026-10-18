"""Offline checks for disparity units, orientation, and missing-depth display."""
import unittest
import os
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
from io import StringIO

import cv2
import numpy as np
import Camera_Jetson_PiHQCam_Stereo as viewer

from Camera_Jetson_PiHQCam_Stereo import (
    colorize_depth, compute_sgbm, depth_from_disparity, disparity_preview,
    make_sgbm, make_wls, cuda_sgm_config, make_cuda_sgm, compute_cuda_sgm,
)


class StereoDepthTests(unittest.TestCase):
    def test_cuda_sgm_configuration_keeps_requested_search_range(self):
        for requested in (64, 128, 256, 480, 512, 640, 896):
            with self.subTest(requested=requested):
                config = cuda_sgm_config((1920, 1080), requested)
                self.assertIn(config["num_disparities"], (64, 128, 256))
                self.assertGreaterEqual(config["num_disparities"] / config["scale"], requested)
                self.assertEqual(config["scale"], config["size"][0] / 1920)
        close = cuda_sgm_config((1920, 1080), 640)
        self.assertEqual(close["size"], (768, 432))
        with self.assertRaises(ValueError):
            cuda_sgm_config((1920, 1080), 1920)
        with self.assertRaises(ValueError):
            cuda_sgm_config((1920, 1080), 255)

    def test_wider_search_recovers_close_surface(self):
        rng = np.random.default_rng(52)
        first = rng.integers(0, 256, (160, 1280), dtype=np.uint8)
        second = np.zeros_like(first)
        second[:, :-448] = first[:, 448:]
        q = np.zeros((4, 4), dtype=np.float64)
        q[2, 3], q[3, 2] = 1282.9049, -4.98832216
        # 448 full-resolution pixels corresponds to about 0.57 m on this rig.
        # The old 256 px search cannot reach this shift.
        for scale in (0.5, 1.0):
            with self.subTest(scale=scale):
                disparity = compute_sgbm(make_sgbm(scale, 640), first, second, scale)
                roi = disparity[30:-30, 700:-80]
                self.assertGreater(np.mean(roi > 0), 0.95)
                self.assertAlmostEqual(float(np.median(roi)), 448, delta=0.25)
                depth, _ = depth_from_disparity(disparity, q, True)
                self.assertAlmostEqual(float(np.median(depth[30:-30, 700:-80])),
                                       0.574, delta=0.01)

    @unittest.skipUnless(hasattr(cv2, "ximgproc") and
                         hasattr(cv2.ximgproc, "createDisparityWLSFilter"),
                         "OpenCV ximgproc is optional")
    def test_wls_keeps_known_shift_and_rejects_unsearchable_border(self):
        rng = np.random.default_rng(61)
        first = rng.integers(0, 256, (160, 960), dtype=np.uint8)
        second = np.zeros_like(first)
        second[:, :-64] = first[:, 64:]
        matcher = make_sgbm(0.5)
        disparity = compute_sgbm(matcher, first, second, 0.5, make_wls(matcher))
        roi = disparity[30:-30, 300:-100]
        self.assertGreater(np.mean(roi > 0), 0.95)
        self.assertAlmostEqual(float(np.median(roi)), 64, delta=0.5)
        self.assertTrue(np.all(disparity[:, :240] <= 0))

    def test_close_disparities_do_not_saturate_preview(self):
        disparity = np.array([[0, 128, 256, 384, 512, 638]], dtype=np.float32)
        preview = disparity_preview(disparity, disparity > 0, 638)
        self.assertEqual(int(preview[0, 0]), 0)
        self.assertTrue(np.all(np.diff(preview[0].astype(int)) > 0))
        self.assertEqual(int(preview[0, -1]), 255)

    def test_known_shift_keeps_depth_at_both_matching_scales(self):
        rng = np.random.default_rng(42)
        first = rng.integers(0, 256, (240, 640), dtype=np.uint8)
        second = np.zeros_like(first)
        second[:, :-64] = first[:, 64:]
        q = np.zeros((4, 4), dtype=np.float64)
        q[2, 3], q[3, 2] = 1000, 5  # 1000 px focal length, 0.2 m baseline
        for scale in (0.5, 1.0):
            with self.subTest(scale=scale):
                disparity = compute_sgbm(make_sgbm(scale), first, second, scale)
                self.assertEqual(disparity.shape, first.shape)
                region = disparity[40:-40, 300:-80]
                self.assertGreater(np.mean(region > 0), 0.95)
                self.assertAlmostEqual(float(np.median(region)), 64, delta=0.25)
                depth, _ = depth_from_disparity(disparity, q, False)
                self.assertAlmostEqual(float(np.median(depth[40:-40, 300:-80])),
                                       3.125, delta=0.02)

    def test_reversed_geometry_and_principal_point_offset(self):
        disparity = np.array([[64, 128]], dtype=np.float32)
        q = np.zeros((4, 4), dtype=np.float64)
        q[2, 3], q[3, 2], q[3, 3] = 1000, 5, 10
        depth, valid = depth_from_disparity(disparity, q, False)
        np.testing.assert_allclose(depth, 1000 / (5 * disparity + 10))
        q[3, 2] = -5
        reversed_depth, reversed_valid = depth_from_disparity(disparity, q, True)
        np.testing.assert_array_equal(depth, reversed_depth)
        np.testing.assert_array_equal(valid, reversed_valid)

    def test_missing_depth_is_black_and_far_depth_has_color(self):
        disparity = np.array([[-1, 0, np.nan, 50]], dtype=np.float32)
        q = np.zeros((4, 4), dtype=np.float64)
        q[2, 3], q[3, 2] = 1000, 5
        depth, valid = depth_from_disparity(disparity, q, False)
        np.testing.assert_array_equal(valid, [[False, False, False, True]])
        np.testing.assert_array_equal(depth[~valid], 0)
        colored = colorize_depth(depth, valid)
        self.assertTrue(np.all(colored[~valid] == 0))
        self.assertTrue(np.any(colored[valid] != 0))


@unittest.skipUnless(os.environ.get("STEREO_TEST_CUDA") == "1",
                     "Set STEREO_TEST_CUDA=1 on a machine with CUDA to run GPU integration")
class CudaSGMTests(unittest.TestCase):
    def test_viewer_captures_cuda_result_with_saved_camera_frames(self):
        rng = np.random.default_rng(23)
        right = rng.integers(0, 256, (240, 1280), dtype=np.uint8)
        left = np.zeros_like(right)
        left[:, :-480] = right[:, 480:]
        map_y, map_x = np.indices(right.shape, dtype=np.float32)
        q = np.zeros((4, 4), dtype=np.float64)
        q[2, 3], q[3, 2] = 1282.9049, -4.98832216
        processes = []

        def initialize(device, *_):
            p = SimpleNamespace(frame=left if device == viewer.LEFT_DEVICE else right,
                                terminate=Mock())
            processes.append(p)
            return p, 0

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(viewer, "CAPTURE_DIR", directory), \
             patch.object(viewer, "load_calibration", return_value=(map_x, map_y, map_x, map_y, q, 1282.9049, 0.20047)), \
             patch.object(viewer, "initialize_camera_jetson", side_effect=initialize), \
             patch.object(viewer, "capture_raw_frame_jetson", side_effect=lambda p, *_: p.frame), \
             patch.object(viewer, "raw_to_gray8", side_effect=lambda frame: frame), \
             patch.object(viewer.cv2, "imshow"), \
             patch.object(viewer.cv2, "waitKey", side_effect=[ord('c'), ord('q')]), \
             patch.object(viewer.cv2, "destroyAllWindows"), \
             patch.object(viewer.sys, "argv", ["viewer", "--matcher", "sgm", "--num-disparities", "640"]), \
             redirect_stdout(StringIO()):
            viewer.main()
            settings = list(Path(directory).glob('settings_*.json'))
            self.assertEqual(len(settings), 1)
            self.assertRegex(settings[0].name, r'^settings_\d{8}_\d{6}_\d{6}\.json$')
            metadata = json.loads(settings[0].read_text())
            self.assertEqual(metadata["matcher"], "sgm")
            self.assertEqual(metadata["reference_camera"], "right")
            self.assertEqual(metadata["matching_scale"], 0.4)
            self.assertEqual(metadata["num_disparities_matching"], 256)
            self.assertFalse(metadata["wls"])
            tag = settings[0].stem.removeprefix('settings_')
            np.testing.assert_array_equal(cv2.imread(str(Path(directory, f'rectR_{tag}.png')), 0), right)
            np.testing.assert_array_equal(cv2.imread(str(Path(directory, f'rectL_{tag}.png')), 0), left)
            depth = np.load(Path(directory, f'depth_m_{tag}.npy'))
            self.assertAlmostEqual(float(np.median(depth[40:-40, 700:-80])), 257.18164 / 480, delta=0.02)
            self.assertEqual(np.load(Path(directory, f'disparity_px_{tag}.npy')).shape, right.shape)
        for p in processes:
            p.terminate.assert_called_once()

    def test_known_shifts_keep_metric_depth_on_gpu(self):
        self.assertGreater(cv2.cuda.getCudaEnabledDeviceCount(), 0)
        rng = np.random.default_rng(82)
        first = rng.integers(0, 256, (240, 1280), dtype=np.uint8)
        for shift, requested, reverse in [(80, 256, False), (480, 640, True)]:
            with self.subTest(shift=shift):
                second = np.zeros_like(first)
                second[:, :-shift] = first[:, shift:]
                gpu_first = cv2.cuda_GpuMat(); gpu_first.upload(first)
                gpu_second = cv2.cuda_GpuMat(); gpu_second.upload(second)
                config = cuda_sgm_config((1280, 240), requested)
                disparity = compute_cuda_sgm(make_cuda_sgm(config), gpu_first, gpu_second, config)
                self.assertEqual(disparity.shape, first.shape)
                roi = disparity[40:-40, requested + 60:-80]
                self.assertGreater(np.mean(roi > 0), 0.9)
                self.assertAlmostEqual(float(np.median(roi)), shift, delta=1.0)
                q = np.zeros((4, 4), dtype=np.float64)
                q[2, 3], q[3, 2] = 1282.9049, -4.98832216 if reverse else 4.98832216
                depth, _ = depth_from_disparity(disparity, q, reverse)
                self.assertAlmostEqual(float(np.median(depth[40:-40, requested + 60:-80])),
                                       257.18164 / shift, delta=0.02)


if __name__ == "__main__":
    unittest.main()
