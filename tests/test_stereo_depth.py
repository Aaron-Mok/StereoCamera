"""Offline checks for disparity units, orientation, and missing-depth display."""
import unittest

import numpy as np

from Camera_Jetson_PiHQCam_Stereo import (
    colorize_depth, compute_sgbm, depth_from_disparity, make_sgbm,
)


class StereoDepthTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
