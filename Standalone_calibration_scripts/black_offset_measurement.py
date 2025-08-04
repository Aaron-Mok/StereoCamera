import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
import time
import cv2

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ISP import *
from camera_utils import *

def exp_saturation(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

# Initialize camera
cam_obj, output_im_size_px = initialize_camera(camera_id = 0, image_size = (2028, 1520))

# Gain sweep values (in analog gain units)
# gain_values = np.linspace(4.0)
# Prompt user to cover the lens
input("Please cover the camera lens completely, then press ENTER to begin...")

black_offsets = []
# for gain in gain_values:
gain = 4.0
set_camera_controls(cam_obj, gain=gain, exposure=500)  # 500 us exposure time
time.sleep(1.5)  # Allow time for settings to take effect
raw_u8 = capture_raw_image(cam_obj)
raw_u16 =  unpack_and_trim_raw(raw_u8, img_width_px=output_im_size_px[0], bit_depth=16)
# raw_unstrided_8bit =  correct_stride_padding(raw_8bit, img_width_px=output_im_size_px[0])
# cv2.imshow("Raw Image", raw_u16)
# cv2.waitKey(1)
black_level = np.mean(raw_u16)
print(f"Black Level: {black_level:.2f}")
black_offsets.append(black_level)

# Plotting
# plt.figure()
# plt.style.use('style.mplstyle')
# plt.plot(gain_values, black_offsets, 'o-', label='Measured black level')
# plt.xlabel("Analog Gain")
# plt.ylabel("Mean Black Offset")
# plt.title("Black Offset vs. Analog Gain")
# plt.grid(True)
# plt.legend()
# plt.show()
# cv2.destroyAllWindows()

plt.figure()
plt.hist(raw_u16.flatten(), bins=30, range=(3500, 4500), color='gray')
plt.title(f"Histogram at Gain {gain:.2f}")
plt.xlabel("Pixel Value (Gray Level)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()