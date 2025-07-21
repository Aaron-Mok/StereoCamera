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
gain_values = np.linspace(1.0, 8.0, 30) 

# Prompt user to cover the lens
input("Please cover the camera lens completely, then press ENTER to begin...")

black_offsets = []

for gain in gain_values:
    set_camera_controls(cam_obj, gain=gain, exposure=1000)  # 1 ms exposure time
    time.sleep(0.5)  # Allow time for settings to take effect
    raw_8bit = capture_raw_image(cam_obj)
    raw_unstrided_8bit =  correct_stride_padding(raw_8bit, img_width_px=output_im_size_px[0])
    # cv2.imshow("Raw Image", raw_unstrided_8bit)
    # cv2.waitKey(1)
    black_level = np.mean(raw_unstrided_8bit)
    black_offsets.append(black_level)

# Plotting
plt.figure()
plt.style.use('style.mplstyle')
plt.plot(gain_values, black_offsets, 'o-', label='Measured black level')
plt.xlabel("Analog Gain")
plt.ylabel("Mean Black Offset")
plt.title("Black Offset vs. Analog Gain")
plt.grid(True)
plt.legend()
plt.show()
# cv2.destroyAllWindows()