from picamera2 import Picamera2
import cv2
import numpy as np
from ISP import *
from camera_utils import *

# Initialize camera
cam_obj, output_im_size_px = initialize_camera(camera_id = 0, image_size = (2028, 1520))

while True:
    # Capture raw image form camera
    raw_8bit = capture_raw_image(cam_obj)

    # Stride/padding correction
    raw_unstrided_8bit =  correct_stride_padding(raw_8bit, img_width_px=output_im_size_px[0])
    cv2.imshow("raw_unstrided_8bit", raw_unstrided_8bit)

    # TODO: Add black offset subtraction. Calibration script written. Need Curve fitting and get the black offset for each analogue gain value.

    # lens shading correction
    correction_map = get_correction_map("./flat_field_output/flat_field_average.npy", show_map=1)
    raw_unstrided_corrected_8bit = (raw_unstrided_8bit * correction_map).astype(np.uint8)
    cv2.imshow("raw_unstrided_corrected_8bit", raw_unstrided_corrected_8bit)

    # Demosaic
    rgb_image_8bit = cv2.cvtColor(raw_unstrided_corrected_8bit, cv2.COLOR_BAYER_BGGR2BGR) # This is Bilinear. There are three options: Bilinear, Edge Aware, and Variable Number of Gradients

    # White Balance
    rgb_image = bit8_to_normalize01(rgb_image_8bit)
    rgb_image_awb = gray_world_awb(rgb_image)
    # rgb_image_awb = from_calibration_wb(rgb_image.astype(np.float32) / 255.0)

    # Gamma correction
    rgb_image_awb_gamma = apply_gamma_correction(rgb_image_awb, gamma=2.2)
    rgb_image_awb_gamma_8bit = normalize01_to_8bit(rgb_image_awb_gamma)

    # Display
    cv2.imshow("Gamma Corrected", rgb_image_awb_gamma_8bit)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
