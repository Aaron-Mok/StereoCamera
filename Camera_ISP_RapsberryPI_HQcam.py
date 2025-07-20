from picamera2 import Picamera2
import cv2
import numpy as np
from ISP import *
from camera_utils import *

# Initialize camera
cam_obj, output_im_size_px = initialize_camera(camera_id = 0, image_size_px = (2028, 1520))

while True:
    # Capture raw image form camera
    raw = capture_raw_image(cam_obj)

    # Stride/padding correction
    raw_unstrided =  unstride_padding(raw)

    raw = raw[:, 1::2]
    raw_float = raw[:,:img_width_px-1]
    raw_8bit = raw_float.astype(np.uint8)
    # cv2.imshow("Raw Image", raw_8bit)

    # TODO: Add black offset subtraction. Calibration script written. Need Curve fitting and get the black offset for each analogue gain value.

    # lens shading correction
    flat_field_img = np.load("./flat_field_output/flat_field_average.npy")
    flat_field_img_blurred = cv2.GaussianBlur(flat_field_img, (11, 11), 0)
    flat_field_img_normalize = flat_field_img_blurred/flat_field_img_blurred.max()
    lens_shading_correction = flat_field_img_normalize
    flat_field_img_normalize_8bit = (flat_field_img_normalize*255).astype(np.uint8)
    # cv2.imshow("Flat Field Image", flat_field_img_normalize_8bit)

    # Define contour levels (e.g., pixel values to outline)
    levels = [150, 200, 250]  # Adjust as needed for contrast

    # Copy for drawing
    overlay_img = flat_field_img_normalize_8bit.copy()

    for level in levels:
        ret, thresh = cv2.threshold(overlay_img, level, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, contours, -1, color=0, thickness=1)  # black contours (0)
    
    # cv2.imshow("Gray Image with Black Contours", overlay_img)

    lens_shade_corrected_float = (raw_float * lens_shading_correction)
    lens_shade_corrected_float_8bit = (lens_shade_corrected_float).astype(np.uint8)

    # Demosaic
    rgb_image = cv2.cvtColor(lens_shade_corrected_float_8bit, cv2.COLOR_BAYER_BGGR2BGR) # This is BIlinear. There are three options: Bilinear, Edge Aware, and Variable Number of Gradients

    # White Balance
    # rgb_image_awb = gray_world_awb(rgb_image.astype(np.float32) / 255.0)
    rgb_image_awb = from_calibration_wb(rgb_image.astype(np.float32) / 255.0)

    # Gamma correction
    gamma = 2.2
    corrected = np.power(rgb_image_awb, 1.0 / gamma)
    gamma_corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)

    # Display
    cv2.imshow("Gamma Corrected", gamma_corrected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
