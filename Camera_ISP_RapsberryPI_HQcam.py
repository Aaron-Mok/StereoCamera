from picamera2 import Picamera2
import cv2
import os
import numpy as np
from ISP import *
from camera_utils import *
from conversion_utils import *
# from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, demosaicing_CFA_Bayer_bilinear
# this is too slow for real time processing, so we use OpenCV's demosaicing instead

# Initialize camera and capture folder
cam_obj, output_im_size_px = initialize_camera(camera_id = 0, image_size = (2028, 1520))
capture_dir = "captures"
os.makedirs(capture_dir, exist_ok=True)
capture_count = 1

while True:
    # Capture raw image form camera
    raw_u8 = capture_raw_image(cam_obj)

    # Stride/padding correction
    raw_unstrided_u8 =  correct_stride_padding(raw_u8, img_width_px=output_im_size_px[0])
    # cv2.imshow("raw_unstrided_8bit", raw_unstrided_8bit)

    # Add black offset subtraction
    raw_unstrided_blckoff_f255 = apply_black_offset(raw_unstrided_u8, offset=15.8)  # Use the measured black offset value

    # lens shading correction
    #TODO: this needs to be applied per channel. and shading measurmenet per color channel 
    # correction_map = get_correction_map("./Calibration_output/flat_field_average.npy", show_map=1)
    # raw_unstrided_corrected_u8 = (raw_unstrided_blckoff_f255 * correction_map).astype(np.uint8)
    # cv2.imshow("raw_unstrided_corrected_8bit", raw_unstrided_corrected_8bit)

    # Demosaic
    raw_unstrided_corrected_u8 = normalize01_to_8bit(raw_unstrided_blckoff_f255/ 255.0)
    rgb_image_u8 = cv2.cvtColor(raw_unstrided_corrected_u8, cv2.COLOR_BAYER_BGGR2BGR) # This is Bilinear. There are three options: Bilinear, Edge Aware, and Variable Number of Gradients
    cv2.imshow("linear_RGB", rgb_image_u8)

    # White Balance
    rgb_image_f01 = bit8_to_normalize01(rgb_image_u8)
    rgb_image_awb_f01 = gray_world_awb(rgb_image_f01)
    # rgb_image_awb = from_calibration_wb(rgb_image.astype(np.float32) / 255.0)

    # Gamma correction
    # rgb_image_awb_gamma_f01 = apply_gamma_correction(rgb_image_awb_f01, gamma=2.2)
    rgb_image_awb_gamma_f01 = apply_sRGB_gamma_encoding(rgb_image_awb_f01)
    rgb_image_awb_gamma_8u = normalize01_to_8bit(rgb_image_awb_gamma_f01)

    # Display
    cv2.imshow("Gamma Corrected", rgb_image_awb_gamma_8u)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        filename = os.path.join(capture_dir, f"capture_{capture_count:04d}.png")
        cv2.imwrite(filename, rgb_image_u8)
        print(f"[INFO] Captured image saved to {filename}")
        capture_count += 1

cv2.destroyAllWindows()
