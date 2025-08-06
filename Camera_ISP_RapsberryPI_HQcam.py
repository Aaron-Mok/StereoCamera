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

    # Bit and stride correction
    raw_u16 =  unpack_and_trim_raw(raw_u8, img_width_px=output_im_size_px[0], bit_depth=16)
    # cv2.imshow("raw16", raw_u16)

    # Add black offset subtraction
    raw_unstrided_blckoff_u16 = apply_black_offset(raw_u16, offset=4110)  # Use the measured black offset value

    # lens shading correction
    #TODO: this needs to be applied per channel. and shading measurmenet per color channel 
    # correction_map = get_correction_map("./Calibration_output/flat_field_average.npy", show_map=1)
    # raw_unstrided_corrected_u8 = (raw_unstrided_blckoff_f255 * correction_map).astype(np.uint8)
    # cv2.imshow("raw_unstrided_corrected_8bit", raw_unstrided_corrected_8bit)

    # Demosaic
    linear_bgr_image_u16 = cv2.cvtColor(raw_unstrided_blckoff_u16, cv2.COLOR_BAYER_BGGR2BGR) # This is Bilinear. There are three options: Bilinear, Edge Aware, and Variable Number of Gradients
    img_rgb_linear_f01 = bit16_to_normalize01(cv2.cvtColor(linear_bgr_image_u16, cv2.COLOR_BGR2RGB))
    # cv2.imshow("linear_BGR", linear_bgr_image_u16)

    # White Balance
    # linear_rgb_image_awb_f01 = gray_world_awb(img_rgb_linear_f01)
    # This is calibrated using gray patch and D50 illumination
    wb_gains = np.load("./Calibration_output/wb_gains.npy")
    linear_rgb_image_awb_f01 = img_rgb_linear_f01 * wb_gains

    # Color Correction
    A = np.load("./Calibration_output/color_correction_matrix.npy")
    linear_rgb_image_awb_ccm_f01 = linear_rgb_image_awb_f01 @ A
    
    # Saturation mask to avoid clipping
    saturated_mask = np.any(linear_rgb_image_awb_ccm_f01 > 1.0, axis=-1)
    linear_rgb_image_awb_ccm_f01[saturated_mask] = [1.0, 1.0, 1.0]

    # Gamma correction
    rgb_image_awb_gamma_f01 = linear_to_srgb(linear_rgb_image_awb_ccm_f01)
    rgb_image_awb_gamma_u8 = normalize01_to_8bit(rgb_image_awb_gamma_f01) # to 8 bit for display


    # Display
    bgr_image_awb_gamma_u8 = cv2.cvtColor(rgb_image_awb_gamma_u8, cv2.COLOR_RGB2BGR)
    bgr_image_awb_gamma_u8_flip = cv2.flip(bgr_image_awb_gamma_u8, -1)  # Flip the image vertically
    cv2.imshow("Gamma Corrected", bgr_image_awb_gamma_u8_flip)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        filename = os.path.join(capture_dir, f"raw_u8_{capture_count:02d}.png")
        cv2.imwrite(filename,raw_u8)
        print(f"[INFO] Captured image saved to {filename}")
        filename = os.path.join(capture_dir, f"raw_u16_{capture_count:02d}.png")
        cv2.imwrite(filename,raw_u16)
        print(f"[INFO] Captured image saved to {filename}")
        filename = os.path.join(capture_dir, f"linear_bgr_image_u16_{capture_count:02d}.png")
        cv2.imwrite(filename,linear_bgr_image_u16)
        print(f"[INFO] Captured image saved to {filename}")
        filename = os.path.join(capture_dir, f"linear_bgr_image_awb_u16{capture_count:02d}.png")
        cv2.imwrite(filename,cv2.cvtColor(normalize01_to_16bit(linear_rgb_image_awb_f01), cv2.COLOR_RGB2BGR))
        print(f"[INFO] Captured image saved to {filename}")
        filename = os.path.join(capture_dir, f"linear_rgb_image_awb_ccm_u16{capture_count:02d}.png")
        cv2.imwrite(filename,cv2.cvtColor(normalize01_to_16bit(linear_rgb_image_awb_ccm_f01), cv2.COLOR_RGB2BGR))
        print(f"[INFO] Captured image saved to {filename}")
        filename = os.path.join(capture_dir, f"bgr_image_awb_gamma_u8{capture_count:02d}.png")
        cv2.imwrite(filename,bgr_image_awb_gamma_u8)
        print(f"[INFO] Captured image saved to {filename}")

        capture_count += 1

cv2.destroyAllWindows()
