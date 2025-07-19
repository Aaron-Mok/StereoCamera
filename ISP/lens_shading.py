import numpy as np
import cv2

def get_correction_map(flat_field_image_dir, show_map):
    """
    Generate a lens shading correction map from a flat-field image.

    Args:
        flat_field_image_dir (str): Path to a saved flat field image in .npy format.

    Returns:
        np.ndarray: Lens shading correction map normalized to [0, 1].
    """
    flat_field_img = np.load(flat_field_image_dir)
    flat_field_img_blurred = cv2.GaussianBlur(flat_field_img, (11, 11), 0)
    correction_map = flat_field_img_blurred / flat_field_img_blurred.max()

    if show_map == 1:
        correction_map_8bit = (correction_map*255).astype(np.uint8)
        levels = [150, 200, 250]


        for level in levels:
            ret, thresh = cv2.threshold(correction_map_8bit, level, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(correction_map_8bit, contours, -1, color=0, thickness=1)  # black contours (0)

    return correction_map