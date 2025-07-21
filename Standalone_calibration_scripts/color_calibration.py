import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ISP import *
from camera_utils import *

def white_balance_to_gray_patch(img_rgb_f01, measured_patch_rgb_f01, reference_patch_rgb_f01):
    """
    Apply white balance so that measured_patch_rgb becomes reference_patch_rgb.
    All inputs are in linear RGB [0,1].
    """
    gains = reference_patch_rgb_f01 / np.clip(measured_patch_rgb_f01, 1e-6, None)
    gains = gains / gains[1]  # normalize green channel gain to 1
    print(f"🎨 White balance gains: {gains}")
    img_wb_f01 = img_rgb_f01 * gains
    return np.clip(img_wb_f01, 0, 1)

def srgb_to_linear(srgb_f01):
    """Convert sRGB (0-1 range) to linear RGB using IEC 61966-2-1"""
    srgb_f01 = np.clip(srgb_f01, 0, 1)
    linear_rgb = np.where(srgb_f01 <= 0.04045,
                      srgb_f01 / 12.92,
                      ((srgb_f01 + 0.055) / 1.055) ** 2.4)
    return np.clip(linear_rgb, 0, 1)

clicked_points = []

img_bgr = cv2.imread("./captures/SpyderCHECKR_linear_rgb_before_wb.png")

# Resize for display
screen_width = 1280  # adjust as needed
scale = screen_width / img_bgr.shape[1]
display = cv2.resize(img_bgr.copy(), None, fx=scale, fy=scale)

# Callback for mouse clicks
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        # Scale back to original resolution
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        clicked_points.append((orig_x, orig_y))

        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(display, f"{len(clicked_points)}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Click 4 corners (TL, TR, BR, BL)", display)

# Show resized image
cv2.imshow("Click 4 corners (TL, TR, BR, BL)", display)
cv2.setMouseCallback("Click 4 corners (TL, TR, BR, BL)", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now do the perspective correction
src_pts = np.array(clicked_points, dtype=np.float32)
w, h = 600, 400  # virtual grid dimensions
dst_pts = np.array([
    [0, 0], [w, 0],
    [w, h], [0, h]
], dtype=np.float32)

# Perspective transform
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped_bgr = cv2.warpPerspective(img_bgr, M, (w, h))

cv2.imshow("Warped Perspective", warped_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sample color patches
patch_w = w // 6
patch_h = h // 4
measured_rgbs = []
annotated = warped_bgr.copy()

for r in range(4):
    for c in range(6):
        idx = r * 6 + c
        x_center = c * patch_w + patch_w // 2
        y_center = r * patch_h + patch_h // 2

        # Extract patch
        patch_bgr = warped_bgr[y_center-10:y_center+10, x_center-10:x_center+10]
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        avg_rgb = np.mean(patch_rgb.reshape(-1, 3), axis=0)
        measured_rgbs.append(avg_rgb)

        # Draw rectangle around patch
        top_left = (x_center - 10, y_center - 10)
        bottom_right = (x_center + 10, y_center + 10)
        cv2.rectangle(annotated, top_left, bottom_right, color=(0, 255, 0), thickness=1)

        # Draw patch index
        cv2.putText(annotated, str(idx+1), (x_center - 5, y_center - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Show the annotated warped image
cv2.imshow("Sampled Patches", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

measured_rgbs = np.array(measured_rgbs) / 255.0
print("✅ Measured RGBs (normalized):")
print(measured_rgbs)

# Reference sRGB values for datacolor Spyder CHECKR (24 patches)
ground_truth_srgb_f01 = np.array([
    [98, 187, 166], [126, 125, 174], [82, 106, 60], [87, 120, 155], [197, 145, 125], [112, 76, 60],
    [222, 118, 32], [58, 88, 159], [195, 79, 95], [83, 58, 106], [157, 188, 54], [238, 158, 25],
    [0, 127, 159], [192, 75, 145], [245, 205, 0], [186, 26, 51], [57, 146, 64], [25, 55, 135],
    [249, 242, 238], [202, 198, 195], [161, 157, 154], [122, 118, 116], [80, 80, 78], [43, 41, 43]
], dtype=np.float32) / 255.0

ground_truth_linear_rgb_f01 = srgb_to_linear(ground_truth_srgb_f01)

gray_patch_index = 20  # Patch #21
measured_gray_rgb_f01 = measured_rgbs[gray_patch_index]
ground_truth_gray_rgb_f01 = ground_truth_linear_rgb_f01[gray_patch_index]

img_rgb_f01 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img_rgb_wb_f01 = white_balance_to_gray_patch(img_rgb_f01, measured_gray_rgb_f01, ground_truth_gray_rgb_f01)

img_rgb_wb_u8 = normalize01_to_8bit(img_rgb_wb_f01)
cv2.imshow("White Balanced Image", cv2.cvtColor(img_rgb_wb_u8, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()



# --- Step 4: Solve for 3x3 Color Correction Matrix ---
# Least squares solution: ref = A @ measured
# A, _, _, _ = np.linalg.lstsq(measured_rgbs, reference_rgbs, rcond=None)
# print("Color Correction Matrix:\n", A)

# # --- Step 5: Apply to your image ---
# img_corrected = img.astype(np.float32) / 255.0
# reshaped = img_corrected.reshape((-1, 3))
# corrected = np.clip(reshaped @ A, 0, 1)
# img_corrected = (corrected.reshape(img.shape) * 255).astype(np.uint8)

# # --- Step 6: Display Before and After ---
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(img)

# plt.subplot(1, 2, 2)
# plt.title("Color Calibrated")
# plt.imshow(img_corrected)

# plt.tight_layout()
# plt.show()