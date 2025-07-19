import cv2
import numpy as np
import matplotlib.pyplot as plt

clicked_points = []
img = cv2.imread("spyderchecker.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display = img.copy()

# Callback function to record 4 mouse clicks
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(display, f"{len(clicked_points)}", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Click 4 corners (TL, TR, BR, BL)", display)

# Show image and register callback
cv2.imshow("Click 4 corners (TL, TR, BR, BL)", display)
cv2.setMouseCallback("Click 4 corners (TL, TR, BR, BL)", click_event)

print("🖱️ Click the 4 corners of the chart in this order: TL, TR, BR, BL")
while len(clicked_points) < 4:
    if cv2.waitKey(1) & 0xFF == 27:
        break  # Press ESC to quit

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
warped = cv2.warpPerspective(img_rgb, M, (w, h))

# Sample color patches
patch_w = w // 6
patch_h = h // 4
measured_rgbs = []
for r in range(4):
    for c in range(6):
        x = c * patch_w + patch_w // 2
        y = r * patch_h + patch_h // 2
        patch = warped[y-10:y+10, x-10:x+10]
        avg_rgb = np.mean(patch.reshape(-1, 3), axis=0)
        measured_rgbs.append(avg_rgb)

measured_rgbs = np.array(measured_rgbs) / 255.0
print("✅ Measured RGBs (normalized):")
print(measured_rgbs)

# --- Step 3: Reference sRGB values for X-Rite ColorChecker (24 patches) ---
reference_rgbs = np.array([
    [0.400, 0.350, 0.290],
    [0.760, 0.590, 0.500],
    [0.250, 0.320, 0.510],
    [0.240, 0.300, 0.180],
    [0.520, 0.510, 0.730],
    [0.400, 0.740, 0.670],
    [0.840, 0.490, 0.170],
    [0.220, 0.260, 0.650],
    [0.760, 0.350, 0.390],
    [0.260, 0.170, 0.420],
    [0.620, 0.740, 0.250],
    [0.880, 0.640, 0.180],
    [0.110, 0.120, 0.590],
    [0.170, 0.580, 0.220],
    [0.690, 0.210, 0.230],
    [0.910, 0.780, 0.120],
    [0.730, 0.340, 0.580],
    [0.030, 0.520, 0.630],
    [0.955, 0.955, 0.955],
    [0.780, 0.780, 0.780],
    [0.620, 0.620, 0.620],
    [0.480, 0.480, 0.480],
    [0.330, 0.330, 0.330],
    [0.210, 0.210, 0.210],
], dtype=np.float32)

# --- Step 4: Solve for 3x3 Color Correction Matrix ---
# Least squares solution: ref = A @ measured
A, _, _, _ = np.linalg.lstsq(measured_rgbs, reference_rgbs, rcond=None)
print("Color Correction Matrix:\n", A)

# --- Step 5: Apply to your image ---
img_corrected = img.astype(np.float32) / 255.0
reshaped = img_corrected.reshape((-1, 3))
corrected = np.clip(reshaped @ A, 0, 1)
img_corrected = (corrected.reshape(img.shape) * 255).astype(np.uint8)

# --- Step 6: Display Before and After ---
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Color Calibrated")
plt.imshow(img_corrected)

plt.tight_layout()
plt.show()