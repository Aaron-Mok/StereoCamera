import cv2
import numpy as np
from picamera2 import Picamera2

selected_roi = None
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, selected_roi, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = frame.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 255, 255), 1)
            cv2.imshow("Select Gray Patch", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        selected_roi = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        x0, y0, w, h = selected_roi
        roi = frame[y0:y0+h, x0:x0+w].astype(np.float32) / 255.0

        # Compute average RGB
        print(roi)
        avg_color = roi.mean(axis=(0, 1))  # [B, G, R]
        print(avg_color)
        B_avg, G_avg, R_avg = avg_color
        r_gain = G_avg / R_avg
        g_gain = 1.0
        b_gain = G_avg / B_avg

        print(f"\n📏 Selected ROI: x={x0}, y={y0}, w={w}, h={h}")
        print(f"🎯 Avg RGB: R={R_avg:.4f}, G={G_avg:.4f}, B={B_avg:.4f}")
        print(f"✅ White balance gains:")
        print(f"  R gain: {r_gain:.4f}")
        print(f"  G gain: {g_gain:.4f}")
        print(f"  B gain: {b_gain:.4f}")

        cv2.rectangle(frame, (ix, iy), (x, y), (0, 0, 0), 2)
        cv2.imshow("Select Gray Patch", frame)

# === Replace with your camera input ===
# Initialize camera
picam2 = Picamera2(0)
config = picam2.create_preview_configuration(raw={'format': 'SBGGR12', 'size': (2028, 1520)})
picam2.configure(config)
print("Camera configuration:")
print(picam2.camera_configuration())
img_width_px, image_height_px = picam2.camera_configuration()['sensor']['output_size']
picam2.start()

raw = picam2.capture_array("raw")
metadata = picam2.capture_metadata()
print("Analogue Gain:", metadata["AnalogueGain"])
print("Digital Gain:", metadata["DigitalGain"])
print("Exposure Time (µs):", metadata["ExposureTime"])


cv2.namedWindow("Select Gray Patch")
cv2.setMouseCallback("Select Gray Patch", draw_rectangle)

print("📷 Live view started. Click and drag to select the 50% gray patch.")
print("🔴 Press ESC to quit.")

while True:
    # Capture new frame every loop
    # Stride/padding correction
    raw = picam2.capture_array("raw")
    raw = raw[:, 1::2]
    raw_float = raw[:,:img_width_px-1]
    raw_8bit = raw_float.astype(np.uint8)

    # lens shading correction
    flat_field_img = np.load("./flat_field_output/flat_field_average.npy")
    flat_field_img_blurred = cv2.GaussianBlur(flat_field_img, (11, 11), 0)
    flat_field_img_normalize = flat_field_img_blurred/flat_field_img_blurred.max()
    lens_shading_correction = flat_field_img_normalize

    lens_shade_corrected_float = (raw_float * lens_shading_correction)
    lens_shade_corrected_8bit = (lens_shade_corrected_float).astype(np.uint8)

    # Demosaic
    frame = cv2.cvtColor(lens_shade_corrected_8bit, cv2.COLOR_BAYER_BGGR2BGR) # This is BIlinear. There are three options: Bilinear, Edge Aware, and Variable Number of Gradients

    cv2.imshow("Select Gray Patch", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
picam2.stop()