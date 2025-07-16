import cv2
import numpy as np

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
        avg_color = roi.mean(axis=(0, 1))  # [B, G, R]
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
cap = cv2.VideoCapture(0)  # Change to 0 or another device ID

cv2.namedWindow("Select Gray Patch")
cv2.setMouseCallback("Select Gray Patch", draw_rectangle)

print("📷 Live view started. Click and drag to select the 50% gray patch.")
print("🔴 Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Select Gray Patch", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()