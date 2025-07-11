from picamera2 import Picamera2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Start camera with raw output (10-bit Bayer)
picam2 = Picamera2()
config = picam2.create_preview_configuration(raw={"format": "SBGGR10"})
picam2.configure(config)
picam2.start()

def show_histogram(image, title, pos):
    """Display histogram using OpenCV drawing"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)

    for x in range(256):
        cv2.line(hist_img, (x, 200), (x, 200 - int(hist[x])), (255, 255, 255))
    cv2.imshow(title, hist_img)

# Finding the black offset from optical black pixels. Using optically black regions on the sensor.
raw = picam2.capture_array("raw").astype(np.float32)
black_rows = np.concatenate((raw[:12, :], raw[-12:, :]), axis=0)
black_cols = np.concatenate((raw[:, :8], raw[:, -8:]), axis=1)
black_mask = np.concatenate((black_rows.flatten(), black_cols.flatten()))
black_level = np.mean(black_mask)

while True:
    # Get raw Bayer image (10-bit)
    raw = picam2.capture_array("raw")  # shape is HxW, single channel
    raw = raw.astype(np.float32)

    # Convert 10-bit to 8-bit for display
    raw_8bit = np.clip(raw / 4, 0, 255).astype(np.uint8)

    corrected = np.clip(raw - black_level, 0, 1023)
    corrected_8bit = (corrected / 4).astype(np.uint8)

    # Demosaic for preview
    bgr_before = cv2.cvtColor(raw_8bit, cv2.COLOR_BAYER_BG2BGR)
    bgr_after = cv2.cvtColor(corrected_8bit, cv2.COLOR_BAYER_BG2BGR)

    # Show live images side by side
    combined = np.hstack((bgr_before, bgr_after))
    cv2.imshow("test", bgr_after)
    # cv2.imshow("Before (Left) vs After Black Level Subtraction (Right)", combined)

    # Show histogram for one channel (e.g., grayscale view)
    gray_before = cv2.cvtColor(bgr_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(bgr_after, cv2.COLOR_BGR2GRAY)
    show_histogram(gray_before, "Histogram: Before", 0)
    show_histogram(gray_after, "Histogram: After", 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()