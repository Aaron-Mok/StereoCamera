from picamera2 import Picamera2
import numpy as np
import cv2
import os
import time

# === CONFIG ===
num_images = 30
output_dir = "flat_field_output"
os.makedirs(output_dir, exist_ok=True)

# === Initialize camera ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Preview running... align setup.")
print("Press Enter to start capturing flat-field images.")
while True:
    frame = picam2.capture_array()
    metadata = picam2.capture_metadata()
    print("Analogue Gain:", metadata["AnalogueGain"])
    print("Digital Gain:", metadata["DigitalGain"])
    print("Exposure Time (µs):", metadata["ExposureTime"])
    cv2.imshow("Pi Camera ISP Output", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        break
        print("Enter pressed.")
cv2.destroyAllWindows()

# Reconfigure to high-resolution raw
picam2.stop()
config = picam2.create_still_configuration(raw={"format": "SBGGR12", "size": (2028, 1520)})
picam2.configure(config)
img_width_px, image_height_px = picam2.camera_configuration()['sensor']['output_size']
picam2.set_controls({
    "ExposureTime": 200000,
    "AnalogueGain": 4.0,
    "AeEnable": False,
    "AwbEnable": False
})
picam2.start()
time.sleep(1)

# === Capture and average ===
accumulator = None

for i in range(num_images):
    raw = picam2.capture_array("raw").astype(np.float32)
    metadata = picam2.capture_metadata()
    print("Analogue Gain:", metadata["AnalogueGain"])
    print("Digital Gain:", metadata["DigitalGain"])
    print("Exposure Time (µs):", metadata["ExposureTime"])

    # Stride/padding correction
    raw = raw[:, 1::2]
    raw = raw[:,:img_width_px-1]

    if accumulator is None:
        accumulator = raw
    else:
        accumulator += raw

    print(f"Captured image {i+1}/{num_images}")
    time.sleep(0.1)

picam2.stop()

# === Compute average and vignetting map ===
avg_image = accumulator / num_images

# === Save output ===
np.save(os.path.join(output_dir, "flat_field_average.npy"), avg_image)

# Convert to 8-bit for visualization
avg_8bit = avg_image.astype(np.uint8)  # 10-bit to 8-bit

# === Show images ===
cv2.imshow("Average Flat Field (8-bit)", avg_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done. Files saved in:", output_dir)
