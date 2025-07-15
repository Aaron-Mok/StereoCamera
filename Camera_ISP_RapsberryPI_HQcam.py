from picamera2 import Picamera2
import cv2
import numpy as np

# Initialize camera
picam2 = Picamera2(0)
config = picam2.create_preview_configuration(raw={'format': 'SBGGR12', 'size': (2028, 1520)})
picam2.configure(config)
print("Camera configuration:")
print(picam2.camera_configuration())
img_width_px, image_height_px = picam2.camera_configuration()['sensor']['output_size']
picam2.start()


while True:
    raw = picam2.capture_array("raw")
    # controls = {
    # "AnalogueGain": 1.0,
    # "ExposureTime": 60000  # in microseconds
    # }
    # picam2.set_controls(controls)

    metadata = picam2.capture_metadata()
    print("Analogue Gain:", metadata["AnalogueGain"])
    print("Digital Gain:", metadata["DigitalGain"])
    print("Exposure Time (µs):", metadata["ExposureTime"])

    cv2.imshow("Raw Image", raw)

    # Stride/padding correction
    raw = raw[:, 1::2]
    raw = raw[:,:img_width_px-1]
    raw_8bit = raw.astype(np.uint8)
    cv2.imshow("Raw Image", raw_8bit)

    # TODO: Add black offset subtraction. Calibration script written. Need Curve fitting and get the black offset for each analogue gain value.

    

    # Demosaic
    rgb_image = cv2.cvtColor(raw_8bit, cv2.COLOR_BAYER_BGGR2BGR) # This is BIlinear. There are three options: Bilinear, Edge Aware, and Variable Number of Gradients

    # Gamma correction
    gamma = 2.2
    norm = rgb_image.astype(np.float32) / 255.0
    corrected = np.power(norm, 1.0 / gamma)
    gamma_corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)

    # Display
    cv2.imshow("Gamma Corrected", rgb_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
