import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(raw={'format': 'SBGGR12', 'size': (2028, 1520)})
picam2.configure(config)
picam2.start()

# Wait for camera to warm up
time.sleep(2)

# Fixed exposure (you can adjust if needed)
exposure_time = 1000  # in microseconds

# Gain sweep values (in analog gain units)
gain_values = np.linspace(1.0, 20.0, 100)  # adjust upper limit as needed

# Prompt user to cover the lens
input("Please cover the camera lens completely, then press ENTER to begin...")

black_offsets = []

for gain in gain_values:
    picam2.set_controls({
        "ExposureTime": exposure_time,
        "AnalogueGain": gain
    })
    metadata = picam2.capture_metadata()
    time.sleep(0.5)  # wait for settings to apply

    # Capture dark frame
    raw = picam2.capture_array("raw")

    # Option 1: whole-frame average (assuming lens is covered)
    black_level = np.mean(raw)

    # Option 2 (if you want optically black pixels only):
    # black_level = np.mean(raw[:, :32])  # leftmost 32 columns

    black_offsets.append(black_level)
    print(f"Gain: {gain:.1f}, Black Level: {black_level:.2f}, Digital Gain: {metadata['DigitalGain']:.2f}")

picam2.stop()

# Plotting
plt.figure()
plt.style.use('style.mplstyle')
plt.plot(gain_values, black_offsets, 'o-', label='Measured black level')
plt.xlabel("Analog Gain")
plt.ylabel("Mean Black Offset")
plt.title("Black Offset vs. Analog Gain")
plt.grid(True)
plt.legend()
plt.show()