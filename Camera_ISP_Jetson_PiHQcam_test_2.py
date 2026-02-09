import os
import numpy as np
import cv2
from ISP import *
from camera_utils import *
from conversion_utils import *

# Parameters for IMX477
WIDTH = 1920
HEIGHT = 1080
# Jetson VI often pads raw lines to 256-byte boundaries (stride alignment)
# You may need to calculate stride if the image looks skewed.

def read_raw_image(filepath):
    # Load data as 16-bit integers (Jetson packs 10/12-bit RAW into 16-bit)
    raw_data = np.fromfile(filepath, dtype=np.uint16)
    
    # Reshape (Handle stride padding if necessary; usually minimal on direct dumps)
    try:
        raw_image = raw_data.reshape((HEIGHT*2, WIDTH*2))
        # raw_u16 =  unpack_and_trim_raw(raw_data, img_width_px=[1920, 1080], bit_depth=16)
    except ValueError:
        print(f"Shape mismatch. Total elements: {raw_data.size}. Expected: {WIDTH*HEIGHT}")
        return None

    return raw_image

# 1. Capture via subprocess (simplest robust method)
os.system("v4l2-ctl -d /dev/video0   --set-fmt-video=width=1920,height=1080,pixelformat=RG10   --set-ctrl bypass_mode=0   --stream-mmap --stream-count=1 --stream-to=raw_frame2.bin")

# 2. Process
bayer_img = read_raw_image("temp_raw.bin")

if bayer_img is not None:
    # Normalize for visualization (shift right if MSB aligned, or scale)
    # Note: NVIDIA TRM states RAW10/12 is often stored as T_R16
    # display_img = (bayer_img >> 6).astype(np.uint8) # Simple downsample for viewing
    
    cv2.imshow("Bayer Pattern (Grayscale)", bayer_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()