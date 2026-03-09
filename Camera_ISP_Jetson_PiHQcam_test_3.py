import subprocess, numpy as np, cv2
from ISP import *
from camera_utils import *
from conversion_utils import *

W, H = 1920, 1080
frame_bytes = W * H * 8   # RG10 often delivered as 16-bit-per-pixel container on Jetson
SAVE_COUNT = 10
frame_count = 0

p = subprocess.Popen([
    "v4l2-ctl", "-d", "/dev/video0",
    "--set-fmt-video=width=1920,height=1080,pixelformat=RG10",
    # "--set-parm=30",
    "--stream-mmap", "--stream-count=0", "--stream-to=-"
], stdout=subprocess.PIPE, bufsize=frame_bytes*1)

use_shift = 2

while True:
    buf = p.stdout.read(frame_bytes)
    if len(buf) != frame_bytes:
        break

    raw_u16 = np.frombuffer(buf, dtype=np.uint16).reshape(H *2, W * 2)

  # Step 3: Reconstruct 10-bit pixels from interleaved LSB/MSB
    # lsb = raw_u16[:, ::2].astype(np.uint16)    # even cols → LSB
    # msb = raw_u16[:, 1::2].astype(np.uint16)   # odd cols  → MSB
    # raw_corrected = (msb << 8) | lsb            # shape: (H, W), values in [0, 1023]

    # display = (raw_corrected >> use_shift).astype(np.uint8)


    if frame_count < SAVE_COUNT:
        # Save as 16-bit PNG to preserve 10-bit data
        cv2.imwrite(f"frame_{frame_count:04d}.png", raw_u16)
        print(f"Saved frame {frame_count+1}/{SAVE_COUNT}")
        frame_count += 1
        if frame_count == SAVE_COUNT:
            print("All 10 frames saved.")


    cv2.imshow("raw10", raw_u16)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:       # ESC to quit
        break
    elif k == ord('2'):
        use_shift = 2
        print("shift = 2")
    elif k == ord('6'):
        use_shift = 6
        print("shift = 6")

p.terminate()
cv2.destroyAllWindows()
