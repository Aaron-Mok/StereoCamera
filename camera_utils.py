import cv2
import numpy as np
import subprocess

# def initialize_camera(camera_id, image_size):
#     width, height = image_size 
    
#     # Use 'pixelformat=RG10' to match your v4l2-ctl output exactly
#     pipeline = (
#         f"v4l2src device=/dev/video{camera_id} ! "
#         f"video/x-bayer, width={width}, height={height}, pixelformat=RG10 ! "
#         f"appsink drop=True"
#     )
    
#     cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#     return cap, (width, height)

def initialize_camera(camera_id, image_size):
    """
    On Jetson, we use a GStreamer pipeline string.
    Note: 'raw' in Jetson ISP terms usually means uncompressed YUV.
    To get Bayer RGGB, you must bypass the ISP.
    """
    width, height = image_size
    # Pipeline to get raw-ish data (YUV is often preferred for performance)
    # If you need TRUE Bayer, you'd use v4l2-ctl to capture to disk.
    # pipeline = (
    #     f"nvarguscamerasrc sensor-id={camera_id} ! "
    #     f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12 ! "
    #     f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    # )

    pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, format=RG10, width=1920, height=1080, framerate=30/1 ! "
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Could not initialize Jetson Camera")
    
    return cap, (width, height)

def set_camera_controls(camera_id, gain=None, exposure=None):
    """
    Jetson controls are handled via the V4L2 driver directly.
    """
    # Device path is usually /dev/video0 or /dev/video1
    device = f"/dev/video{camera_id}"
    
    if exposure is not None:
        # Turn off auto exposure first (1 is manual, 0 is auto in some drivers)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'exposure_auto=1'])
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'exposure_absolute={exposure}'])
        
    if gain is not None:
        subprocess.run(['v4l2-ctl', '-d', device, '-c', f'gain={gain}'])

def capture_raw_image(cap):
    """
    Captures a frame and returns it as a numpy array.
    """
    ret, frame = cap.read()
    if not ret:
        return None
    
    # Metadata is harder to get in real-time via OpenCV 
    # compared to Picamera2. You'd typically read v4l2-ctl again.
    return frame

# def set_camera_controls(camera_id, gain=None, exposure=None):
#     device = f"/dev/video{camera_id}"
    
#     # 1. Force manual mode to stop the Jetson from auto-adjusting
#     # Note: 'gain_auto' and 'exposure_auto' names vary by driver version
#     subprocess.run(['v4l2-ctl', '-d', device, '-c', 'exposure_auto=1']) 
    
#     if exposure is not None:
#         # For IMX477, exposure is often 'exposure' or 'exposure_absolute'
#         # Units are usually line-counts or microseconds
#         subprocess.run(['v4l2-ctl', '-d', device, '-c', f'exposure={exposure}'])
        
#     if gain is not None:
#         # Gain is often 'gain' or 'analogue_gain'
#         subprocess.run(['v4l2-ctl', '-d', device, '-c', f'gain={gain}'])

# def capture_raw_image(cap):
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         return None
    
#     # Optional: Debug to see how many bytes the hardware is actually sending
#     # For 1920x1080 RG10, this should be 2,592,000 bytes (1.25 bytes per pixel)
#     print(f"Captured buffer size: {frame.nbytes} bytes")
#     return frame.flatten()

def initialize_camera_with_ISP(camera_id, image_size):
    """
    Initialize a Raspberry Pi camera using Picamera2 with ISP processing.

    Args:
        camera_id (int): ID of the camera to use (0 or 1 for CSI port).
        image_size (tuple): Desired image resolution as (width, height), e.g., (2028, 1520).

    Returns:
        Picamera2: Initialized camera object.
        tuple: Actual image size as (width, height).
    """
    picam2 = Picamera2(camera_id)
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": image_size})
    picam2.configure(config)
    print("Camera configuration:")
    print(picam2.camera_configuration())
    img_width_px, image_height_px = picam2.camera_configuration()['sensor']['output_size']
    output_im_size = (img_width_px, image_height_px)
    picam2.start()
    return picam2, output_im_size

def initialize_camera_jetson(device, W=3840, H=2160, exposure=33000, gain=200, frame_rate=30000000):
    """
    Initialize a Jetson camera via v4l2-ctl and return a streaming subprocess.
    Returns the subprocess and frame_bytes size.
    """
    frame_bytes = W * H * 2  # 16-bit-per-pixel, 2 bytes

    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=override_enable=1"])
    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=group_hold=1"])
    subprocess.run([
        "v4l2-ctl", "-d", device,
        f"--set-ctrl=exposure={exposure}",
        f"--set-ctrl=gain={gain}",
        f"--set-ctrl=frame_rate={frame_rate}"
    ])

    p = subprocess.Popen([
        "v4l2-ctl", "-d", device,
        "--stream-mmap", "--stream-count=0", "--stream-to=-"
    ], stdout=subprocess.PIPE, bufsize=frame_bytes)

    return p, frame_bytes


def capture_raw_frame_jetson(p, W=3840, H=2160):
    """
    Read one raw frame from the v4l2 stream subprocess.
    Returns np.ndarray of shape (H, W) dtype uint16, or None on EOF.
    """
    frame_bytes = W * H * 2  # Ensure this matches the expected frame size
    buf = p.stdout.read(frame_bytes)
    if len(buf) != frame_bytes:
        return None
    return np.frombuffer(buf, dtype=np.uint16).reshape(H, W)
