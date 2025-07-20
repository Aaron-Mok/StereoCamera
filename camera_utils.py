from picamera2 import Picamera2


def initialize_camera(camera_id, image_size):
    """
    Initialize a Raspberry Pi camera using Picamera2.

    Args:
        camera_id (int): ID of the camera to use (0 or 1 for CSI port).
        image_size (tuple): Desired image resolution as (width, height), e.g., (2028, 1520).

    Returns:
        Picamera2: Initialized camera object.
        tuple: Actual image size as (width, height).
    """
    picam2 = Picamera2(camera_id)
    config = picam2.create_preview_configuration(raw={'format': 'SBGGR12', 'size': image_size})
    picam2.configure(config)
    print("Camera configuration:")
    print(picam2.camera_configuration())
    img_width_px, image_height_px = picam2.camera_configuration()['sensor']['output_size']
    picam2.start()
    output_im_size = (img_width_px, image_height_px)
    return picam2, output_im_size


def set_camera_controls(picam2, gain=None, exposure=None):
    """
    Set camera analog gain and exposure time if specified.

    Args:
        picam2 (Picamera2): The initialized camera object.
        gain (float, optional): Analog gain to apply. If None, it won't be set.
        exposure (int, optional): Exposure time in microseconds. If None, it won't be set.

    Returns:
        None
    """
    controls = {}

    if gain is not None:
        controls["AnalogueGain"] = gain

    if exposure is not None:
        controls["ExposureTime"] = exposure

    if controls:
        picam2.set_controls(controls)

def capture_raw_image(picam2):
    """
    Capture a raw image from the Picamera2 and print exposure metadata.

    Args:
        picam2 (Picamera2): The initialized camera object.

    Returns:
        Raw Bayer image as a array.
    """
    raw = picam2.capture_array("raw")
    metadata = picam2.capture_metadata()
    print("Analogue Gain:", metadata["AnalogueGain"])
    print("Digital Gain:", metadata["DigitalGain"])
    print("Exposure Time (µs):", metadata["ExposureTime"])
    return raw