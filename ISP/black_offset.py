import numpy as np

def apply_black_offset(image, offset):
    """
    Subtracts a constant black offset from the image. 
    Offset can be measured from the black_offset_measurement.py script.

    Args:
        image (np.ndarray): Input raw image (can be float or integer type), 8bit or 16 bit image.
        offset (int): The constant black level to subtract.

    Returns:
        np.ndarray (int): Black-offset corrected image (clipped to [0, inf]).
    """
    # Subtract black offset and clip
    corrected = image.astype(np.float32) - offset   #COnvert to float to prevent overflow

    if image.dtype == np.uint8:
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        corrected = np.clip(corrected, 0, 65535).astype(np.uint16)

    return corrected