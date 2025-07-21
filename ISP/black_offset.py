import numpy as np

def apply_black_offset(image, offset):
    """
    Subtracts a constant black offset from the image. 
    Offset can be measured from the black_offset_measurement.py script.

    Args:
        image (np.ndarray): Input raw image (can be float or integer type), 8bit image.
        offset (float): The constant black level to subtract.

    Returns:
        np.ndarray (float): Black-offset corrected image (clipped to [0, inf]).
    """
    # Convert to float for safe subtraction if needed
    image_float = image.astype(np.float32)
    
    # Subtract black offset and clip
    corrected = np.clip(image_float - offset, 0, None)

    return corrected