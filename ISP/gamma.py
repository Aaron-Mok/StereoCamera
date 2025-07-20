import numpy as np

def apply_gamma_correction(image, gamma=2.2):
    """
    Apply gamma correction to an image.

    Args:
        image (np.ndarray): Input image in [0, 1] float format.
        gamma (float): Gamma value to apply. Common values are 2.2 or 1/2.2.

    Returns:
        np.ndarray: Gamma-corrected image in [0, 1] float format.
    """
    # Avoid modifying original image
    corrected = np.power(np.clip(image, 0, 1), 1.0 / gamma)
    return corrected