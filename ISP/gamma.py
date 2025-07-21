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

def apply_sRGB_gamma_encoding(image):
    a = 0.055
    threshold = 0.0031308
    srgb = np.where(
        image <= threshold,
        12.92 * image,
        (1 + a) * np.power(image, 1/2.4) - a
    )
    return np.clip(srgb, 0, 1)