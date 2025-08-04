import numpy as np
import colour

def normalize01_to_8bit(img_normalized):
    """
    Convert an image with values in [0, 1] (normalized float32/float64)
    to 8-bit (uint8) format.
    """
    img_8bit = np.clip(img_normalized * 255.0, 0, 255).astype(np.uint8)
    return img_8bit

def bit8_to_normalize01(img_8bit):
    """
    Convert an 8-bit (uint8) image to float32 format with values normalized to [0, 1].
    """
    img_normalized = img_8bit.astype(np.float32) / 255.0
    return img_normalized

def bit16_to_normalize01(img_8bit):
    """
    Convert an 8-bit (uint8) image to float32 format with values normalized to [0, 1].
    """
    img_normalized = img_8bit.astype(np.float32) / 65535.0
    return img_normalized

def normalize01_to_16bit(img_normalized):
    """
    Convert an image with values in [0, 1] (normalized float32/float64)
    to 8-bit (uint8) format.
    """
    img_16bit = np.clip(img_normalized * 65535.0, 0, 65535).astype(np.uint16)
    return img_16bit


def srgb_d65_to_srgb_d50(rgb_srgb_d65):
    """
    Convert sRGB values from D65 illuminant to D50 illuminant.
    """
    # Convert sRGB (D65) to XYZ
    xyz_d65 = colour.sRGB_to_XYZ(rgb_srgb_d65)

    # Adapt D65 to D50 using Bradford method
    xyz_d50 = colour.adaptation.chromatic_adaptation(
        xyz_d65,
        colour.CCS_ILLUMINANTS['CIE 1931']['D65'],
        colour.CCS_ILLUMINANTS['CIE 1931']['D50'],
        method='Bradford'
    )

    # Convert XYZ (D50) back to sRGB
    rgb_srgb_d50 = colour.XYZ_to_sRGB(xyz_d50)
    
    return rgb_srgb_d50

def srgb_to_linear(srgb_f01):
    """Convert sRGB (0-1 range) to linear RGB using IEC 61966-2-1"""
    srgb_f01 = np.clip(srgb_f01, 0, 1)
    linear_rgb = np.where(srgb_f01 <= 0.04045,
                      srgb_f01 / 12.92,
                      ((srgb_f01 + 0.055) / 1.055) ** 2.4)
    return np.clip(linear_rgb, 0, 1)


def linear_to_srgb(linear_rgb):
    """
    Convert linear RGB [0,1] to sRGB [0,1] using standard gamma encoding.
    Accepts a NumPy array or image of shape (..., 3).
    """
    linear_rgb = np.clip(linear_rgb, 0, 1)  # Ensure valid input

    threshold = 0.0031308
    a = 0.055

    srgb = np.where(
        linear_rgb <= threshold,
        linear_rgb * 12.92,
        (1 + a) * np.power(linear_rgb, 1 / 2.4) - a
    )
    return np.clip(srgb, 0, 1)