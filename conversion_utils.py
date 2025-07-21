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