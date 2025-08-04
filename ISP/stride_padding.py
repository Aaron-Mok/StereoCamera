import numpy as np

def unpack_and_trim_raw(raw, img_width_px, bit_depth=16):
    """
    Remove padding or stride artifacts from a raw Bayer image.

    For 16-bit raw images (from Pi HQ camera), the data is packed into two interleaved 
    8-bit columns: the lower 8 bits (LSB, Least Significant Bit) and higher 8 bits (MSB, Most Significant Bit). This function 
    reconstructs the 16-bit image and removes stride artifacts.

    For 8-bit raw images, only stride correction and trimming are applied.

    Args:
        raw (np.ndarray): Raw image array. For 16-bit, shape (H, 2*W) with LSB and MSB interleaved.
        img_width_px (int): Target width (in pixels) after correction.
        bit_depth (int): Bit depth of raw image. Must be 8 or 16.

    Returns:
        np.ndarray: Corrected raw image with shape (H, img_width_px) and dtype uint8 or uint16.
    """
    if bit_depth == 16:
        # Extract LSB and MSB from interleaved columns and reconstruct 16-bit raw
        lsb = raw[:, ::2].astype(np.uint16)
        msb = raw[:, 1::2].astype(np.uint16)
        raw_corrected = (msb << 8) | lsb

    elif bit_depth == 8:
        # For 8-bit data, simply remove padding (if any) and use as-is
        lsb = raw[:, ::2].astype(np.uint16)
        msb = raw[:, 1::2].astype(np.uint16)
        raw_corrected = (msb << 8) | lsb
        # Convert to 8-bit, this is same as (raw_corrected/256).astype('uint8'), but faster. 65535/256 = 255.99, int (255.99) = 255, 65535 is 2^16 - 1
        raw_corrected = (raw_corrected >> 8).astype('uint8') 

    else:
        raise ValueError("bit_depth must be either 8 or 16")

    # Trim to the exact target width
    raw_corrected = raw_corrected[:, :img_width_px]

    return raw_corrected