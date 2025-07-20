import numpy as np

def correct_stride_padding(raw, img_width_px):
    """
    Remove padding or stride artifacts from a raw Bayer image.

    In Pi5 HQ camera, every other column contains padding (e.g., from stride),
    and have extra columns beyond the target width.

    Args:
        raw (np.ndarray): Raw image array with potential padding.
        img_width_px (int): Target width after stride correction.

    Returns:
        np.ndarray: Corrected raw image with shape (H, img_width_px)
    """
    # Remove interleaved padding (e.g., every other column)
    raw = raw[:, 1::2]

    # Trim to exact width if needed
    if raw.shape[1] > img_width_px:
        raw = raw[:, :img_width_px]

    return raw