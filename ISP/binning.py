import numpy as np

def bin_bayer_2x2(bayer: np.ndarray) -> np.ndarray:
    """2x2 bin a Bayer raw image, preserving the Bayer pattern.
    Input:  (H, W) uint16 — must be multiples of 4 (e.g. 2160x3840 BGGR)
    Output: (H//2, W//2) uint16 — same Bayer pattern (e.g. 1080x1920 BGGR)
    """
    H, W = bayer.shape
    # Integer sum then >> 2 (divide by 4) — avoids float64 conversion from .mean()
    out = np.empty((H // 2, W // 2), dtype=np.uint16)
    out[0::2, 0::2] = (bayer[0::4, 0::4].astype(np.uint32) + bayer[0::4, 2::4] + bayer[2::4, 0::4] + bayer[2::4, 2::4]) >> 2
    out[0::2, 1::2] = (bayer[0::4, 1::4].astype(np.uint32) + bayer[0::4, 3::4] + bayer[2::4, 1::4] + bayer[2::4, 3::4]) >> 2
    out[1::2, 0::2] = (bayer[1::4, 0::4].astype(np.uint32) + bayer[1::4, 2::4] + bayer[3::4, 0::4] + bayer[3::4, 2::4]) >> 2
    out[1::2, 1::2] = (bayer[1::4, 1::4].astype(np.uint32) + bayer[1::4, 3::4] + bayer[3::4, 1::4] + bayer[3::4, 3::4]) >> 2
    return out