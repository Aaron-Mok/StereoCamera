import numpy as np

def gray_world_awb(img):
    # img: float RGB image in [0, 1]
    r_avg = img[:, :, 0].mean()
    g_avg = img[:, :, 1].mean()
    b_avg = img[:, :, 2].mean()

    # avg_gray = (r_avg + g_avg + b_avg) / 3.0
    # r_gain = avg_gray / r_avg
    # g_gain = avg_gray / g_avg
    # b_gain = avg_gray / b_avg

    # Normalize green gain to 1
    g_gain = 1.0
    r_gain = g_avg / r_avg
    b_gain = g_avg / b_avg

    img_awb = img.copy()
    img_awb[:, :, 0] *= r_gain
    img_awb[:, :, 1] *= g_gain
    img_awb[:, :, 2] *= b_gain

    img_awb = np.clip(img_awb / img_awb.max(), 0, 1)

    print(f"Gray World AWB gains: R={r_gain:.4f}, G={g_gain:.4f}, B={b_gain:.4f}")
    return img_awb

def from_calibration_wb(img):
    r_gain = 1.77
    g_gain = 1.000
    b_gain = 1.42

    img_awb = img.copy()
    img_awb[:, :, 0] *= r_gain
    img_awb[:, :, 1] *= g_gain
    img_awb[:, :, 2] *= b_gain

    img_awb = np.clip(img_awb / img_awb.max(), 0, 1)

    print(f"Calibration AWB gains: R={r_gain:.4f}, G={g_gain:.4f}, B={b_gain:.4f}")
    return img_awb

def white_patch_awb(img):
    # Assumes brightest R, G, B pixels are the illuminant
    r_max = img[:, :, 0].max()
    g_max = img[:, :, 1].max()
    b_max = img[:, :, 2].max()

    max_rgb = (r_max + g_max + b_max) / 3.0
    r_gain = max_rgb / r_max
    g_gain = max_rgb / g_max
    b_gain = max_rgb / b_max

    img_awb = img.copy()
    img_awb[:, :, 0] *= r_gain
    img_awb[:, :, 1] *= g_gain
    img_awb[:, :, 2] *= b_gain

    img_awb = np.clip(img_awb / img_awb.max(), 0, 1)
    print(f"White Patch AWB gains: R={r_gain:.4f}, G={g_gain:.4f}, B={b_gain:.4f}")
    return img_awb

def shades_of_gray_awb(img, p=6):
    R = np.power(img[:, :, 0], p).mean() ** (1 / p)
    G = np.power(img[:, :, 1], p).mean() ** (1 / p)
    B = np.power(img[:, :, 2], p).mean() ** (1 / p)
    mean_power = (R + G + B) / 3.0

    img_awb = img.copy()
    img_awb[:, :, 0] *= mean_power / R
    img_awb[:, :, 1] *= mean_power / G
    img_awb[:, :, 2] *= mean_power / B

    img_awb = np.clip(img_awb / img_awb.max(), 0, 1)
    return img_awb