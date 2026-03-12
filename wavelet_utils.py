"""Shared helpers for stationary wavelet transform (SWT) operations.

The SWT requires input dimensions to be multiples of 2^levels.
These functions handle the reflect-padding and unpadding needed
to apply SWT to arbitrary-sized images.
"""

import numpy as np
import pywt


def swt_pad(image: np.ndarray, levels: int) -> tuple[np.ndarray, int, int]:
    """Reflect-pad an image so its dimensions are multiples of 2^levels.

    Returns (padded_image, pad_h, pad_w) where pad_h/pad_w are the
    top/left padding amounts needed to unpad the result.
    """
    pad_size = max(32, 2 ** levels)
    factor = 2 ** levels
    h, w = image.shape
    ph = int(np.ceil((h + 2 * pad_size) / factor) * factor)
    pw = int(np.ceil((w + 2 * pad_size) / factor) * factor)
    pad_h = (ph - h) // 2
    pad_w = (pw - w) // 2
    padded = np.pad(
        image,
        ((pad_h, ph - h - pad_h), (pad_w, pw - w - pad_w)),
        mode="reflect",
    )
    return padded, pad_h, pad_w


def swt_unpad(result: np.ndarray, h: int, w: int, pad_h: int, pad_w: int) -> np.ndarray:
    """Remove SWT padding and clip to non-negative values."""
    return np.clip(result[pad_h:pad_h + h, pad_w:pad_w + w], 0, None)
