"""
No-reference image sharpness metrics for planetary imaging.

All metrics operate on the planet disk region (bright pixels) to avoid
background sky from skewing results.  Higher values = sharper / cleaner.

Metrics
-------
laplacian_variance   Variance of the Laplacian — classic focus measure.
tenengrad            Mean squared Sobel gradient — robust sharpness.
normalised_power_hf  Fraction of spectral power in high frequencies.
brenner              Horizontal gradient energy — fast, robust.
quality_ratio        tenengrad / sqrt(laplacian_variance) — rewards clean
                     sharpening without ringing.
smoothness           Structure-to-noise ratio via multiscale gradient
                     comparison — close to 1.0 means clean result.
"""

import numpy as np
from scipy.ndimage import laplace, sobel, gaussian_filter


# ---------------------------------------------------------------------------
# Planet mask
# ---------------------------------------------------------------------------

def planet_mask(image: np.ndarray) -> np.ndarray:
    """
    Boolean mask selecting the planetary disk (and rings if present).

    Uses adaptive thresholding based on sky statistics rather than a fixed
    percentile.  This handles both large planets (Jupiter filling the frame)
    and small ones (Saturn at 10% of the frame).

    Strategy:
    1. Estimate sky level from the 25th percentile (always background).
    2. Estimate sky noise from pixels at or below that level.
    3. Threshold at sky + 5 sigma — captures the planet regardless of size.
    """
    img = image.astype(np.float64)

    # Sky estimation: 25th percentile is always background
    # (even Jupiter rarely fills >50% of the frame)
    sky_level = np.percentile(img, 25)

    # Sky noise: standard deviation of pixels at or below sky level
    sky_pixels = img[img <= sky_level]
    sky_sigma = float(np.std(sky_pixels)) if len(sky_pixels) > 100 else 1.0

    # Planet threshold: well above sky noise
    threshold = sky_level + max(5.0 * sky_sigma, 1.0)
    mask = img > threshold

    # Sanity: mask should capture at least 0.1% of the image
    if mask.sum() < 0.001 * mask.size:
        mask = img > np.percentile(img, 95)

    return mask


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def laplacian_variance(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian within the planet disk."""
    img = image.astype(np.float64)
    if mask is None:
        mask = planet_mask(img)
    lap = laplace(img)
    return float(np.var(lap[mask]))


def tenengrad(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared Sobel gradient magnitude on the planet disk."""
    img = image.astype(np.float64)
    if mask is None:
        mask = planet_mask(img)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return float(np.mean((gx**2 + gy**2)[mask]))


def normalised_power_hf(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of spectral power in the outer half of the frequency domain."""
    img = image.astype(np.float64)
    if mask is None:
        mask = planet_mask(img)

    # Crop to planet bounding box to exclude background
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    crop = img[r0:r1 + 1, c0:c1 + 1]

    # Power spectrum
    f = np.fft.fftshift(np.abs(np.fft.fft2(crop)) ** 2)
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    r_max = min(cy, cx)

    total = f.sum() + 1e-30
    hf = f[r > r_max / 2].sum()
    return float(hf / total)


def brenner(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared horizontal difference (2-pixel gap) on the planet disk."""
    img = image.astype(np.float64)
    if mask is None:
        mask = planet_mask(img)
    diff = img[:, 2:] - img[:, :-2]
    mask_trim = mask[:, 1:-1]  # align with diff width
    return float(np.mean(diff[mask_trim] ** 2))


def quality_ratio(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    tenengrad / sqrt(laplacian_variance).

    Rewards sharp images with low ringing — the key balance metric.
    PixInsight-quality results score ~70k–85k on planetary images.
    """
    if mask is None:
        mask = planet_mask(image)
    return tenengrad(image, mask) / (np.sqrt(laplacian_variance(image, mask)) + 1e-10)


def smoothness(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Structure-to-noise ratio via multiscale gradient comparison.

    Compares gradient energy at sigma=1.5 (real edges survive) vs raw
    gradient (noise-dominated).  Ratio close to 1.0 = clean result.
    """
    img = image.astype(np.float64)
    if mask is None:
        mask = planet_mask(img)

    def _grad_energy(arr: np.ndarray) -> float:
        gx = sobel(arr, axis=1)
        gy = sobel(arr, axis=0)
        return float(np.mean((gx**2 + gy**2)[mask])) + 1e-30

    raw = _grad_energy(img)
    smooth = _grad_energy(gaussian_filter(img, sigma=1.5))
    return float(smooth / raw)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

# Weights tuned for planetary imaging:
# - tenengrad + brenner reward real edges
# - smoothness penalises noise/ringing
# - laplacian_variance (inverted) penalises ringing artifacts
# - quality_ratio + normalised_power_hf have low weight (can reward noise)
_WEIGHTS = {
    "laplacian_variance": 0.25,   # inverted during scoring
    "tenengrad":          0.30,
    "normalised_power_hf": 0.05,
    "brenner":            0.20,
    "quality_ratio":      0.05,
    "smoothness":         0.15,
}

_INVERTED_METRICS = {"laplacian_variance"}

_METRIC_FNS = {
    "laplacian_variance": laplacian_variance,
    "tenengrad":          tenengrad,
    "normalised_power_hf": normalised_power_hf,
    "brenner":            brenner,
    "quality_ratio":      quality_ratio,
    "smoothness":         smoothness,
}


def all_metrics(image: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute all individual metrics and return as a dict.

    If *mask* is not provided, it is computed once from planet_mask() and
    passed to every metric function (avoids redundant recomputation).
    """
    if mask is None:
        mask = planet_mask(image)
    return {name: fn(image, mask=mask) for name, fn in _METRIC_FNS.items()}


def composite_score(image: np.ndarray, metrics: dict[str, float] | None = None) -> float:
    """
    Weighted composite sharpness score (raw values, not normalised).

    The optimizer normalises scores across the candidate pool before ranking.
    """
    if metrics is None:
        metrics = all_metrics(image)
    total = 0.0
    for name, weight in _WEIGHTS.items():
        value = metrics[name]
        if name in _INVERTED_METRICS:
            total -= weight * value
        else:
            total += weight * value
    return float(total)
