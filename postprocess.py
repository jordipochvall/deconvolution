"""
Post-processing pipeline for deconvolved planetary images.

Applied after Optuna finds the best RL+TV parameters:
  1. Wavelet soft-thresholding — removes fine-scale noise and ringing residuals.
  2. Non-Local Means (NLM) — patch-based denoising that preserves edges.

Both steps are adaptive: the planet disk gets lighter denoising (preserving
detail) while the limb and sky get stronger denoising (suppressing artifacts).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, \
    generate_binary_structure, median_filter
from deconvolve import _wavelet_denoise

try:
    from skimage.restoration import denoise_nl_means
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Limb deringing (median-based, edge-preserving)
# ---------------------------------------------------------------------------

def _limb_zone_mask(image: np.ndarray, width: int = 8) -> np.ndarray:
    """
    Soft mask ≈1.0 in a narrow annulus around the planet limb, ≈0.0 elsewhere.

    The limb is detected as the zone of strongest gradient magnitude
    (the sharp brightness transition between planet and sky).
    """
    img = image.astype(np.float64)
    smoothed = gaussian_filter(img, sigma=2.0)
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # 97th percentile → only the planet edge, not internal bands
    limb_pixels = grad_mag >= np.percentile(grad_mag, 97)

    # Dilate to create an annulus of *width* pixels around the limb
    struct = generate_binary_structure(2, 2)  # 8-connectivity
    dilated = binary_dilation(limb_pixels, structure=struct, iterations=width)

    # Smooth transition (sigma = width/3 keeps it localised)
    mask = gaussian_filter(dilated.astype(np.float64), sigma=width / 3.0)
    return np.clip(mask, 0, 1)


def _deringing_limb(image: np.ndarray, strength: float = 1.0,
                    width: int = 8, median_size: int = 5,
                    iterations: int = 2) -> np.ndarray:
    """
    Suppress Gibbs ringing around the planet limb with iterated median filtering.

    Median filters are ideal here: they eliminate oscillatory artifacts while
    preserving step edges (the limb itself stays sharp) and are idempotent on
    monotonic regions (disk detail is untouched).

    Multiple passes with a small kernel are gentler than one pass with a large
    kernel — they progressively damp oscillations without blocky artifacts.
    """
    if strength <= 0:
        return image

    limb_mask = _limb_zone_mask(image, width=width)

    # Iterated median: each pass further damps oscillations
    filtered = image.copy()
    for _ in range(iterations):
        filtered = median_filter(filtered, size=median_size)

    # Blend: only replace pixels in the limb zone
    blend = limb_mask * strength
    return (1.0 - blend) * image + blend * filtered


# ---------------------------------------------------------------------------
# Adaptive disk mask
# ---------------------------------------------------------------------------

def _adaptive_blend_mask(image: np.ndarray) -> np.ndarray:
    """
    Smooth mask for spatially adaptive denoising strength.

    ≈1.0 on planet disk interior (less denoising → preserve detail).
    ≈0.0 on sky / limb region (full denoising → suppress artifacts).

    Percentiles are computed on the detected planet region so the mask
    works correctly for both large and small planets.
    """
    from metrics import planet_mask
    pmask = planet_mask(image)
    if not pmask.any():
        return np.ones_like(image)

    # Use percentiles within the planet region to define the gradient
    planet_vals = image[pmask]
    p30 = np.percentile(planet_vals, 30)
    p90 = np.percentile(planet_vals, 90)
    span = p90 - p30
    if span < 1e-10:
        return np.ones_like(image)
    mask = np.clip((image - p30) / span, 0, 1)
    return np.clip(gaussian_filter(mask, sigma=8.0), 0, 1)


# ---------------------------------------------------------------------------
# Wavelet sharpening (frequency-selective detail enhancement)
# ---------------------------------------------------------------------------

def _wavelet_denoise_fine(image: np.ndarray, strength: float = 3.0,
                          levels: int = 3, wavelet: str = "bior1.5") -> np.ndarray:
    """
    Wavelet denoising that ONLY thresholds the finest scale.

    Unlike full wavelet denoising (which thresholds all detail levels and
    destroys real features on small planets), this targets exclusively the
    finest SWT level — where noise dominates — and leaves medium/coarse
    detail coefficients completely untouched.

    This reduces laplacian_variance (fine-scale noise) without affecting
    tenengrad (medium-scale edge energy), improving quality_ratio.
    """
    min_dim = min(image.shape)
    max_levels = pywt.swt_max_level(min_dim)
    levels = min(levels, max_levels)
    if levels < 2:
        return image

    # Reflect-pad
    pad_size = max(32, 2 ** levels)
    factor = 2 ** levels
    h, w = image.shape
    ph = int(np.ceil((h + 2 * pad_size) / factor) * factor)
    pw = int(np.ceil((w + 2 * pad_size) / factor) * factor)
    pad_h = (ph - h) // 2
    pad_w = (pw - w) // 2
    padded = np.pad(image,
                    ((pad_h, ph - h - pad_h), (pad_w, pw - w - pad_w)),
                    mode="reflect")

    coeffs = pywt.swt2(padded, wavelet, level=levels, trim_approx=True)

    # Noise estimate from finest level (MAD)
    finest = coeffs[-1]
    noise_sigma = np.median(np.abs(finest[0])) / 0.6745

    # Threshold ONLY the finest level (last in the list)
    thresh_val = strength * noise_sigma
    coeffs[-1] = tuple(
        np.sign(c) * np.maximum(np.abs(c) - thresh_val, 0)
        for c in coeffs[-1]
    )

    result = pywt.iswt2(coeffs, wavelet)
    return np.clip(result[pad_h:pad_h + h, pad_w:pad_w + w], 0, None)


def _wavelet_sharpen(image: np.ndarray, gain: float = 1.5,
                     levels: int = 4, wavelet: str = "bior1.5",
                     protect_fine: bool = True) -> np.ndarray:
    """
    Enhance planetary detail by amplifying wavelet detail coefficients.

    Similar to PixInsight's Multiscale Linear Transform: decomposes the image
    into spatial frequency layers via the stationary wavelet transform, then
    selectively boosts medium-scale detail (planetary bands, ring gaps, cloud
    features) while leaving the finest scale (noise) and coarsest scales
    (overall shape) untouched.

    Parameters
    ----------
    image          Input image (H, W), float.
    gain           Enhancement factor for the peak detail scale (>1 = sharpen).
    levels         SWT decomposition depth (3–5).
    wavelet        Wavelet family (default bior1.5, good edge preservation).
    protect_fine   If True, leave the finest scale untouched (noise protection).
    """
    if gain <= 1.0:
        return image

    min_dim = min(image.shape)
    max_levels = pywt.swt_max_level(min_dim)
    levels = min(levels, max_levels)
    if levels < 2:
        return image

    # Reflect-pad for SWT (same strategy as _wavelet_denoise)
    pad_size = max(32, 2 ** levels)
    factor = 2 ** levels
    h, w = image.shape
    ph = int(np.ceil((h + 2 * pad_size) / factor) * factor)
    pw = int(np.ceil((w + 2 * pad_size) / factor) * factor)
    pad_h = (ph - h) // 2
    pad_w = (pw - w) // 2
    padded = np.pad(image,
                    ((pad_h, ph - h - pad_h), (pad_w, pw - w - pad_w)),
                    mode="reflect")

    coeffs = pywt.swt2(padded, wavelet, level=levels, trim_approx=True)

    # Scale-dependent gain: bell curve peaking at medium scales.
    # Level 1 (finest) = noise → gain 1.0 if protect_fine else full gain.
    # Levels 2–3 = planetary detail → peak gain.
    # Level N (coarsest) = planet shape → gain 1.0.
    n_detail = len(coeffs) - 1  # number of detail levels
    for level_idx in range(1, len(coeffs)):
        depth = len(coeffs) - level_idx  # 1 = finest, n_detail = coarsest

        if protect_fine and depth == 1:
            scale_gain = 1.0  # don't amplify noise
        elif depth == n_detail:
            scale_gain = 1.0  # don't distort overall shape
        else:
            # Bell-shaped: peak at intermediate scales
            centre = n_detail / 2.0
            dist = abs(depth - centre) / max(centre, 1.0)
            scale_gain = 1.0 + (gain - 1.0) * (1.0 - dist ** 2)

        if scale_gain != 1.0:
            coeffs[level_idx] = tuple(c * scale_gain for c in coeffs[level_idx])

    result = pywt.iswt2(coeffs, wavelet)
    return np.clip(result[pad_h:pad_h + h, pad_w:pad_w + w], 0, None)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def postprocess(
    image: np.ndarray,
    wavelet_threshold: float = 10.0,
    wavelet_levels: int = 4,
    nlm_h: float = 0.003,
    disk_preservation: float = 0.5,
    limb_deringing: float = 0.0,
    limb_width: int = 8,
    median_size: int = 5,
    sharpen_gain: float = 0.0,
) -> np.ndarray:
    """
    Adaptive wavelet + NLM post-processing for deconvolved images.

    Parameters
    ----------
    image              Deconvolved image (H, W), float.
    wavelet_threshold  Wavelet soft-threshold strength (0 = disabled).
    wavelet_levels     Wavelet decomposition levels (3–5).
    nlm_h              NLM filter strength (0 = disabled).
    disk_preservation  Reduce denoising on planet disk: 0.0 = uniform,
                       0.5 = disk gets 50% less, 1.0 = disk untouched.
    limb_deringing     Limb median filter strength 0..1 (0 = disabled).
    limb_width         Half-width of the limb zone (pixels).
    median_size        Median filter kernel size (must be odd).
    sharpen_gain       Wavelet sharpening gain (0 = disabled, 1.5 = moderate,
                       2.0 = strong).  Enhances medium-scale detail (bands,
                       ring gaps) without amplifying fine-scale noise.
    """
    result = image.copy()

    # Step 1: Optional limb deringing (before denoising)
    result = _deringing_limb(result, strength=limb_deringing,
                             width=limb_width, median_size=median_size)

    # Build adaptive mask once (reused for wavelet and NLM)
    disk_mask = _adaptive_blend_mask(image) if disk_preservation > 0 else None

    # Step 2: Wavelet denoising (adaptive)
    if wavelet_threshold > 0:
        denoised = np.clip(
            _wavelet_denoise(result, threshold=wavelet_threshold, levels=wavelet_levels),
            0, None,
        )
        if disk_mask is not None:
            blend = disk_mask * disk_preservation
            result = blend * result + (1.0 - blend) * denoised
        else:
            result = denoised

    # Step 3: Wavelet sharpening (frequency-selective detail enhancement)
    if sharpen_gain > 1.0:
        sharpened = _wavelet_sharpen(result, gain=sharpen_gain,
                                     levels=wavelet_levels)
        if disk_mask is not None:
            # Apply sharpening only on the planet disk (not on sky)
            blend = disk_mask
            result = blend * sharpened + (1.0 - blend) * result
        else:
            result = sharpened

    # Step 4: Non-Local Means denoising (adaptive)
    if nlm_h > 0:
        if not _HAS_SKIMAGE:
            print("  WARNING: scikit-image not installed, skipping NLM. "
                  "Install with: pip install scikit-image")
        else:
            vmin, vmax = result.min(), result.max()
            span = vmax - vmin
            if span > 1e-30:
                norm = (result - vmin) / span
                denoised_nlm = denoise_nl_means(
                    norm, h=nlm_h, patch_size=5, patch_distance=7, fast_mode=True,
                )
                denoised_nlm = denoised_nlm * span + vmin
                if disk_mask is not None:
                    blend = disk_mask * disk_preservation
                    result = blend * result + (1.0 - blend) * denoised_nlm
                else:
                    result = denoised_nlm

    return result


def postprocess_rgb(
    rgb: np.ndarray,
    wavelet_threshold: float = 10.0,
    wavelet_levels: int = 4,
    nlm_h: float = 0.003,
    disk_preservation: float = 0.5,
    limb_deringing: float = 0.0,
    limb_width: int = 8,
    median_size: int = 5,
    sharpen_gain: float = 0.0,
    jobs: int = 1,
) -> np.ndarray:
    """Apply post-processing to each channel of an (3, H, W) image."""
    n_channels = rgb.shape[0]
    workers = max(1, min(int(jobs), n_channels))

    def _run_channel(ch: int) -> np.ndarray:
        return postprocess(
            rgb[ch],
            wavelet_threshold=wavelet_threshold,
            wavelet_levels=wavelet_levels,
            nlm_h=nlm_h,
            disk_preservation=disk_preservation,
            limb_deringing=limb_deringing,
            limb_width=limb_width,
            median_size=median_size,
            sharpen_gain=sharpen_gain,
        )

    if workers == 1:
        channels = [_run_channel(ch) for ch in range(n_channels)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            channels = list(ex.map(_run_channel, range(n_channels)))

    return np.stack(channels, axis=0)
