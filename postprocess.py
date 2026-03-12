"""
Post-processing pipeline for deconvolved planetary images.

Applied after Optuna finds the best RL+TV parameters:
  1. Wavelet soft-thresholding — removes fine-scale noise and ringing residuals.
  2. Non-Local Means (NLM) — patch-based denoising that preserves edges.

Both steps are adaptive: the planet disk gets lighter denoising (preserving
detail) while the limb and sky get stronger denoising (suppressing artifacts).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, \
    generate_binary_structure, median_filter
from deconvolve import _wavelet_denoise


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PostprocessConfig:
    """Group the post-processing parameters that flow through the pipeline.

    This replaces the 4 individual parameters (wv, nlm, sharpen, dp) that
    were passed separately through adapt_postprocessing, rerank_candidates,
    _finalize_result, postprocess, and postprocess_rgb.
    """
    wv: float = 25.0
    """Wavelet soft-threshold strength (0 = disabled)."""

    nlm: float = 0.008
    """Non-Local Means filter strength (0 = disabled)."""

    sharpen: float = 0.0
    """Wavelet sharpening gain (0 = disabled, 1.5 = moderate, 2.0 = strong)."""

    dp: float = 0.5
    """Disk preservation: 0.0 = uniform, 0.5 = 50% less denoise on disk, 1.0 = skip disk."""

    level_gains: list[float] | None = None
    """Per-wavelet-level sharpening gains (finest first). None = use bell curve from ``sharpen``."""

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
    from wavelet_utils import swt_pad, swt_unpad

    min_dim = min(image.shape)
    max_levels = pywt.swt_max_level(min_dim)
    levels = min(levels, max_levels)
    if levels < 2:
        return image

    h, w = image.shape
    padded, pad_h, pad_w = swt_pad(image, levels)

    coeffs = pywt.swt2(padded, wavelet, level=levels, trim_approx=True)

    # Noise estimate from finest level (MAD)
    noise_sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745

    # Threshold ONLY the finest level (last in the list)
    thresh_val = strength * noise_sigma
    coeffs[-1] = tuple(
        np.sign(c) * np.maximum(np.abs(c) - thresh_val, 0)
        for c in coeffs[-1]
    )

    result = pywt.iswt2(coeffs, wavelet)
    return swt_unpad(result, h, w, pad_h, pad_w)


def _wavelet_sharpen(image: np.ndarray, gain: float = 1.5,
                     levels: int = 4, wavelet: str = "bior1.5",
                     protect_fine: bool = True,
                     level_gains: list[float] | None = None) -> np.ndarray:
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
                   Ignored when *level_gains* is provided.
    levels         SWT decomposition depth (3–5).
    wavelet        Wavelet family (default bior1.5, good edge preservation).
    protect_fine   If True, leave the finest scale untouched (noise protection).
                   Ignored when *level_gains* is provided.
    level_gains    Per-level gains (finest scale first, coarsest last).
                   When provided, overrides the bell-curve logic entirely.
                   1.0 = unchanged, >1.0 = amplify, <1.0 = attenuate.
    """
    if level_gains is None and gain <= 1.0:
        return image
    if level_gains is not None and all(g <= 1.0 for g in level_gains):
        return image

    from wavelet_utils import swt_pad, swt_unpad

    min_dim = min(image.shape)
    max_levels = pywt.swt_max_level(min_dim)
    levels = min(levels, max_levels)
    if levels < 2:
        return image

    h, w = image.shape
    padded, pad_h, pad_w = swt_pad(image, levels)

    coeffs = pywt.swt2(padded, wavelet, level=levels, trim_approx=True)

    n_detail = len(coeffs) - 1
    for level_idx in range(1, len(coeffs)):
        depth = len(coeffs) - level_idx  # 1 = finest, n_detail = coarsest

        if level_gains is not None:
            # Per-level mode: depth 1 → index 0 (finest first)
            gi = depth - 1
            scale_gain = level_gains[gi] if gi < len(level_gains) else 1.0
        else:
            # Bell-curve mode (original behaviour)
            if protect_fine and depth == 1:
                scale_gain = 1.0
            elif depth == n_detail:
                scale_gain = 1.0
            else:
                centre = n_detail / 2.0
                dist = abs(depth - centre) / max(centre, 1.0)
                scale_gain = 1.0 + (gain - 1.0) * (1.0 - dist ** 2)

        if scale_gain != 1.0:
            coeffs[level_idx] = tuple(c * scale_gain for c in coeffs[level_idx])

    result = pywt.iswt2(coeffs, wavelet)
    return swt_unpad(result, h, w, pad_h, pad_w)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _blend_with_disk_mask(
    original: np.ndarray,
    processed: np.ndarray,
    disk_mask: np.ndarray | None,
    preservation: float,
) -> np.ndarray:
    """Blend processed result with original using disk preservation mask.

    On the planet disk, more of the original is preserved (less denoising).
    On sky/limb, the processed result is used fully.
    """
    if disk_mask is not None and preservation > 0:
        blend = disk_mask * preservation
        return blend * original + (1.0 - blend) * processed
    return processed


def _apply_nlm(
    result: np.ndarray,
    nlm_h: float,
    disk_mask: np.ndarray | None,
    dp: float,
) -> np.ndarray:
    """Apply Non-Local Means denoising with adaptive disk preservation."""
    if nlm_h <= 0:
        return result
    if not _HAS_SKIMAGE:
        print("  WARNING: scikit-image not installed, skipping NLM. "
              "Install with: pip install scikit-image")
        return result

    vmin, vmax = result.min(), result.max()
    span = vmax - vmin
    if span <= 1e-30:
        return result

    norm = (result - vmin) / span
    denoised = denoise_nl_means(
        norm, h=nlm_h, patch_size=5, patch_distance=7, fast_mode=True,
    )
    denoised = denoised * span + vmin
    return _blend_with_disk_mask(result, denoised, disk_mask, dp)


def postprocess(
    image: np.ndarray,
    cfg: PostprocessConfig | None = None,
    *,
    wavelet_levels: int = 4,
    limb_deringing: float = 0.0,
    limb_width: int = 8,
    median_size: int = 5,
    # Legacy keyword-only args (used when cfg is None)
    wavelet_threshold: float = 10.0,
    nlm_h: float = 0.003,
    disk_preservation: float = 0.5,
    sharpen_gain: float = 0.0,
) -> np.ndarray:
    """Adaptive wavelet + NLM post-processing for deconvolved images."""
    if cfg is None:
        cfg = PostprocessConfig(wv=wavelet_threshold, nlm=nlm_h,
                                sharpen=sharpen_gain, dp=disk_preservation)

    result = _deringing_limb(image.copy(), strength=limb_deringing,
                             width=limb_width, median_size=median_size)

    disk_mask = _adaptive_blend_mask(image) if cfg.dp > 0 else None

    # Wavelet denoising
    if cfg.wv > 0:
        denoised = np.clip(
            _wavelet_denoise(result, threshold=cfg.wv, levels=wavelet_levels),
            0, None,
        )
        result = _blend_with_disk_mask(result, denoised, disk_mask, cfg.dp)

    # Wavelet sharpening (opposite blend: apply sharpening ON disk, not sky)
    if cfg.sharpen > 1.0 or cfg.level_gains is not None:
        sharpened = _wavelet_sharpen(result, gain=cfg.sharpen, levels=wavelet_levels,
                                     level_gains=cfg.level_gains)
        if disk_mask is not None:
            result = _blend_with_disk_mask(sharpened, result, disk_mask, 1.0)
        else:
            result = sharpened

    # NLM denoising
    result = _apply_nlm(result, cfg.nlm, disk_mask, cfg.dp)

    return result


def postprocess_rgb(
    rgb: np.ndarray,
    cfg: PostprocessConfig | None = None,
    *,
    wavelet_levels: int = 4,
    limb_deringing: float = 0.0,
    limb_width: int = 8,
    median_size: int = 5,
    jobs: int = 1,
    # Legacy keyword-only args (used when cfg is None)
    wavelet_threshold: float = 10.0,
    nlm_h: float = 0.003,
    disk_preservation: float = 0.5,
    sharpen_gain: float = 0.0,
) -> np.ndarray:
    """Apply post-processing to each channel of an (3, H, W) image."""
    if cfg is None:
        cfg = PostprocessConfig(wv=wavelet_threshold, nlm=nlm_h,
                                sharpen=sharpen_gain, dp=disk_preservation)

    n_channels = rgb.shape[0]
    workers = max(1, min(int(jobs), n_channels))

    def _run_channel(ch: int) -> np.ndarray:
        return postprocess(
            rgb[ch], cfg,
            wavelet_levels=wavelet_levels,
            limb_deringing=limb_deringing,
            limb_width=limb_width,
            median_size=median_size,
        )

    if workers == 1:
        channels = [_run_channel(ch) for ch in range(n_channels)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            channels = list(ex.map(_run_channel, range(n_channels)))

    return np.stack(channels, axis=0)
