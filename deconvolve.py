"""
Deconvolution algorithms for planetary imaging.

Methods
-------
richardson_lucy          FFT-based RL with TV regularisation, deringing,
                         and limb correction clamping.
richardson_lucy_classic  Basic RL without regularisation (for comparison).
wiener                   Frequency-domain Wiener filter.
tikhonov                 Regularised inverse filter.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
import pywt


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_float(image: np.ndarray) -> np.ndarray:
    """Convert to float64 and clamp negatives to zero."""
    return np.clip(image.astype(np.float64), 0, None)


def _pad_psf_to_image(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-pad PSF to image size and centre it for FFT convolution."""
    h, w = shape
    ph, pw = psf.shape
    padded = np.zeros((h, w), dtype=np.float64)
    padded[:ph, :pw] = psf
    # Shift so the PSF centre sits at (0, 0) — required for FFT convolution
    padded = np.roll(padded, -ph // 2, axis=0)
    padded = np.roll(padded, -pw // 2, axis=1)
    return padded


def _preserve_brightness(
    result: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scale result so its mean brightness matches the original (planet region)."""
    if mask is None:
        from metrics import planet_mask
        mask = planet_mask(original)
    if not mask.any():
        return result
    ratio = original[mask].mean() / max(result[mask].mean(), 1e-10)
    return result * ratio


def _contrast_stretch(result: np.ndarray, original: np.ndarray,
                      boost: float = 1.3) -> np.ndarray:
    """
    Stretch result's dynamic range relative to the original.

    Percentiles are computed on the planet region only so that sky pixels
    don't dominate the stretch for small planets (e.g. Saturn).
    """
    from metrics import planet_mask
    mask = planet_mask(original)
    if not mask.any():
        return result

    # Compute percentiles on the planet region only
    planet_orig = original[mask]
    p1_orig, p99_orig = np.percentile(planet_orig, [1, 99])
    target_range = (p99_orig - p1_orig) * boost

    planet_res = result[mask]
    p1_res, p99_res = np.percentile(planet_res, [1, 99])
    if p99_res - p1_res < 1e-10:
        return result

    stretched = (result - p1_res) / (p99_res - p1_res) * target_range + p1_orig
    return np.clip(stretched, 0, None)


# ---------------------------------------------------------------------------
# TV regularisation
# ---------------------------------------------------------------------------

def _tv_denoise_step(image: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    One step of anisotropic Total Variation diffusion.

    Applied as a small additive correction each RL iteration — stable and
    artifact-free, unlike the Dey et al. divisive formula.
    """
    eps = 1e-8
    # Forward differences
    gy = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    gx[:, :-1] = image[:, 1:] - image[:, :-1]

    # Normalised gradients (edge-preserving)
    norm = np.sqrt(gy**2 + gx**2 + eps)
    ny = gy / norm
    nx = gx / norm

    # Divergence of the normalised gradient field
    div = np.zeros_like(image)
    div[1:, :]  += ny[1:, :] - ny[:-1, :]
    div[0, :]   += ny[0, :]
    div[:, 1:]  += nx[:, 1:] - nx[:, :-1]
    div[:, 0]   += nx[:, 0]

    return image + dt * div


# ---------------------------------------------------------------------------
# Wavelet denoising
# ---------------------------------------------------------------------------

def _wavelet_denoise(image: np.ndarray, threshold: float = 0.1,
                     levels: int = 4, wavelet: str = "bior1.5") -> np.ndarray:
    """
    Wavelet soft-thresholding (similar to PixInsight's wavelet regularisation).

    Decomposes the image via the stationary wavelet transform (SWT),
    soft-thresholds detail coefficients at each scale, and reconstructs.
    Fine-scale noise is removed while large structures are preserved.

    The threshold decreases geometrically with scale: fine scales (noise)
    get full thresholding, coarse scales (real features) get almost none.

    Uses reflect-padding before SWT to prevent periodic boundary artifacts
    at the planet limb.
    """
    min_dim = min(image.shape)
    max_levels = pywt.swt_max_level(min_dim)
    levels = min(levels, max_levels)
    if levels < 1:
        return image

    # Reflect-pad to avoid periodic boundary artifacts.
    # Padded dimensions must be multiples of 2^levels for SWT.
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

    # Stationary wavelet transform (shift-invariant, no decimation)
    coeffs = pywt.swt2(padded, wavelet, level=levels, trim_approx=True)

    # Noise estimate from finest detail level (MAD estimator)
    finest_detail = coeffs[-1]
    noise_sigma = np.median(np.abs(finest_detail[0])) / 0.6745

    # Soft-threshold each detail level with geometrically decreasing strength
    for level_idx in range(1, len(coeffs)):  # skip coeffs[0] (approximation)
        depth = len(coeffs) - level_idx       # 1 = finest, levels = coarsest
        scale_factor = 1.0 / (2 ** (depth - 1))
        thresh_val = threshold * noise_sigma * scale_factor

        coeffs[level_idx] = tuple(
            np.sign(c) * np.maximum(np.abs(c) - thresh_val, 0)
            for c in coeffs[level_idx]
        )

    # Inverse SWT and crop back to original size
    result = pywt.iswt2(coeffs, wavelet)
    return result[pad_h:pad_h + h, pad_w:pad_w + w]


# ---------------------------------------------------------------------------
# Masks for RL regularisation
# ---------------------------------------------------------------------------

def _build_deringing_mask(image: np.ndarray) -> np.ndarray:
    """
    Soft mask protecting bright regions (planet disk) from sky ringing.

    Bright pixels → mask ≈ 1 (full RL correction preserved).
    Dark sky      → mask ≈ 0 (corrections dampened to prevent ringing).

    Uses adaptive planet detection so the mask works correctly for both
    large planets (Jupiter) and small ones (Saturn).
    """
    from metrics import planet_mask
    binary = planet_mask(image)
    # Convert to soft mask: smooth transition at the planet boundary
    soft = gaussian_filter(binary.astype(np.float64), sigma=5.0)
    return np.clip(soft, 0, 1)


def _build_limb_mask(image: np.ndarray, width: int = 15) -> np.ndarray:
    """
    Soft mask that is ≈1.0 at the planet limb and ≈0.0 elsewhere.

    The limb is detected from the INPUT image (before deconvolution) where
    it is clean and well-defined.  The mask covers a band of *width* pixels
    on each side of the limb edge.

    Used by RL to clamp corrections at the limb, preventing Gibbs ringing
    from accumulating over many iterations.
    """
    from scipy.ndimage import sobel as _sobel, binary_dilation, generate_binary_structure

    img = image.astype(np.float64)
    # Pre-smooth to detect only the main limb, not internal bands
    smoothed = gaussian_filter(img, sigma=4.0)
    gx = _sobel(smoothed, axis=1)
    gy = _sobel(smoothed, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # 97th percentile selects only the planet limb (strongest gradients)
    limb_pixels = grad_mag >= np.percentile(grad_mag, 97)

    # Dilate to cover the ringing zone on both sides
    struct = generate_binary_structure(2, 2)  # 8-connectivity
    dilated = binary_dilation(limb_pixels, structure=struct, iterations=width)

    # Gaussian blur for a smooth transition (sigma = width/3)
    return np.clip(gaussian_filter(dilated.astype(np.float64), sigma=width / 3.0), 0, 1)


# ---------------------------------------------------------------------------
# Richardson-Lucy with TV regularisation + limb protection
# ---------------------------------------------------------------------------

def richardson_lucy(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 50,
    damping: float = 0.0,
    tv_lambda: float = 0.0,
    deringing: float = 0.0,
    contrast_boost: float = 1.0,
    wavelet_reg: float = 0.0,
    wavelet_levels: int = 4,
    limb_suppression: float = 0.85,
    dering_mask: np.ndarray | None = None,
    limb_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    FFT-based Richardson-Lucy with TV regularisation and limb protection.

    Key features:
    - Frequency-domain convolution (exact, no boundary accumulation).
    - Per-iteration TV diffusion to suppress noise without killing detail.
    - Deringing mask to dampen corrections in dark sky regions.
    - Limb correction clamping: the multiplicative RL update is dampened
      toward 1.0 in the planet limb zone, preventing Gibbs overshoot from
      accumulating over iterations.  The disk interior is unaffected.

    Parameters
    ----------
    image          Observed (blurred, noisy) image.
    psf            Point spread function (must sum to 1).
    iterations     Number of RL iterations.
    damping        Added to denominators to suppress noise amplification.
    tv_lambda      TV diffusion step size per iteration (0 = disabled).
    deringing      Sky deringing strength 0..1 (0 = disabled).
    contrast_boost Dynamic range multiplier vs original (1.0 = no change).
    wavelet_reg    Post-RL wavelet threshold (0 = disabled).
    wavelet_levels Wavelet decomposition depth (3–5).
    limb_suppression  Correction clamping at limb (0 = off, 0.85 = default,
                      1.0 = limb frozen at initial estimate).
    """
    img = _to_float(image)
    psf = psf.astype(np.float64)

    # Pre-compute PSF transfer function and its conjugate
    H = fft2(_pad_psf_to_image(psf, img.shape))
    H_conj = np.conj(H)

    estimate = img.copy()
    eps = damping if damping > 0 else 1e-12

    # Build masks once before the loop
    if deringing > 0:
        dering_mask = dering_mask if dering_mask is not None else _build_deringing_mask(img)
    else:
        dering_mask = None

    if limb_suppression > 0:
        base_limb_mask = limb_mask if limb_mask is not None else _build_limb_mask(img)
        limb_mask = base_limb_mask * limb_suppression
    else:
        limb_mask = None

    for _ in range(iterations):
        # Forward model: H * estimate
        blurred = np.real(ifft2(H * fft2(estimate)))
        ratio = img / (blurred + eps)

        # Back-projection: H^T * ratio
        correction = np.real(ifft2(H_conj * fft2(ratio)))

        # Deringing: dampen corrections in dark (sky) regions
        if dering_mask is not None:
            correction = 1.0 + (correction - 1.0) * (
                dering_mask + (1.0 - deringing) * (1.0 - dering_mask)
            )

        # Limb clamping: dampen corrections toward 1.0 at the planet limb
        # to prevent Gibbs overshoot from accumulating across iterations
        if limb_mask is not None:
            correction = 1.0 + (correction - 1.0) * (1.0 - limb_mask)

        # Multiplicative RL update
        estimate = np.clip(estimate * correction, 0, None)

        # TV diffusion step (edge-preserving noise suppression)
        if tv_lambda > 0:
            estimate = np.clip(_tv_denoise_step(estimate, dt=tv_lambda), 0, None)

    # Optional post-RL wavelet denoising
    if wavelet_reg > 0:
        estimate = np.clip(
            _wavelet_denoise(estimate, threshold=wavelet_reg, levels=wavelet_levels), 0, None
        )

    # Optional contrast boost
    if contrast_boost != 1.0:
        estimate = _contrast_stretch(estimate, img, boost=contrast_boost)

    return estimate


# ---------------------------------------------------------------------------
# Classic Richardson-Lucy (no regularisation)
# ---------------------------------------------------------------------------

def richardson_lucy_classic(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 50,
    damping: float = 0.0,
) -> np.ndarray:
    """Basic FFT-based Richardson-Lucy without any regularisation."""
    img = _to_float(image)
    psf = psf.astype(np.float64)

    H = fft2(_pad_psf_to_image(psf, img.shape))
    H_conj = np.conj(H)

    estimate = img.copy()
    eps = damping if damping > 0 else 1e-12

    for _ in range(iterations):
        blurred = np.real(ifft2(H * fft2(estimate)))
        correction = np.real(ifft2(H_conj * fft2(img / (blurred + eps))))
        estimate = np.clip(estimate * correction, 0, None)

    return estimate


# ---------------------------------------------------------------------------
# Wiener filter
# ---------------------------------------------------------------------------

def wiener(
    image: np.ndarray,
    psf: np.ndarray,
    snr: float = 30.0,
    brightness_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Frequency-domain Wiener deconvolution.

    F_hat = H* / (|H|² + 1/SNR²) · G

    Higher SNR → less regularisation → sharper but noisier.
    """
    img = _to_float(image)
    H = fft2(_pad_psf_to_image(psf.astype(np.float64), img.shape))
    G = fft2(img)

    H_conj = np.conj(H)
    F_hat = H_conj / (np.abs(H) ** 2 + 1.0 / snr**2) * G

    result = np.clip(np.real(ifft2(F_hat)), 0, None)
    return _preserve_brightness(result, img, mask=brightness_mask)


# ---------------------------------------------------------------------------
# Tikhonov (regularised inverse filter)
# ---------------------------------------------------------------------------

def tikhonov(
    image: np.ndarray,
    psf: np.ndarray,
    regularization: float = 1e-3,
    brightness_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Tikhonov-regularised deconvolution (frequency domain).

    F_hat = H* / (|H|² + λ) · G

    Higher λ → smoother; lower λ → sharper.
    """
    img = _to_float(image)
    H = fft2(_pad_psf_to_image(psf.astype(np.float64), img.shape))
    G = fft2(img)

    H_conj = np.conj(H)
    F_hat = H_conj / (np.abs(H) ** 2 + regularization) * G

    result = np.clip(np.real(ifft2(F_hat)), 0, None)
    return _preserve_brightness(result, img, mask=brightness_mask)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

DECONV_METHODS = {
    "richardson_lucy":         richardson_lucy,
    "richardson_lucy_classic": richardson_lucy_classic,
    "wiener":                  wiener,
    "tikhonov":                tikhonov,
}


def deconvolve(method: str, image: np.ndarray, psf: np.ndarray, params: dict) -> np.ndarray:
    """Run a deconvolution method by name with the given parameters."""
    if method not in DECONV_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(DECONV_METHODS)}")
    return DECONV_METHODS[method](image, psf, **params)
