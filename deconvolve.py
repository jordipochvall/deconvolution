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


def contrast_stretch(result: np.ndarray, original: np.ndarray,
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
    from wavelet_utils import swt_pad, swt_unpad

    min_dim = min(image.shape)
    max_levels = pywt.swt_max_level(min_dim)
    levels = min(levels, max_levels)
    if levels < 1:
        return image

    h, w = image.shape
    padded, pad_h, pad_w = swt_pad(image, levels)

    coeffs = pywt.swt2(padded, wavelet, level=levels, trim_approx=True)

    # Noise estimate from finest detail level (MAD estimator)
    noise_sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745

    # Soft-threshold each detail level with geometrically decreasing strength
    for level_idx in range(1, len(coeffs)):
        depth = len(coeffs) - level_idx
        scale_factor = 1.0 / (2 ** (depth - 1))
        thresh_val = threshold * noise_sigma * scale_factor

        coeffs[level_idx] = tuple(
            np.sign(c) * np.maximum(np.abs(c) - thresh_val, 0)
            for c in coeffs[level_idx]
        )

    result = pywt.iswt2(coeffs, wavelet)
    return swt_unpad(result, h, w, pad_h, pad_w)


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

def build_rl_masks(
    image: np.ndarray,
    deringing: float,
    limb_suppression: float,
    limb_width: int = 15,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pre-compute RL regularisation masks from the input image.

    Useful when running RL in segments (checkpointing for trial pruning):
    compute the masks once from the original image, then pass them to each
    segment to guarantee identical results to a single continuous run.

    Returns (dering_mask, limb_mask) — either may be None if disabled.
    """
    dm = _build_deringing_mask(image) if deringing > 0 else None
    lm = _build_limb_mask(image, width=limb_width) if limb_suppression > 0 else None
    return dm, lm


def _rl_setup_masks(
    img: np.ndarray,
    deringing: float,
    limb_suppression: float,
    dering_mask: np.ndarray | None,
    limb_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Prepare deringing and limb masks for the RL iteration loop."""
    if deringing > 0:
        dm = dering_mask if dering_mask is not None else _build_deringing_mask(img)
    else:
        dm = None

    if limb_suppression > 0:
        base = limb_mask if limb_mask is not None else _build_limb_mask(img)
        lm = base * limb_suppression
    else:
        lm = None
    return dm, lm


def _rl_compute_correction(
    img: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    H_conj: np.ndarray,
    eps: float,
    deringing: float,
    dering_mask: np.ndarray | None,
    limb_mask: np.ndarray | None,
) -> np.ndarray:
    """Compute one RL correction step: forward model, ratio, back-projection, masking."""
    blurred = np.real(ifft2(H * fft2(y)))
    ratio = img / (blurred + eps)
    correction = np.real(ifft2(H_conj * fft2(ratio)))

    if dering_mask is not None:
        correction = 1.0 + (correction - 1.0) * (
            dering_mask + (1.0 - deringing) * (1.0 - dering_mask)
        )
    if limb_mask is not None:
        correction = 1.0 + (correction - 1.0) * (1.0 - limb_mask)
    return correction


def _rl_finalize(
    estimate: np.ndarray,
    img: np.ndarray,
    wavelet_reg: float,
    wavelet_levels: int,
    contrast_boost: float,
) -> np.ndarray:
    """Apply optional post-RL wavelet denoising and contrast boost."""
    if wavelet_reg > 0:
        estimate = np.clip(
            _wavelet_denoise(estimate, threshold=wavelet_reg, levels=wavelet_levels),
            0, None,
        )
    if contrast_boost != 1.0:
        estimate = contrast_stretch(estimate, img, boost=contrast_boost)
    return estimate


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
    initial_estimate: np.ndarray | None = None,
    use_nesterov: bool = False,
) -> np.ndarray:
    """FFT-based Richardson-Lucy with TV regularisation and limb protection.

    Features: frequency-domain convolution, per-iteration TV diffusion,
    deringing mask, limb correction clamping, optional Nesterov momentum,
    and warm-start from a previous segment.
    """
    img = _to_float(image)
    psf = psf.astype(np.float64)

    H = fft2(_pad_psf_to_image(psf, img.shape))
    H_conj = np.conj(H)

    estimate = _to_float(initial_estimate) if initial_estimate is not None else img.copy()
    eps = damping if damping > 0 else 1e-12

    dm, lm = _rl_setup_masks(img, deringing, limb_suppression, dering_mask, limb_mask)

    prev_estimate = estimate.copy() if use_nesterov else None

    for k in range(iterations):
        # Nesterov momentum: extrapolate with k/(k+3) annealing schedule
        if use_nesterov and k > 0:
            beta = k / (k + 3.0)
            y = np.clip(estimate + beta * (estimate - prev_estimate), 1e-10, None)
        else:
            y = estimate

        if use_nesterov:
            prev_estimate = estimate.copy()

        correction = _rl_compute_correction(img, y, H, H_conj, eps, deringing, dm, lm)
        estimate = np.clip(y * correction, 0, None)

        if tv_lambda > 0:
            estimate = np.clip(_tv_denoise_step(estimate, dt=tv_lambda), 0, None)

    return _rl_finalize(estimate, img, wavelet_reg, wavelet_levels, contrast_boost)


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
