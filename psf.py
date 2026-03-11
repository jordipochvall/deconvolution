"""
PSF (Point Spread Function) models for atmospheric seeing.

Supported models
----------------
- Gaussian : simplest blur approximation
- Moffat   : standard atmospheric PSF (heavier tails than Gaussian)
- Airy     : diffraction-limited PSF (theoretical, rarely applies to seeing)
"""

import numpy as np
from scipy.special import j1


# ---------------------------------------------------------------------------
# Coordinate grid
# ---------------------------------------------------------------------------

def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a centred (x, y) coordinate grid of the given odd size."""
    if size % 2 == 0:
        size += 1
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    return x, y


# ---------------------------------------------------------------------------
# PSF generators
# ---------------------------------------------------------------------------

def gaussian_psf(fwhm: float, size: int) -> np.ndarray:
    """
    Gaussian PSF, normalised to unit sum.

    Parameters
    ----------
    fwhm : float   Full-width at half-maximum (pixels).
    size : int     Kernel size (forced odd).
    """
    x, y = _grid(size)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def moffat_psf(fwhm: float, beta: float, size: int) -> np.ndarray:
    """
    Moffat PSF — the standard model for atmospheric seeing.

    Parameters
    ----------
    fwhm : float   Full-width at half-maximum (pixels).
    beta : float   Power-law index (typically 2–5; lower = heavier tails).
    size : int     Kernel size (forced odd).
    """
    x, y = _grid(size)
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    kernel = (1.0 + (x**2 + y**2) / alpha**2) ** (-beta)
    return kernel / kernel.sum()


def airy_psf(diameter_pixels: float, size: int) -> np.ndarray:
    """
    Airy disk PSF (diffraction-limited).

    Parameters
    ----------
    diameter_pixels : float   Diameter of the first dark ring in pixels
                              (= 2.44 * lambda/D in image pixels).
    size : int                Kernel size (forced odd).
    """
    x, y = _grid(size)
    r = np.sqrt(x**2 + y**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.pi * r / (diameter_pixels / 2.0)
        kernel = np.where(r == 0, 1.0, (2.0 * j1(z) / z) ** 2)
    return kernel / kernel.sum()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

PSF_BUILDERS = {
    "gaussian": lambda p: gaussian_psf(p["fwhm"], p["size"]),
    "moffat":   lambda p: moffat_psf(p["fwhm"], p["beta"], p["size"]),
    "airy":     lambda p: airy_psf(p["diameter_pixels"], p["size"]),
}


def build_psf(psf_type: str, params: dict) -> np.ndarray:
    """Build a PSF kernel by name ('gaussian', 'moffat', or 'airy')."""
    if psf_type not in PSF_BUILDERS:
        raise ValueError(f"Unknown PSF type '{psf_type}'. Choose from {list(PSF_BUILDERS)}")
    return PSF_BUILDERS[psf_type](params)
