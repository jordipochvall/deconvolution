"""
FITS image I/O for planetary deconvolution.

Handles loading multi-format FITS files (2-D mono, 3-D RGB cubes) and
saving deconvolved results with metadata headers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from astropy.io import fits
except ImportError:
    sys.exit("astropy is required.  Install with:\n    pip install astropy\n")


def load_fits_image(
    path: str, channel: int | None = None, verbose: bool = True,
) -> tuple[np.ndarray, bool]:
    """Load a FITS image and return (data, is_color).

    Supports 2-D mono images, 3-D RGB cubes (3, H, W), and 3-D cubes
    with channel selection.  NaN/Inf pixels are replaced with the median.

    Parameters
    ----------
    path      File path to a FITS file.
    channel   If set, extract this channel index from a 3-D cube instead
              of treating it as RGB.  ``None`` = auto-detect RGB.
    verbose   Print loading information.

    Returns
    -------
    (data, is_color)  where data is (H, W) float64 for mono or
    (3, H, W) float64 for RGB.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data

    # Fallback: search other HDUs if primary has no data
    if data is None:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break

    if data is None:
        raise ValueError(f"No image data found in {path}")

    data = data.astype(np.float64)

    # Sanitise invalid pixels
    bad = ~np.isfinite(data)
    if bad.any():
        data[bad] = np.nanmedian(data)

    # (3, H, W) -> treat as RGB unless a specific channel was requested
    if data.ndim == 3 and data.shape[0] == 3 and channel is None:
        if verbose:
            print(f"  RGB FITS detected ({data.shape}). Processing all 3 channels.")
        return data, True

    if data.ndim == 3:
        idx = channel if channel is not None else 0
        if idx < 0 or idx >= data.shape[0]:
            raise ValueError(
                f"Channel index {idx} out of range for FITS cube with shape {data.shape}."
            )
        if verbose:
            print(f"  3-D FITS cube ({data.shape}). Using channel {idx}.")
        return data[idx], False

    if data.ndim == 2:
        return data, False

    raise ValueError(f"Unsupported FITS data shape: {data.shape}")


def to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) RGB to luminance (H, W) using BT.601 weights."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def output_fits_name(stem: str, extension: str, rank: int) -> str:
    """Build output FITS filename from the input stem.

    Rank 1 -> ``<stem>_DC<ext>``; higher ranks ->
    ``<stem>_DC_r<rank><ext>`` to avoid overwriting.
    """
    if rank == 1:
        return f"{stem}_DC{extension}"
    return f"{stem}_DC_r{rank}{extension}"


def save_fits(
    data: np.ndarray,
    candidate,
    stem: str,
    extension: str,
    output_dir: Path,
    rank: int,
) -> Path:
    """Save a deconvolved result (mono or RGB) as FITS with metadata headers.

    Headers include RANK, SCORE, PSF_TYPE, FWHM, and DMETHOD so the
    processing parameters are preserved alongside the image data.
    """
    out_path = output_dir / output_fits_name(stem, extension, rank)

    hdr = fits.Header()
    hdr["RANK"]     = rank
    hdr["SCORE"]    = round(candidate.normalised_score, 6)
    hdr["PSF_TYPE"] = candidate.psf_type
    hdr["FWHM"]     = candidate.psf_params.get("fwhm", -1)
    hdr["DMETHOD"]  = candidate.deconv_method

    fits.writeto(str(out_path), data.astype(np.float32), header=hdr, overwrite=True)
    return out_path
