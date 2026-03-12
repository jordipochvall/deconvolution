"""
RGB colour pipeline for planetary deconvolution.

Applies deconvolution parameters (optimised on luminance) independently
to each R/G/B channel, then corrects colour artifacts introduced by
per-channel processing:

1. White balance restoration — eliminates colour cast from differential
   noise amplification across Bayer channels with different SNR.
2. Global contrast boost — applies a single luminance-derived linear
   transform to all channels (preserves colour ratios).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fits_io import to_luminance


def _correct_white_balance(rgb_in: np.ndarray, rgb_out: np.ndarray) -> np.ndarray:
    """Restore input channel/luminance ratios on the planet disk.

    RL applied with identical parameters to channels with different SNR
    (RGGB Bayer: R and B have lower SNR than G) causes differential noise
    amplification, producing a colour cast.  This function scales each
    channel so its mean ratio to luminance on the planet disk matches the
    input, eliminating the residual shift.

    Parameters
    ----------
    rgb_in   Original (3, H, W) RGB image.
    rgb_out  Deconvolved (3, H, W) RGB image to correct.

    Returns
    -------
    Corrected (3, H, W) RGB with colour ratios matching the input.
    """
    from metrics import planet_mask

    n_channels = rgb_in.shape[0]
    orig_lum = to_luminance(rgb_in)
    res_lum  = to_luminance(rgb_out)
    mask = planet_mask(orig_lum)
    if not mask.any():
        return rgb_out

    orig_lum_mean = orig_lum[mask].mean()
    res_lum_mean  = res_lum[mask].mean()
    if orig_lum_mean <= 1e-10 or res_lum_mean <= 1e-10:
        return rgb_out

    corrections = [
        (rgb_in[c][mask].mean() / orig_lum_mean) /
        max(rgb_out[c][mask].mean() / res_lum_mean, 1e-10)
        for c in range(n_channels)
    ]
    return np.stack(
        [np.clip(rgb_out[c] * corrections[c], 0, None) for c in range(n_channels)],
        axis=0,
    )


def _apply_global_contrast_boost(
    rgb_in: np.ndarray,
    channels: list[np.ndarray],
    result: np.ndarray,
    boost: float,
) -> np.ndarray:
    """Apply contrast boost as a single global luminance-derived linear transform.

    Instead of stretching each channel independently (which shifts colour),
    this function computes the luminance that per-channel stretching *would*
    produce, then derives a single scale+offset that maps the current
    luminance to that target.  The same transform is applied to all channels,
    preserving colour ratios.

    Parameters
    ----------
    rgb_in    Original (3, H, W) RGB image (for planet mask and reference).
    channels  List of per-channel deconvolved images [R, G, B].
    result    Current (3, H, W) RGB result (after white balance).
    boost     Contrast boost factor (1.0 = no change).

    Returns
    -------
    Boosted (3, H, W) RGB with non-negative pixel values.
    """
    from deconvolve import contrast_stretch
    from metrics import planet_mask

    n_channels = rgb_in.shape[0]
    per_ch_stretched = np.stack(
        [contrast_stretch(channels[c], rgb_in[c], boost=boost)
         for c in range(n_channels)],
        axis=0,
    )
    target_lum = to_luminance(per_ch_stretched)
    result_lum = to_luminance(result)

    orig_lum = to_luminance(rgb_in)
    mask = planet_mask(orig_lum)
    if not mask.any():
        return result

    p1_t, p99_t = np.percentile(target_lum[mask], [1, 99])
    p1_r, p99_r = np.percentile(result_lum[mask], [1, 99])
    span_r = p99_r - p1_r
    if span_r <= 1e-10:
        return result

    scale  = (p99_t - p1_t) / span_r
    offset = p1_t - p1_r * scale
    return np.clip(result * scale + offset, 0, None)


def apply_best_to_color(
    rgb: np.ndarray, candidate, rgb_jobs: int = 1,
) -> np.ndarray:
    """Apply a candidate's PSF + deconvolution to each RGB channel.

    After per-channel deconvolution, two colour-correction steps run:

    1. **White balance** — restores input channel/luminance ratios on the
       planet disk (see ``_correct_white_balance``).
    2. **Global contrast boost** — applies a single luminance-derived
       linear transform to all channels (see ``_apply_global_contrast_boost``).

    The ``contrast_boost`` deconvolution parameter is stripped from per-channel
    processing and handled globally to prevent independent per-channel
    stretching from shifting colours.

    Parameters
    ----------
    rgb        Input (3, H, W) RGB image.
    candidate  Candidate with psf_type, psf_params, deconv_method, deconv_params.
    rgb_jobs   Number of parallel workers for per-channel deconvolution.

    Returns
    -------
    Deconvolved + colour-corrected (3, H, W) RGB image.
    """
    from psf import build_psf
    from deconvolve import deconvolve

    n_channels = rgb.shape[0]
    workers = max(1, min(int(rgb_jobs), n_channels))

    psf = build_psf(candidate.psf_type, candidate.psf_params)

    # Strip contrast_boost: handled globally after white balance
    ch_params = dict(candidate.deconv_params)
    contrast_boost = ch_params.pop("contrast_boost", 1.0)

    def _run_channel(ch: int) -> np.ndarray:
        return deconvolve(candidate.deconv_method, rgb[ch], psf, ch_params)

    if workers == 1:
        channels = [_run_channel(ch) for ch in range(n_channels)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            channels = list(ex.map(_run_channel, range(n_channels)))

    result = np.stack(channels, axis=0)
    result = _correct_white_balance(rgb, result)

    if contrast_boost != 1.0:
        result = _apply_global_contrast_boost(rgb, channels, result, contrast_boost)

    return result
