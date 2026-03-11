"""
Integration tests: run the full pipeline on real images and compare against
external reference deconvolutions.

Each test_images/imageN/ directory contains:
  - input/   : original FITS file
  - reference/: externally deconvolved reference (e.g. PixInsight)

The pipeline must produce results that match or exceed the reference for
every image.  Scoring uses the same proportional weighted-ratio method
as test_compare.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

# Add project root to path so imports work when running from tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import all_metrics, planet_mask, _WEIGHTS
from optimizer import run_search, _INVERTED_METRICS
from postprocess import postprocess, postprocess_rgb

TEST_DIR = PROJECT_ROOT / "test_images"

N_TRIALS = 100


def _load_fits(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
    if data is None:
        raise ValueError(f"No image data in {path}")
    arr = np.asarray(data, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = np.nanmedian(arr)
    return arr


def _to_luminance(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 3:
        return 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"Unsupported shape: {arr.shape}")


def _score_vs_reference(output_lum: np.ndarray, ref_lum: np.ndarray) -> float:
    """
    Proportional weighted ratio: output/reference per metric.

    Returns a value >= 1.0 when the output matches or exceeds the reference.
    """
    out_metrics = all_metrics(output_lum)
    ref_metrics = all_metrics(ref_lum)

    weighted_ratio = 0.0
    for metric_name, weight in _WEIGHTS.items():
        ov, rv = out_metrics[metric_name], ref_metrics[metric_name]
        if metric_name in _INVERTED_METRICS:
            ratio = rv / ov if ov > 1e-30 else 1.0
        else:
            ratio = ov / rv if rv > 1e-30 else 1.0
        weighted_ratio += weight * ratio

    return weighted_ratio


def _run_pipeline(input_image: np.ndarray) -> np.ndarray:
    """
    Run the full pipeline on an input image: Optuna search + post-processing.

    Returns the final deconvolved image (same shape as input).
    """
    is_color = input_image.ndim == 3 and input_image.shape[0] == 3

    if is_color:
        opt_image = _to_luminance(input_image)
    else:
        opt_image = input_image

    # Bayesian search
    candidates = run_search(opt_image, n_trials=N_TRIALS, verbose=False)
    best = candidates[0]

    # Determine post-processing strength
    pmask = planet_mask(opt_image)
    planet_frac = pmask.sum() / pmask.size
    small_planet = planet_frac < 0.20

    if small_planet:
        wv_strength = 0.0
        nlm_strength = 0.003
        sharpen_gain = 1.5
    else:
        wv_strength = 25.0
        nlm_strength = 0.008
        sharpen_gain = 0.0

    # Re-rank top candidates after post-processing (mirrors main.py logic)
    if len(candidates) > 1:
        rerank_n = min(20, len(candidates))
        rerank_pool = list(candidates[:rerank_n])
        seen_ids = {id(c) for c in rerank_pool}
        for c in candidates:
            if getattr(c, "source", "optuna") == "seed" and id(c) not in seen_ids:
                rerank_pool.append(c)
                seen_ids.add(id(c))

        best_final_score = -np.inf
        cached_best_result = None

        for c in rerank_pool:
            if is_color:
                from deconvolve import deconvolve
                from psf import build_psf
                psf = build_psf(c.psf_type, c.psf_params)
                channels = [deconvolve(c.deconv_method, input_image[ch], psf, c.deconv_params)
                            for ch in range(3)]
                trial_result = np.stack(channels, axis=0)
                trial_result = postprocess_rgb(
                    trial_result,
                    wavelet_threshold=wv_strength,
                    nlm_h=nlm_strength,
                    sharpen_gain=sharpen_gain,
                )
                lum = _to_luminance(trial_result)
            else:
                trial_result = postprocess(
                    c.result,
                    wavelet_threshold=wv_strength,
                    nlm_h=nlm_strength,
                    sharpen_gain=sharpen_gain,
                )
                lum = trial_result

            metrics = all_metrics(lum, mask=pmask)

            # Normalise and score (simplified inline version)
            # We collect all metrics first, then normalise
            from optimizer import _composite_normalised, _normalise_metrics
            # Store for later normalisation
            c._rerank_result = trial_result
            c._rerank_metrics = metrics

        # Normalise across pool
        for name in _WEIGHTS:
            values = np.array([c._rerank_metrics[name] for c in rerank_pool])
            lo, hi = values.min(), values.max()
            span = hi - lo if hi > lo else 1.0
            for c in rerank_pool:
                c._rerank_metrics[f"{name}_norm"] = (c._rerank_metrics[name] - lo) / span

        for c in rerank_pool:
            final_score = 0.0
            for k, w in _WEIGHTS.items():
                v = c._rerank_metrics.get(f"{k}_norm", 0.0)
                if k in _INVERTED_METRICS:
                    v = 1.0 - v
                final_score += w * v
            if final_score > best_final_score:
                best_final_score = final_score
                best = c
                cached_best_result = c._rerank_result

        if cached_best_result is not None:
            return cached_best_result

    # Fallback: apply best without re-ranking
    if is_color:
        from deconvolve import deconvolve
        from psf import build_psf
        psf = build_psf(best.psf_type, best.psf_params)
        channels = [deconvolve(best.deconv_method, input_image[ch], psf, best.deconv_params)
                    for ch in range(3)]
        result = np.stack(channels, axis=0)
        result = postprocess_rgb(result, wavelet_threshold=wv_strength,
                                 nlm_h=nlm_strength, sharpen_gain=sharpen_gain)
    else:
        result = postprocess(best.result, wavelet_threshold=wv_strength,
                             nlm_h=nlm_strength, sharpen_gain=sharpen_gain)

    return result


def _find_fits_files(directory: Path) -> list[Path]:
    """Find all FITS files in a directory."""
    if not directory.is_dir():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in (".fit", ".fits")
    )


@unittest.skipUnless(TEST_DIR.is_dir(), f"test_images/ not found at {TEST_DIR}")
class TestPipelineVsReference(unittest.TestCase):
    """Run the full pipeline on each test image and compare to reference."""

    def _run_image_test(self, image_dir: Path) -> None:
        input_dir = image_dir / "input"
        ref_dir = image_dir / "reference"

        input_files = _find_fits_files(input_dir)
        ref_files = _find_fits_files(ref_dir)

        self.assertTrue(len(input_files) > 0, f"No input FITS in {input_dir}")
        self.assertTrue(len(ref_files) > 0, f"No reference FITS in {ref_dir}")

        # Load
        input_image = _load_fits(input_files[0])
        ref_image = _load_fits(ref_files[0])

        # Run pipeline
        result = _run_pipeline(input_image)

        # Compare
        result_lum = _to_luminance(result)
        ref_lum = _to_luminance(ref_image)

        score = _score_vs_reference(result_lum, ref_lum)
        pct = score * 100

        print(f"\n  [{image_dir.name}] Score: {pct:.1f}% of reference", flush=True)

        self.assertGreaterEqual(
            score, 1.0,
            f"{image_dir.name}: pipeline score {pct:.1f}% < 100% of reference"
        )


# Dynamically create a test method for each image directory
def _make_test(image_dir: Path):
    def test_method(self):
        self._run_image_test(image_dir)
    test_method.__doc__ = f"Pipeline vs reference: {image_dir.name}"
    return test_method


if TEST_DIR.is_dir():
    for _img_dir in sorted(d for d in TEST_DIR.iterdir() if d.is_dir()):
        _test_name = f"test_{_img_dir.name}"
        setattr(TestPipelineVsReference, _test_name, _make_test(_img_dir))


if __name__ == "__main__":
    unittest.main()
