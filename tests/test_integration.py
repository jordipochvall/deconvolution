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

from main import adapt_postprocessing
from rerank import rerank_candidates
from rgb import apply_best_to_color
from metrics import all_metrics, planet_mask, _WEIGHTS, _INVERTED_METRICS
from optimizer import run_search
from postprocess import PostprocessConfig, postprocess, postprocess_rgb

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

    # Determine post-processing strength (same defaults as CLI)
    pp_cfg = adapt_postprocessing(
        opt_image, PostprocessConfig(wv=25.0, nlm=0.008, sharpen=0.0),
    )

    # Re-rank top candidates after post-processing
    best, cached_best_result = rerank_candidates(
        candidates, input_image, opt_image, is_color, pp_cfg,
    )

    if cached_best_result is not None:
        return cached_best_result

    # Fallback: apply best without re-ranking
    if is_color:
        result = apply_best_to_color(input_image, best)
        result = postprocess_rgb(result, pp_cfg)
    else:
        result = postprocess(best.result, pp_cfg)

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
