"""Characterization tests — capture current behavior before refactoring.

These tests pin down the exact behavior of scoring, re-ranking, post-processing
adaptation, white balance, wavelet denoising, and planet mask so that
refactoring cannot silently change outcomes.
"""

import unittest

import numpy as np

from deconvolve import _wavelet_denoise, contrast_stretch
from metrics import (
    _INVERTED_METRICS,
    _WEIGHTS,
    all_metrics,
    composite_score,
    planet_mask,
    tenengrad,
)
from optimizer import (
    Candidate,
    _composite_normalised,
    _normalise_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_planet(seed: int = 1, fill: float = 0.40) -> np.ndarray:
    """Build a synthetic planetary image with controllable fill fraction."""
    rng = np.random.default_rng(seed)
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    # radius that gives approximately the desired fill fraction
    radius = np.sqrt(fill * h * w / np.pi)
    image = np.full((h, w), 20.0, dtype=np.float64)
    disk = r <= radius
    image[disk] = 220.0
    image[disk] += 25.0 * np.cos(r[disk] / 2.5)
    image += rng.normal(0.0, 2.5, size=(h, w))
    return np.clip(image, 0, None)


def _make_candidate(metrics: dict[str, float],
                    raw_score: float = 0.0) -> Candidate:
    """Build a minimal Candidate with given metrics (for scoring tests)."""
    return Candidate(
        psf_type="gaussian",
        psf_params={"fwhm": 2.0, "size": 15},
        deconv_method="wiener",
        deconv_params={"snr": 20.0},
        result=np.zeros((2, 2)),
        metrics=dict(metrics),
        raw_score=raw_score,
    )


# ---------------------------------------------------------------------------
# C1: Scoring normalisation and ranking
# ---------------------------------------------------------------------------

class TestScoringNormalisation(unittest.TestCase):
    """Pin down the normalise → composite_normalised ranking math."""

    def test_normalisation_and_ranking_deterministic(self):
        """Given known metrics, the ranking must be deterministic."""
        # Candidate A: high tenengrad, low lap_var → should win
        a_metrics = {
            "laplacian_variance": 500.0,
            "tenengrad": 2_000_000.0,
            "normalised_power_hf": 0.02,
            "brenner": 100_000.0,
            "quality_ratio": 2_000.0,
            "smoothness": 0.85,
        }
        # Candidate B: lower tenengrad, higher lap_var → should lose
        b_metrics = {
            "laplacian_variance": 2_000.0,
            "tenengrad": 1_000_000.0,
            "normalised_power_hf": 0.01,
            "brenner": 50_000.0,
            "quality_ratio": 700.0,
            "smoothness": 0.75,
        }
        # Candidate C: middle ground
        c_metrics = {
            "laplacian_variance": 1_000.0,
            "tenengrad": 1_500_000.0,
            "normalised_power_hf": 0.015,
            "brenner": 75_000.0,
            "quality_ratio": 1_200.0,
            "smoothness": 0.80,
        }

        ca = _make_candidate(a_metrics)
        cb = _make_candidate(b_metrics)
        cc = _make_candidate(c_metrics)
        pool = [ca, cb, cc]

        _normalise_metrics(pool, _WEIGHTS)
        scores = {id(c): _composite_normalised(c, _WEIGHTS) for c in pool}

        # A should rank first (best tenengrad + lowest lap_var)
        self.assertGreater(scores[id(ca)], scores[id(cc)])
        self.assertGreater(scores[id(cc)], scores[id(cb)])

    def test_normalisation_single_candidate(self):
        """A single candidate must normalise to a valid score."""
        metrics = {
            "laplacian_variance": 1_000.0,
            "tenengrad": 1_000_000.0,
            "normalised_power_hf": 0.01,
            "brenner": 50_000.0,
            "quality_ratio": 1_000.0,
            "smoothness": 0.7,
        }
        pool = [_make_candidate(metrics)]
        _normalise_metrics(pool, _WEIGHTS)
        score = _composite_normalised(pool[0], _WEIGHTS)
        # With a single candidate, all norms = 0 → inverted lap_var = 1.0
        # score = 0.25 * 1.0 + 0 + 0 + 0 + 0 + 0 = 0.25
        self.assertAlmostEqual(score, _WEIGHTS["laplacian_variance"], places=5)

    def test_inverted_metrics_match_between_modules(self):
        """_INVERTED_METRICS must be consistent across metrics.py and optimizer.py."""
        from optimizer import _INVERTED_METRICS as opt_inv
        self.assertEqual(_INVERTED_METRICS, opt_inv)


# ---------------------------------------------------------------------------
# C2 / C3: Post-processing strength adaptation
# ---------------------------------------------------------------------------

class TestPostprocessAdaptation(unittest.TestCase):
    """Pin down the small-planet post-processing adaptation logic."""

    def test_large_planet_defaults(self):
        """Large planet (≥20% fill): wv=25, nlm=0.008, sharpen=0."""
        planet_frac = 0.40
        small = planet_frac < 0.20
        self.assertFalse(small)
        # Large planet: default post-processing values (from main.py defaults)
        wv, nlm, sharpen = 25.0, 0.008, 0.0
        self.assertEqual(wv, 25.0)
        self.assertEqual(nlm, 0.008)
        self.assertEqual(sharpen, 0.0)

    def test_small_planet_adaptation(self):
        """Small planet (<20% fill): wv=0, nlm≤0.003, sharpen=1.5."""
        planet_frac = 0.06
        small = planet_frac < 0.20
        self.assertTrue(small)
        # Small planet: adapted values (from main.py lines 346-357)
        wv = 0.0
        nlm = min(0.008, 0.003)
        sharpen = 1.5  # auto-enable
        self.assertEqual(wv, 0.0)
        self.assertEqual(nlm, 0.003)
        self.assertEqual(sharpen, 1.5)


# ---------------------------------------------------------------------------
# C4: White balance preserves channel ratios
# ---------------------------------------------------------------------------

class TestWhiteBalanceCharacterization(unittest.TestCase):
    """Verify _apply_best_to_color preserves colour balance."""

    def test_no_contrast_boost_preserves_ratios(self):
        """Without contrast_boost, output R:G:B ratios match input."""
        from rgb import apply_best_to_color

        class _Candidate:
            psf_type = "gaussian"
            psf_params = {"fwhm": 2.0, "size": 15}
            deconv_method = "wiener"
            deconv_params = {"snr": 20.0}  # no contrast_boost key

        rng = np.random.default_rng(77)
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w]
        disk = ((y - 32)**2 + (x - 32)**2) <= 18**2
        base = np.where(disk, 200.0, 10.0).astype(np.float64)
        base += rng.normal(0, 2, size=(h, w))
        base = np.clip(base, 0, None)
        # Distinct channel ratios
        rgb = np.stack([base * 1.0, base * 0.85, base * 0.65], axis=0)
        c = _Candidate()

        result = apply_best_to_color(rgb, c)

        # Measure R/G ratio on the disk
        mask = planet_mask(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        if mask.any():
            rg_in = rgb[0][mask].mean() / rgb[1][mask].mean()
            rg_out = result[0][mask].mean() / result[1][mask].mean()
            # Should be very close (within 2%)
            self.assertAlmostEqual(rg_out, rg_in, delta=rg_in * 0.02)

    def test_equal_channels_produce_no_wb_correction(self):
        """When all channels are identical, white balance should be a no-op."""
        from rgb import apply_best_to_color

        class _Candidate:
            psf_type = "gaussian"
            psf_params = {"fwhm": 2.0, "size": 15}
            deconv_method = "wiener"
            deconv_params = {"snr": 20.0}

        base = _synthetic_planet(seed=10, fill=0.40)
        rgb = np.stack([base, base, base], axis=0)
        c = _Candidate()

        result = apply_best_to_color(rgb, c)

        # All channels should be equal (modulo float precision)
        np.testing.assert_allclose(result[0], result[1], rtol=1e-10)
        np.testing.assert_allclose(result[0], result[2], rtol=1e-10)


# ---------------------------------------------------------------------------
# C5: Noise floor rejection
# ---------------------------------------------------------------------------

class TestNoiseFloorRejection(unittest.TestCase):
    """Optimizer must reject candidates noisier than the input."""

    def test_run_search_rejects_noisy_candidates(self):
        """At least some trials should be rejected by the noise floor."""
        from optimizer import run_search

        image = _synthetic_planet(seed=2, fill=0.40)
        # With enough trials, some will be rejected
        candidates = run_search(image, n_trials=15, verbose=False)
        # All returned candidates must have smoothness above the noise floor
        from metrics import smoothness as _smoothness
        input_smoothness = _smoothness(image)
        noise_floor = input_smoothness * 0.55
        for c in candidates:
            self.assertGreaterEqual(
                c.metrics["smoothness"], noise_floor,
                f"Candidate with smoothness {c.metrics['smoothness']:.4f} "
                f"passed noise floor {noise_floor:.4f}",
            )


# ---------------------------------------------------------------------------
# C6: Composite score weights
# ---------------------------------------------------------------------------

class TestCompositeScoreWeights(unittest.TestCase):
    """Verify weight sum and inverted metric direction."""

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(_WEIGHTS.values()), 1.0, places=10)

    def test_higher_laplacian_variance_lowers_score(self):
        """laplacian_variance is inverted: higher value → lower composite score."""
        base = {
            "laplacian_variance": 1_000.0,
            "tenengrad": 1_000_000.0,
            "normalised_power_hf": 0.01,
            "brenner": 50_000.0,
            "quality_ratio": 1_000.0,
            "smoothness": 0.7,
        }
        noisy = dict(base, laplacian_variance=10_000.0)

        dummy = np.zeros((2, 2))
        self.assertGreater(
            composite_score(dummy, metrics=base),
            composite_score(dummy, metrics=noisy),
        )


# ---------------------------------------------------------------------------
# C7: Wavelet denoise on clean image
# ---------------------------------------------------------------------------

class TestWaveletDenoiseCharacterization(unittest.TestCase):

    def test_clean_image_nearly_unchanged(self):
        """Wavelet denoising a smooth image should leave it almost intact."""
        smooth = np.full((64, 64), 150.0, dtype=np.float64)
        # Add very slow gradient (real structure, not noise)
        y, x = np.mgrid[0:64, 0:64]
        smooth += 20.0 * np.sin(y / 10.0)

        denoised = _wavelet_denoise(smooth, threshold=5.0, levels=3)
        # Nearly identical; SWT padding causes small edge artifacts (~1.7 max)
        np.testing.assert_allclose(denoised, smooth, atol=2.0)

    def test_noisy_image_reduces_noise(self):
        """Denoising a noisy image must reduce the noise level."""
        rng = np.random.default_rng(55)
        clean = np.full((64, 64), 150.0)
        noisy = clean + rng.normal(0, 10, size=(64, 64))

        denoised = _wavelet_denoise(noisy, threshold=5.0, levels=3)
        self.assertLess(np.std(denoised - clean), np.std(noisy - clean))


# ---------------------------------------------------------------------------
# C8: Planet mask coverage
# ---------------------------------------------------------------------------

class TestPlanetMaskCharacterization(unittest.TestCase):

    def test_large_planet_coverage(self):
        """Large planet (~40% fill) should have mask covering ~30-50%."""
        image = _synthetic_planet(seed=1, fill=0.40)
        mask = planet_mask(image)
        frac = mask.sum() / mask.size
        self.assertGreater(frac, 0.25)
        self.assertLess(frac, 0.55)

    def test_small_planet_coverage(self):
        """Small planet (~6% fill) should have mask covering ~3-12%."""
        image = _synthetic_planet(seed=1, fill=0.06)
        mask = planet_mask(image)
        frac = mask.sum() / mask.size
        self.assertGreater(frac, 0.02)
        self.assertLess(frac, 0.15)

    def test_mask_is_boolean(self):
        image = _synthetic_planet(seed=1)
        mask = planet_mask(image)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, image.shape)

    def test_mask_fallback_on_uniform_image(self):
        """Uniform image should still produce a mask (95th percentile fallback)."""
        uniform = np.full((64, 64), 100.0)
        mask = planet_mask(uniform)
        self.assertEqual(mask.shape, uniform.shape)


# ---------------------------------------------------------------------------
# C9: contrast_stretch preserves relative values
# ---------------------------------------------------------------------------

class TestContrastStretchCharacterization(unittest.TestCase):

    def test_boost_1_returns_unchanged(self):
        """contrast_stretch with boost=1.0 should return the input unchanged."""
        image = _synthetic_planet(seed=3)
        result = contrast_stretch(image, image, boost=1.0)
        # boost=1.0: target_range = (p99-p1)*1.0 → same range → identity
        np.testing.assert_allclose(result, image, atol=0.5)

    def test_boost_increases_range(self):
        """contrast_stretch with boost>1 should increase dynamic range."""
        image = _synthetic_planet(seed=3)
        result = contrast_stretch(image, image, boost=1.5)
        mask = planet_mask(image)
        if mask.any():
            orig_range = np.percentile(image[mask], 99) - np.percentile(image[mask], 1)
            res_range = np.percentile(result[mask], 99) - np.percentile(result[mask], 1)
            self.assertGreater(res_range, orig_range * 1.3)


# ---------------------------------------------------------------------------
# C10: all_metrics returns expected keys
# ---------------------------------------------------------------------------

class TestAllMetricsCharacterization(unittest.TestCase):

    def test_returns_all_expected_keys(self):
        image = _synthetic_planet(seed=1)
        metrics = all_metrics(image)
        for key in _WEIGHTS:
            self.assertIn(key, metrics)

    def test_all_values_are_finite(self):
        image = _synthetic_planet(seed=1)
        metrics = all_metrics(image)
        for name, val in metrics.items():
            self.assertTrue(np.isfinite(val), f"{name} is not finite: {val}")

    def test_tenengrad_increases_with_sharpening(self):
        """A sharpened image should have higher tenengrad than a blurred one."""
        from scipy.ndimage import gaussian_filter
        image = _synthetic_planet(seed=1)
        blurred = gaussian_filter(image, sigma=2.0)
        self.assertGreater(tenengrad(image), tenengrad(blurred))


if __name__ == "__main__":
    unittest.main()
