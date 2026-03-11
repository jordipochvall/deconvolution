"""Unit tests for the planetary deconvolution pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from deconvolve import deconvolve, _wavelet_denoise
from main import load_fits_image, _output_fits_name, parse_args, _apply_best_to_color
from postprocess import postprocess, postprocess_rgb
from psf import gaussian_psf, moffat_psf, build_psf
from metrics import composite_score


# ---------------------------------------------------------------------------
# PSF
# ---------------------------------------------------------------------------

class TestPSF(unittest.TestCase):
    def test_psf_normalised_and_odd_size(self):
        g = gaussian_psf(fwhm=2.0, size=20)
        m = moffat_psf(fwhm=2.0, beta=3.0, size=30)
        self.assertEqual(g.shape, (21, 21))
        self.assertEqual(m.shape, (31, 31))
        self.assertAlmostEqual(float(g.sum()), 1.0, places=10)
        self.assertAlmostEqual(float(m.sum()), 1.0, places=10)

    def test_build_psf_dispatch(self):
        g = build_psf("gaussian", {"fwhm": 2.0, "size": 15})
        m = build_psf("moffat", {"fwhm": 2.0, "beta": 3.0, "size": 15})
        self.assertEqual(g.shape, (15, 15))
        self.assertEqual(m.shape, (15, 15))
        with self.assertRaises(ValueError):
            build_psf("invalid_type", {})


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------

class TestDeconvolution(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.image = np.clip(rng.normal(loc=200, scale=15, size=(64, 64)), 0, None)
        self.psf = gaussian_psf(fwhm=2.5, size=15)

    def test_wiener_preserves_shape_and_non_negative(self):
        out = deconvolve("wiener", self.image, self.psf, {"snr": 15.0})
        self.assertEqual(out.shape, self.image.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_tikhonov_preserves_shape_and_non_negative(self):
        out = deconvolve("tikhonov", self.image, self.psf, {"regularization": 1e-2})
        self.assertEqual(out.shape, self.image.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_richardson_lucy_preserves_shape_and_non_negative(self):
        out = deconvolve("richardson_lucy", self.image, self.psf, {
            "iterations": 10, "damping": 1e-3, "tv_lambda": 0.1,
            "deringing": 0.5, "contrast_boost": 1.0,
        })
        self.assertEqual(out.shape, self.image.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            deconvolve("fake_method", self.image, self.psf, {})

    def test_wiener_accepts_brightness_mask(self):
        mask = np.ones_like(self.image, dtype=bool)
        out = deconvolve("wiener", self.image, self.psf, {"snr": 20.0, "brightness_mask": mask})
        self.assertEqual(out.shape, self.image.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_rl_accepts_precomputed_masks(self):
        dering_mask = np.ones_like(self.image, dtype=np.float64)
        limb_mask = np.zeros_like(self.image, dtype=np.float64)
        out = deconvolve(
            "richardson_lucy",
            self.image,
            self.psf,
            {
                "iterations": 6,
                "damping": 1e-3,
                "tv_lambda": 0.05,
                "deringing": 0.5,
                "contrast_boost": 1.0,
                "dering_mask": dering_mask,
                "limb_mask": limb_mask,
            },
        )
        self.assertEqual(out.shape, self.image.shape)


# ---------------------------------------------------------------------------
# Wavelet denoising
# ---------------------------------------------------------------------------

class TestWaveletDenoise(unittest.TestCase):
    def test_wavelet_denoise_reduces_noise(self):
        rng = np.random.default_rng(42)
        clean = np.ones((64, 64)) * 100.0
        noisy = clean + rng.normal(0, 5, size=(64, 64))
        denoised = _wavelet_denoise(noisy, threshold=5.0, levels=3)
        self.assertEqual(denoised.shape, noisy.shape)
        self.assertLess(np.std(denoised - clean), np.std(noisy - clean))

    def test_wavelet_denoise_small_image(self):
        small = np.ones((4, 4)) * 100.0
        result = _wavelet_denoise(small, threshold=1.0, levels=4)
        self.assertEqual(result.shape, (4, 4))


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

class TestPostprocess(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        self.image = np.clip(rng.normal(loc=500, scale=20, size=(64, 64)), 0, None)

    def test_postprocess_returns_same_shape(self):
        out = postprocess(self.image, wavelet_threshold=10.0, nlm_h=0.0)
        self.assertEqual(out.shape, self.image.shape)

    def test_postprocess_no_ops(self):
        out = postprocess(self.image, wavelet_threshold=0.0, nlm_h=0.0)
        np.testing.assert_array_equal(out, self.image)

    def test_postprocess_rgb_returns_same_shape(self):
        rgb = np.stack([self.image, self.image * 0.9, self.image * 0.8], axis=0)
        out = postprocess_rgb(rgb, wavelet_threshold=5.0, nlm_h=0.0)
        self.assertEqual(out.shape, rgb.shape)

    def test_postprocess_rgb_parallel_matches_serial(self):
        rgb = np.stack([self.image, self.image * 0.9, self.image * 0.8], axis=0)
        out_serial = postprocess_rgb(rgb, wavelet_threshold=5.0, nlm_h=0.0, jobs=1)
        out_parallel = postprocess_rgb(rgb, wavelet_threshold=5.0, nlm_h=0.0, jobs=3)
        np.testing.assert_allclose(out_parallel, out_serial, rtol=0, atol=1e-10)


class TestMetricsScoring(unittest.TestCase):
    def test_composite_score_penalises_higher_laplacian_variance(self):
        base_metrics = {
            "laplacian_variance": 1_000.0,
            "tenengrad": 1_000_000.0,
            "normalised_power_hf": 0.01,
            "brenner": 50_000.0,
            "quality_ratio": 1_000.0,
            "smoothness": 0.7,
        }
        noisy_metrics = dict(base_metrics)
        noisy_metrics["laplacian_variance"] = 10_000.0

        score_base = composite_score(np.zeros((1, 1), dtype=np.float64), metrics=base_metrics)
        score_noisy = composite_score(np.zeros((1, 1), dtype=np.float64), metrics=noisy_metrics)

        self.assertGreater(score_base, score_noisy)


class TestRGBDeconvolution(unittest.TestCase):
    def test_apply_best_to_color_parallel_matches_serial(self):
        class _Candidate:
            psf_type = "gaussian"
            psf_params = {"fwhm": 2.0, "size": 15}
            deconv_method = "wiener"
            deconv_params = {"snr": 20.0}

        rng = np.random.default_rng(123)
        base = np.clip(rng.normal(loc=250, scale=10, size=(64, 64)), 0, None)
        rgb = np.stack([base, base * 0.95, base * 1.05], axis=0)
        c = _Candidate()

        out_serial = _apply_best_to_color(rgb, c, rgb_jobs=1)
        out_parallel = _apply_best_to_color(rgb, c, rgb_jobs=3)
        self.assertEqual(out_parallel.shape, rgb.shape)
        np.testing.assert_allclose(out_parallel, out_serial, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# FITS loading
# ---------------------------------------------------------------------------

class TestLoadFits(unittest.TestCase):
    def test_load_fits_2d(self):
        data = np.ones((32, 32), dtype=np.float32) * 100
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mono.fits"
            fits.writeto(path, data, overwrite=True)
            loaded, is_color = load_fits_image(str(path))
            self.assertFalse(is_color)
            self.assertEqual(loaded.shape, (32, 32))

    def test_load_fits_rgb(self):
        cube = np.ones((3, 16, 16), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rgb.fits"
            fits.writeto(path, cube, overwrite=True)
            data, is_color = load_fits_image(str(path))
            self.assertTrue(is_color)
            self.assertEqual(data.shape, (3, 16, 16))

    def test_load_fits_channel_selection(self):
        cube = np.ones((3, 8, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cube.fits"
            fits.writeto(path, cube, overwrite=True)

            data, is_color = load_fits_image(str(path), channel=2)
            self.assertFalse(is_color)
            self.assertEqual(data.shape, (8, 8))

            with self.assertRaises(ValueError):
                load_fits_image(str(path), channel=5)

    def test_load_fits_nan_replaced(self):
        data = np.ones((16, 16), dtype=np.float32) * 100
        data[5, 5] = np.nan
        data[10, 10] = np.inf
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.fits"
            fits.writeto(path, data, overwrite=True)
            loaded, _ = load_fits_image(str(path))
            self.assertTrue(np.all(np.isfinite(loaded)))


class TestOutputNaming(unittest.TestCase):
    def test_output_fits_name_uses_dc_suffix(self):
        self.assertEqual(_output_fits_name("jupiter", ".fits", rank=1), "jupiter_DC.fits")
        self.assertEqual(_output_fits_name("jupiter", ".fits", rank=2), "jupiter_DC_r2.fits")


class TestCLI(unittest.TestCase):
    def test_parse_args_default_top_and_plots(self):
        with patch("sys.argv", ["main.py", "images"]):
            args = parse_args()
        self.assertEqual(args.top, 1)
        self.assertFalse(args.save_plots)
        self.assertFalse(args.quiet)
        self.assertEqual(args.file_jobs, 1)
        self.assertEqual(args.rgb_jobs, 1)

    def test_parse_args_custom_top_and_save_plots(self):
        with patch("sys.argv", ["main.py", "images", "--top", "4", "--save-plots"]):
            args = parse_args()
        self.assertEqual(args.top, 4)
        self.assertTrue(args.save_plots)

    def test_parse_args_quiet(self):
        with patch("sys.argv", ["main.py", "images", "--quiet"]):
            args = parse_args()
        self.assertTrue(args.quiet)

    def test_parse_args_parallel_flags(self):
        with patch("sys.argv", ["main.py", "images", "--file-jobs", "3", "--rgb-jobs", "2"]):
            args = parse_args()
        self.assertEqual(args.file_jobs, 3)
        self.assertEqual(args.rgb_jobs, 2)


# ---------------------------------------------------------------------------
# Optimizer integration
# ---------------------------------------------------------------------------

class TestOptimizer(unittest.TestCase):
    @staticmethod
    def _synthetic_planet(seed: int = 1) -> np.ndarray:
        """
        Build a stable synthetic planetary target for optimizer tests.

        This avoids flaky runs where all random-noise trials are rejected by
        the smoothness floor.
        """
        rng = np.random.default_rng(seed)
        h, w = 64, 64
        y, x = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        image = np.full((h, w), 20.0, dtype=np.float64)
        disk = r <= 18
        image[disk] = 220.0
        # Add smooth radial texture inside the disk
        image[disk] += 25.0 * np.cos(r[disk] / 2.5)
        # Small gaussian noise everywhere
        image += rng.normal(0.0, 2.5, size=(h, w))
        return np.clip(image, 0, None)

    def test_run_search_returns_candidates(self):
        from optimizer import run_search
        image = self._synthetic_planet(seed=1)
        candidates = run_search(image, n_trials=6)
        self.assertGreater(len(candidates), 0)
        scores = [c.normalised_score for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_run_search_progress_callback(self):
        from optimizer import run_search
        image = self._synthetic_planet(seed=3)
        steps = []

        def cb(done, total):
            steps.append((done, total))

        run_search(image, n_trials=6, progress_callback=cb)
        self.assertEqual(len(steps), 6)
        self.assertEqual(steps[-1], (6, 6))

    def test_run_search_uses_candidate_metrics_mask(self):
        from optimizer import run_search
        from metrics import all_metrics as real_all_metrics

        image = self._synthetic_planet(seed=5)
        seen_masks = []

        def _wrapped_all_metrics(*args, **kwargs):
            seen_masks.append(kwargs.get("mask"))
            return real_all_metrics(*args, **kwargs)

        with patch("optimizer.all_metrics", side_effect=_wrapped_all_metrics):
            run_search(image, n_trials=6, verbose=False)

        self.assertGreater(len(seen_masks), 0)
        self.assertTrue(all(mask is None for mask in seen_masks))


if __name__ == "__main__":
    unittest.main()
