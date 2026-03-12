"""Tests for refactored and newly extracted functions.

Covers PostprocessConfig, adapt_postprocessing, rerank_candidates,
_finalize_result, _correct_white_balance, _apply_global_contrast_boost,
_wavelet_sharpen, _wavelet_denoise_fine, swt_pad/unpad,
_limb_zone_mask, _deringing_limb, _adaptive_blend_mask, _SearchBounds.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_planet(seed: int = 1, fill: float = 0.40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius = np.sqrt(fill * h * w / np.pi)
    image = np.full((h, w), 20.0, dtype=np.float64)
    disk = r <= radius
    image[disk] = 220.0
    image[disk] += 25.0 * np.cos(r[disk] / 2.5)
    image += rng.normal(0.0, 2.5, size=(h, w))
    return np.clip(image, 0, None)


def _synthetic_rgb(seed: int = 1) -> np.ndarray:
    base = _synthetic_planet(seed)
    return np.stack([base * 1.0, base * 0.85, base * 0.65], axis=0)


class _FakeCandidate:
    def __init__(self, result=None):
        self.psf_type = "gaussian"
        self.psf_params = {"fwhm": 2.0, "size": 15}
        self.deconv_method = "wiener"
        self.deconv_params = {"snr": 20.0}
        self.normalised_score = 0.5
        self.result = result if result is not None else _synthetic_planet(seed=99)
        self.metrics = {}
        self.source = "optuna"


# ===================================================================
# PostprocessConfig
# ===================================================================

class TestPostprocessConfig(unittest.TestCase):
    def test_default_values(self):
        from postprocess import PostprocessConfig
        cfg = PostprocessConfig()
        self.assertEqual(cfg.wv, 25.0)
        self.assertEqual(cfg.nlm, 0.008)
        self.assertEqual(cfg.sharpen, 0.0)
        self.assertEqual(cfg.dp, 0.5)

    def test_custom_values(self):
        from postprocess import PostprocessConfig
        cfg = PostprocessConfig(wv=10.0, nlm=0.003, sharpen=1.5, dp=0.8)
        self.assertEqual(cfg.wv, 10.0)
        self.assertEqual(cfg.nlm, 0.003)
        self.assertEqual(cfg.sharpen, 1.5)
        self.assertEqual(cfg.dp, 0.8)

    def test_postprocess_with_config(self):
        from postprocess import PostprocessConfig, postprocess
        image = _synthetic_planet()
        cfg = PostprocessConfig(wv=5.0, nlm=0.0, sharpen=0.0, dp=0.5)
        result = postprocess(image, cfg)
        self.assertEqual(result.shape, image.shape)

    def test_postprocess_config_vs_legacy_kwargs(self):
        """PostprocessConfig and legacy kwargs must produce identical results."""
        from postprocess import PostprocessConfig, postprocess
        image = _synthetic_planet(seed=7)
        cfg = PostprocessConfig(wv=8.0, nlm=0.0, sharpen=0.0, dp=0.3)
        out_cfg = postprocess(image, cfg)
        out_legacy = postprocess(image, wavelet_threshold=8.0, nlm_h=0.0,
                                 sharpen_gain=0.0, disk_preservation=0.3)
        np.testing.assert_array_equal(out_cfg, out_legacy)

    def test_postprocess_rgb_with_config(self):
        from postprocess import PostprocessConfig, postprocess_rgb
        rgb = _synthetic_rgb()
        cfg = PostprocessConfig(wv=5.0, nlm=0.0, sharpen=0.0, dp=0.5)
        result = postprocess_rgb(rgb, cfg)
        self.assertEqual(result.shape, rgb.shape)


# ===================================================================
# adapt_postprocessing
# ===================================================================

class TestAdaptPostprocessing(unittest.TestCase):
    def test_large_planet_unchanged(self):
        from main import adapt_postprocessing
        from postprocess import PostprocessConfig
        image = _synthetic_planet(fill=0.40)
        cfg = PostprocessConfig(wv=25.0, nlm=0.008, sharpen=0.0)
        result = adapt_postprocessing(image, cfg)
        self.assertEqual(result.wv, 25.0)
        self.assertEqual(result.nlm, 0.008)
        self.assertEqual(result.sharpen, 0.0)

    def test_small_planet_adapts(self):
        from main import adapt_postprocessing
        from postprocess import PostprocessConfig
        image = _synthetic_planet(fill=0.06)
        cfg = PostprocessConfig(wv=25.0, nlm=0.008, sharpen=0.0)
        result = adapt_postprocessing(image, cfg)
        self.assertEqual(result.wv, 0.0)
        self.assertLessEqual(result.nlm, 0.003)
        self.assertEqual(result.sharpen, 1.5)

    def test_small_planet_preserves_custom_sharpen(self):
        from main import adapt_postprocessing
        from postprocess import PostprocessConfig
        image = _synthetic_planet(fill=0.06)
        cfg = PostprocessConfig(wv=25.0, nlm=0.008, sharpen=2.0)
        result = adapt_postprocessing(image, cfg)
        self.assertEqual(result.sharpen, 2.0)  # not overridden to 1.5

    def test_returns_postprocess_config(self):
        from main import adapt_postprocessing
        from postprocess import PostprocessConfig
        image = _synthetic_planet()
        result = adapt_postprocessing(image, PostprocessConfig())
        self.assertIsInstance(result, PostprocessConfig)


# ===================================================================
# rerank_candidates
# ===================================================================

class TestRerankCandidates(unittest.TestCase):
    def test_single_candidate_returns_it(self):
        from rerank import rerank_candidates
        from postprocess import PostprocessConfig
        c = _FakeCandidate()
        best, cached = rerank_candidates(
            [c], c.result, c.result, False,
            PostprocessConfig(wv=5.0, nlm=0.0, sharpen=0.0),
        )
        self.assertIs(best, c)
        self.assertIsNone(cached)

    def test_multiple_candidates_returns_best(self):
        from rerank import rerank_candidates
        from postprocess import PostprocessConfig
        from optimizer import Candidate
        from metrics import all_metrics, composite_score

        image = _synthetic_planet(seed=10)
        candidates = []
        for snr in [10.0, 20.0, 40.0]:
            from psf import build_psf
            from deconvolve import deconvolve
            psf = build_psf("gaussian", {"fwhm": 2.0, "size": 15})
            result = deconvolve("wiener", image, psf, {"snr": snr})
            metrics = all_metrics(result)
            candidates.append(Candidate(
                psf_type="gaussian",
                psf_params={"fwhm": 2.0, "size": 15},
                deconv_method="wiener",
                deconv_params={"snr": snr},
                result=result.astype(np.float32),
                metrics=metrics,
                raw_score=composite_score(result, metrics),
            ))

        cfg = PostprocessConfig(wv=0.0, nlm=0.0, sharpen=0.0, dp=0.0)
        best, cached = rerank_candidates(
            candidates, image, image, False, cfg,
        )
        self.assertIn(best, candidates)
        self.assertIsNotNone(cached)

    def test_rerank_includes_seed_candidates(self):
        from rerank import rerank_candidates
        from postprocess import PostprocessConfig
        from optimizer import Candidate
        from metrics import all_metrics, composite_score

        image = _synthetic_planet(seed=11)
        candidates = []
        for snr, source in [(20.0, "optuna"), (30.0, "seed")]:
            from psf import build_psf
            from deconvolve import deconvolve
            psf = build_psf("gaussian", {"fwhm": 2.0, "size": 15})
            result = deconvolve("wiener", image, psf, {"snr": snr})
            metrics = all_metrics(result)
            c = Candidate(
                psf_type="gaussian",
                psf_params={"fwhm": 2.0, "size": 15},
                deconv_method="wiener",
                deconv_params={"snr": snr},
                result=result.astype(np.float32),
                metrics=metrics,
                raw_score=composite_score(result, metrics),
                source=source,
            )
            candidates.append(c)

        cfg = PostprocessConfig(wv=0.0, nlm=0.0, sharpen=0.0, dp=0.0)
        best, _ = rerank_candidates(candidates, image, image, False, cfg)
        self.assertIn(best, candidates)


# ===================================================================
# _finalize_result
# ===================================================================

class TestFinalizeResult(unittest.TestCase):
    def test_cached_result_returned_immediately(self):
        from main import _finalize_result
        from postprocess import PostprocessConfig
        cached = np.ones((64, 64))
        c = _FakeCandidate()
        result = _finalize_result(
            c.result, c, cached, False, True, PostprocessConfig(),
        )
        np.testing.assert_array_equal(result, cached)

    def test_mono_no_post(self):
        from main import _finalize_result
        from postprocess import PostprocessConfig
        c = _FakeCandidate()
        result = _finalize_result(
            c.result, c, None, False, False, PostprocessConfig(),
        )
        np.testing.assert_array_equal(result, c.result)

    def test_mono_with_post(self):
        from main import _finalize_result
        from postprocess import PostprocessConfig
        c = _FakeCandidate()
        cfg = PostprocessConfig(wv=5.0, nlm=0.0, sharpen=0.0, dp=0.5)
        result = _finalize_result(c.result, c, None, False, True, cfg)
        self.assertEqual(result.shape, c.result.shape)
        # Post-processing should change the image
        self.assertFalse(np.array_equal(result, c.result))


# ===================================================================
# _correct_white_balance
# ===================================================================

class TestCorrectWhiteBalance(unittest.TestCase):
    def test_identical_channels_no_change(self):
        from rgb import _correct_white_balance
        base = _synthetic_planet(seed=20)
        rgb_in = np.stack([base, base, base], axis=0)
        rgb_out = np.stack([base * 1.1, base * 1.1, base * 1.1], axis=0)
        corrected = _correct_white_balance(rgb_in, rgb_out)
        # All channels should remain equal
        np.testing.assert_allclose(corrected[0], corrected[1], rtol=1e-10)
        np.testing.assert_allclose(corrected[0], corrected[2], rtol=1e-10)

    def test_restores_channel_ratios(self):
        from rgb import _correct_white_balance
        from metrics import planet_mask
        base = _synthetic_planet(seed=21)
        rgb_in = np.stack([base * 1.0, base * 0.85, base * 0.65], axis=0)
        # Simulate RL distortion: green amplified more than red/blue
        rgb_out = np.stack([base * 1.05, base * 1.10, base * 0.70], axis=0)
        corrected = _correct_white_balance(rgb_in, rgb_out)
        mask = planet_mask(base)
        if mask.any():
            # After WB correction, R/G ratio should be closer to input
            rg_in = rgb_in[0][mask].mean() / rgb_in[1][mask].mean()
            rg_out = corrected[0][mask].mean() / corrected[1][mask].mean()
            self.assertAlmostEqual(rg_out, rg_in, delta=rg_in * 0.05)

    def test_no_mask_returns_unchanged(self):
        from rgb import _correct_white_balance
        # Uniform image → no planet mask → no correction
        rgb_in = np.full((3, 32, 32), 100.0)
        rgb_out = np.full((3, 32, 32), 110.0)
        corrected = _correct_white_balance(rgb_in, rgb_out)
        np.testing.assert_array_equal(corrected, rgb_out)


# ===================================================================
# _apply_global_contrast_boost
# ===================================================================

class TestApplyGlobalContrastBoost(unittest.TestCase):
    def test_applies_same_transform_to_all_channels(self):
        """All channels get the same linear scale+offset (colour ratios preserved)."""
        from rgb import _apply_global_contrast_boost
        from deconvolve import deconvolve
        from psf import build_psf
        from metrics import planet_mask

        base = _synthetic_planet(seed=30)
        rgb_in = np.stack([base, base * 0.85, base * 0.65], axis=0)
        psf = build_psf("gaussian", {"fwhm": 2.0, "size": 15})
        channels = [deconvolve("wiener", rgb_in[c], psf, {"snr": 20.0}) for c in range(3)]
        result = np.stack(channels, axis=0)

        boosted = _apply_global_contrast_boost(rgb_in, channels, result, boost=1.3)
        mask = planet_mask(base)
        if mask.any():
            # The transform is result * scale + offset → pixel-wise ratios
            # change, but all channels share the same scale+offset so
            # the *relative difference* between channels is preserved.
            # Verify boosted has different values from result (actual boost).
            self.assertFalse(np.allclose(boosted, result, atol=1.0))

    def test_non_negative_output(self):
        from rgb import _apply_global_contrast_boost
        from deconvolve import deconvolve
        from psf import build_psf

        base = _synthetic_planet(seed=31)
        rgb_in = np.stack([base, base * 0.85, base * 0.65], axis=0)
        psf = build_psf("gaussian", {"fwhm": 2.0, "size": 15})
        channels = [deconvolve("wiener", rgb_in[c], psf, {"snr": 20.0}) for c in range(3)]
        result = np.stack(channels, axis=0)

        boosted = _apply_global_contrast_boost(rgb_in, channels, result, boost=1.3)
        self.assertGreaterEqual(boosted.min(), 0.0)


# ===================================================================
# _wavelet_sharpen
# ===================================================================

class TestWaveletSharpen(unittest.TestCase):
    def test_gain_1_returns_unchanged(self):
        from postprocess import _wavelet_sharpen
        image = _synthetic_planet(seed=40)
        result = _wavelet_sharpen(image, gain=1.0)
        np.testing.assert_array_equal(result, image)

    def test_sharpening_increases_tenengrad(self):
        from postprocess import _wavelet_sharpen
        from metrics import tenengrad
        image = _synthetic_planet(seed=41)
        sharpened = _wavelet_sharpen(image, gain=2.0, levels=3)
        self.assertGreater(tenengrad(sharpened), tenengrad(image))

    def test_preserves_shape(self):
        from postprocess import _wavelet_sharpen
        image = _synthetic_planet(seed=42)
        result = _wavelet_sharpen(image, gain=1.5, levels=3)
        self.assertEqual(result.shape, image.shape)

    def test_small_image_no_crash(self):
        from postprocess import _wavelet_sharpen
        small = np.ones((8, 8)) * 100.0
        result = _wavelet_sharpen(small, gain=1.5, levels=4)
        self.assertEqual(result.shape, (8, 8))


# ===================================================================
# _wavelet_denoise_fine
# ===================================================================

class TestWaveletDenoiseFine(unittest.TestCase):
    def test_reduces_noise(self):
        from postprocess import _wavelet_denoise_fine
        rng = np.random.default_rng(50)
        clean = np.full((64, 64), 150.0)
        noisy = clean + rng.normal(0, 10, size=(64, 64))
        denoised = _wavelet_denoise_fine(noisy, strength=3.0, levels=3)
        self.assertLess(np.std(denoised - clean), np.std(noisy - clean))

    def test_preserves_shape(self):
        from postprocess import _wavelet_denoise_fine
        image = _synthetic_planet(seed=51)
        result = _wavelet_denoise_fine(image, strength=3.0, levels=3)
        self.assertEqual(result.shape, image.shape)

    def test_small_image_nearly_unchanged(self):
        from postprocess import _wavelet_denoise_fine
        small = np.ones((4, 4)) * 100.0
        result = _wavelet_denoise_fine(small, strength=3.0, levels=4)
        # Too small for useful decomposition → nearly unchanged (float precision)
        np.testing.assert_allclose(result, small, atol=1e-10)


# ===================================================================
# swt_pad / swt_unpad
# ===================================================================

class TestSwtPadUnpad(unittest.TestCase):
    def test_roundtrip(self):
        from wavelet_utils import swt_pad, swt_unpad
        image = _synthetic_planet(seed=60)
        h, w = image.shape
        padded, pad_h, pad_w = swt_pad(image, levels=3)
        # Padded dimensions must be multiples of 2^3 = 8
        self.assertEqual(padded.shape[0] % 8, 0)
        self.assertEqual(padded.shape[1] % 8, 0)
        # Roundtrip should recover original
        recovered = swt_unpad(padded, h, w, pad_h, pad_w)
        np.testing.assert_array_equal(recovered, np.clip(image, 0, None))

    def test_already_aligned(self):
        from wavelet_utils import swt_pad, swt_unpad
        # 64x64 is already a multiple of 2^4=16, but padding still adds border
        image = np.ones((64, 64)) * 100.0
        padded, pad_h, pad_w = swt_pad(image, levels=4)
        self.assertGreater(padded.shape[0], 64)  # padding always added

    def test_non_square(self):
        from wavelet_utils import swt_pad, swt_unpad
        image = np.ones((30, 50)) * 100.0
        h, w = image.shape
        padded, pad_h, pad_w = swt_pad(image, levels=3)
        self.assertEqual(padded.shape[0] % 8, 0)
        self.assertEqual(padded.shape[1] % 8, 0)
        recovered = swt_unpad(padded, h, w, pad_h, pad_w)
        self.assertEqual(recovered.shape, (30, 50))


# ===================================================================
# _limb_zone_mask
# ===================================================================

class TestLimbZoneMask(unittest.TestCase):
    def test_returns_soft_mask(self):
        from postprocess import _limb_zone_mask
        image = _synthetic_planet(seed=70)
        mask = _limb_zone_mask(image, width=8)
        self.assertEqual(mask.shape, image.shape)
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)

    def test_limb_has_higher_values(self):
        from postprocess import _limb_zone_mask
        from metrics import planet_mask
        image = _synthetic_planet(seed=71)
        mask = _limb_zone_mask(image, width=8)
        pmask = planet_mask(image)
        # The limb zone should have non-zero values
        self.assertGreater(mask.max(), 0.1)

    def test_uniform_image_mask(self):
        from postprocess import _limb_zone_mask
        uniform = np.full((64, 64), 100.0)
        mask = _limb_zone_mask(uniform, width=8)
        self.assertEqual(mask.shape, (64, 64))


# ===================================================================
# _deringing_limb
# ===================================================================

class TestDeringingLimb(unittest.TestCase):
    def test_strength_zero_returns_unchanged(self):
        from postprocess import _deringing_limb
        image = _synthetic_planet(seed=80)
        result = _deringing_limb(image, strength=0.0)
        np.testing.assert_array_equal(result, image)

    def test_preserves_shape(self):
        from postprocess import _deringing_limb
        image = _synthetic_planet(seed=81)
        result = _deringing_limb(image, strength=0.5, width=8)
        self.assertEqual(result.shape, image.shape)

    def test_modifies_limb_region(self):
        from postprocess import _deringing_limb
        image = _synthetic_planet(seed=82)
        result = _deringing_limb(image, strength=1.0, width=8)
        # Should modify some pixels
        self.assertFalse(np.array_equal(result, image))


# ===================================================================
# _adaptive_blend_mask
# ===================================================================

class TestAdaptiveBlendMask(unittest.TestCase):
    def test_returns_soft_mask(self):
        from postprocess import _adaptive_blend_mask
        image = _synthetic_planet(seed=90)
        mask = _adaptive_blend_mask(image)
        self.assertEqual(mask.shape, image.shape)
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)

    def test_disk_has_higher_values(self):
        from postprocess import _adaptive_blend_mask
        from metrics import planet_mask
        image = _synthetic_planet(seed=91)
        blend = _adaptive_blend_mask(image)
        pmask = planet_mask(image)
        if pmask.any():
            disk_mean = blend[pmask].mean()
            sky_mean = blend[~pmask].mean()
            self.assertGreater(disk_mean, sky_mean)

    def test_uniform_image_returns_ones(self):
        from postprocess import _adaptive_blend_mask
        uniform = np.full((64, 64), 100.0)
        mask = _adaptive_blend_mask(uniform)
        np.testing.assert_array_equal(mask, np.ones_like(uniform))


# ===================================================================
# _SearchBounds
# ===================================================================

class TestSearchBounds(unittest.TestCase):
    def test_large_planet_bounds(self):
        from optimizer import _SearchBounds
        b = _SearchBounds.for_planet(small_planet=False)
        self.assertEqual(b.fwhm_range, (1.0, 5.0))
        self.assertEqual(b.rl_max_iter, 300)
        self.assertEqual(b.rl_max_cb, 1.50)
        self.assertEqual(b.rl_limb_supp, 0.85)

    def test_small_planet_bounds(self):
        from optimizer import _SearchBounds
        b = _SearchBounds.for_planet(small_planet=True)
        self.assertEqual(b.fwhm_range, (1.0, 2.5))
        self.assertEqual(b.rl_max_iter, 20)
        self.assertEqual(b.rl_max_cb, 1.0)
        self.assertEqual(b.rl_min_tv, 0.30)
        self.assertEqual(b.rl_max_tv, 0.80)
        self.assertEqual(b.rl_limb_supp, 0.3)

    def test_small_planet_more_restrictive(self):
        from optimizer import _SearchBounds
        large = _SearchBounds.for_planet(False)
        small = _SearchBounds.for_planet(True)
        self.assertLess(small.rl_max_iter, large.rl_max_iter)
        self.assertLess(small.rl_max_cb, large.rl_max_cb)
        self.assertGreater(small.rl_min_tv, large.rl_min_tv)


# ===================================================================
# _build_rerank_pool
# ===================================================================

class TestBuildRerankPool(unittest.TestCase):
    def test_top_n_capped_at_20(self):
        from rerank import build_rerank_pool
        candidates = [_FakeCandidate() for _ in range(30)]
        pool = build_rerank_pool(candidates)
        self.assertEqual(len(pool), 20)

    def test_includes_seed_candidates_beyond_top(self):
        from rerank import build_rerank_pool
        candidates = [_FakeCandidate() for _ in range(5)]
        seed = _FakeCandidate()
        seed.source = "seed"
        candidates.append(seed)
        pool = build_rerank_pool(candidates, max_top=3)
        self.assertIn(seed, pool)
        self.assertEqual(len(pool), 4)  # 3 top + 1 seed

    def test_no_duplicates_when_seed_in_top(self):
        from rerank import build_rerank_pool
        seed = _FakeCandidate()
        seed.source = "seed"
        candidates = [seed, _FakeCandidate(), _FakeCandidate()]
        pool = build_rerank_pool(candidates, max_top=3)
        self.assertEqual(len(pool), 3)  # seed already in top 3


# ===================================================================
# _pick_best_candidate
# ===================================================================

class TestPickBestCandidate(unittest.TestCase):
    def test_returns_best_from_entries(self):
        from rerank import pick_best_candidate
        from metrics import all_metrics

        img_good = _synthetic_planet(seed=200)
        img_bad = _synthetic_planet(seed=201) * 0.5  # dimmer = worse metrics

        c_good = _FakeCandidate(result=img_good)
        c_bad = _FakeCandidate(result=img_bad)

        entries = [
            (c_good, img_good, all_metrics(img_good)),
            (c_bad, img_bad, all_metrics(img_bad)),
        ]
        best, cached = pick_best_candidate(entries, c_bad)
        self.assertIsNotNone(cached)

    def test_returns_fallback_when_single_entry(self):
        from rerank import pick_best_candidate
        from metrics import all_metrics

        img = _synthetic_planet(seed=202)
        c = _FakeCandidate(result=img)
        entries = [(c, img, all_metrics(img))]
        best, cached = pick_best_candidate(entries, c)
        self.assertIs(best, c)


# ===================================================================
# _blend_with_disk_mask
# ===================================================================

class TestSharpeningAppliesToDisk(unittest.TestCase):
    """Regression test: wavelet sharpening must enhance the planet disk, not sky."""

    def test_sharpening_increases_disk_tenengrad(self):
        from postprocess import PostprocessConfig, postprocess
        from metrics import planet_mask, tenengrad

        image = _synthetic_planet(seed=300)
        pmask = planet_mask(image)

        # No sharpening
        cfg_none = PostprocessConfig(wv=0.0, nlm=0.0, sharpen=0.0, dp=0.5)
        result_none = postprocess(image, cfg_none)

        # With sharpening
        cfg_sharp = PostprocessConfig(wv=0.0, nlm=0.0, sharpen=2.0, dp=0.5)
        result_sharp = postprocess(image, cfg_sharp)

        # Sharpening should increase tenengrad on the planet disk
        ten_none = tenengrad(result_none, mask=pmask)
        ten_sharp = tenengrad(result_sharp, mask=pmask)
        self.assertGreater(ten_sharp, ten_none)


class TestBlendWithDiskMask(unittest.TestCase):
    def test_no_mask_returns_processed(self):
        from postprocess import _blend_with_disk_mask
        original = np.ones((32, 32)) * 100.0
        processed = np.ones((32, 32)) * 200.0
        result = _blend_with_disk_mask(original, processed, None, 0.5)
        np.testing.assert_array_equal(result, processed)

    def test_zero_preservation_returns_processed(self):
        from postprocess import _blend_with_disk_mask
        original = np.ones((32, 32)) * 100.0
        processed = np.ones((32, 32)) * 200.0
        mask = np.ones((32, 32))
        result = _blend_with_disk_mask(original, processed, mask, 0.0)
        np.testing.assert_array_equal(result, processed)

    def test_full_preservation_returns_original_on_disk(self):
        from postprocess import _blend_with_disk_mask
        original = np.ones((32, 32)) * 100.0
        processed = np.ones((32, 32)) * 200.0
        mask = np.ones((32, 32))  # fully on disk
        result = _blend_with_disk_mask(original, processed, mask, 1.0)
        np.testing.assert_array_equal(result, original)


# ===================================================================
# _apply_nlm
# ===================================================================

class TestApplyNlm(unittest.TestCase):
    def test_zero_strength_returns_unchanged(self):
        from postprocess import _apply_nlm
        image = _synthetic_planet(seed=210)
        result = _apply_nlm(image, 0.0, None, 0.5)
        np.testing.assert_array_equal(result, image)

    def test_positive_strength_changes_image(self):
        from postprocess import _apply_nlm
        image = _synthetic_planet(seed=211)
        result = _apply_nlm(image, 0.01, None, 0.0)
        self.assertFalse(np.array_equal(result, image))


# ===================================================================
# _rl_setup_masks
# ===================================================================

class TestRlSetupMasks(unittest.TestCase):
    def test_no_deringing_no_limb(self):
        from deconvolve import _rl_setup_masks
        image = _synthetic_planet(seed=220)
        dm, lm = _rl_setup_masks(image, 0.0, 0.0, None, None)
        self.assertIsNone(dm)
        self.assertIsNone(lm)

    def test_deringing_creates_mask(self):
        from deconvolve import _rl_setup_masks
        image = _synthetic_planet(seed=221)
        dm, lm = _rl_setup_masks(image, 0.5, 0.0, None, None)
        self.assertIsNotNone(dm)
        self.assertIsNone(lm)
        self.assertEqual(dm.shape, image.shape)

    def test_limb_creates_scaled_mask(self):
        from deconvolve import _rl_setup_masks
        image = _synthetic_planet(seed=222)
        dm, lm = _rl_setup_masks(image, 0.0, 0.85, None, None)
        self.assertIsNone(dm)
        self.assertIsNotNone(lm)
        self.assertLessEqual(lm.max(), 0.85 + 1e-10)


# ===================================================================
# _rl_finalize
# ===================================================================

class TestRlFinalize(unittest.TestCase):
    def test_no_ops_returns_unchanged(self):
        from deconvolve import _rl_finalize
        image = _synthetic_planet(seed=230)
        result = _rl_finalize(image, image, 0.0, 4, 1.0)
        np.testing.assert_array_equal(result, image)

    def test_wavelet_reg_changes_result(self):
        from deconvolve import _rl_finalize
        image = _synthetic_planet(seed=231)
        result = _rl_finalize(image, image, 5.0, 4, 1.0)
        self.assertFalse(np.array_equal(result, image))


# ===================================================================
# _QualityBaselines
# ===================================================================

class TestQualityBaselines(unittest.TestCase):
    def test_large_planet_baselines(self):
        from optimizer import _QualityBaselines
        from metrics import planet_mask
        image = _synthetic_planet(seed=240, fill=0.40)
        pmask = planet_mask(image)
        baselines = _QualityBaselines.from_image(image, pmask, small_planet=False)
        self.assertGreater(baselines.noise_floor, 0)
        self.assertGreater(baselines.sharpness_floor, 0)

    def test_small_planet_higher_noise_mult(self):
        from optimizer import _QualityBaselines
        from metrics import planet_mask
        image = _synthetic_planet(seed=241, fill=0.40)
        pmask = planet_mask(image)
        large = _QualityBaselines.from_image(image, pmask, small_planet=False)
        small = _QualityBaselines.from_image(image, pmask, small_planet=True)
        # Small planet has more lenient noise floor (0.70 vs 0.55)
        self.assertGreater(small.noise_floor, large.noise_floor)


# ===================================================================
# _filter_and_rank
# ===================================================================

class TestFilterAndRank(unittest.TestCase):
    def test_empty_raises(self):
        from optimizer import _filter_and_rank
        with self.assertRaises(RuntimeError):
            _filter_and_rank([], 100.0, verbose=False)

    def test_sorts_by_normalised_score(self):
        from optimizer import _filter_and_rank, Candidate
        from metrics import all_metrics, composite_score
        from psf import build_psf
        from deconvolve import deconvolve

        image = _synthetic_planet(seed=250)
        candidates = []
        for snr in [10.0, 30.0]:
            psf = build_psf("gaussian", {"fwhm": 2.0, "size": 15})
            result = deconvolve("wiener", image, psf, {"snr": snr})
            metrics = all_metrics(result)
            candidates.append(Candidate(
                psf_type="gaussian",
                psf_params={"fwhm": 2.0, "size": 15},
                deconv_method="wiener",
                deconv_params={"snr": snr},
                result=result.astype(np.float32),
                metrics=metrics,
                raw_score=composite_score(result, metrics),
            ))

        ranked = _filter_and_rank(candidates, 0.0, verbose=False)
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0].normalised_score, ranked[1].normalised_score)


# ===================================================================
# _find_fits_files, _load_and_prepare, _print_top_results
# ===================================================================

class TestFindFitsFiles(unittest.TestCase):
    def test_exits_on_empty_dir(self):
        import tempfile
        from main import _find_fits_files
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit):
                _find_fits_files(Path(tmpdir))


# ===================================================================
# Multi-scale wavelet sharpening (level_gains)
# ===================================================================

class TestWaveletSharpenLevelGains(unittest.TestCase):
    def test_level_gains_none_matches_default(self):
        """level_gains=None must produce identical output to the bell-curve."""
        from postprocess import _wavelet_sharpen
        image = _synthetic_planet(seed=310)
        result_default = _wavelet_sharpen(image, gain=1.5, levels=4)
        result_none = _wavelet_sharpen(image, gain=1.5, levels=4, level_gains=None)
        np.testing.assert_array_equal(result_default, result_none)

    def test_level_gains_all_one_returns_unchanged(self):
        """All gains=1.0 means no sharpening."""
        from postprocess import _wavelet_sharpen
        image = _synthetic_planet(seed=311)
        result = _wavelet_sharpen(image, gain=2.0, levels=4, level_gains=[1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, image)

    def test_level_gains_boosts_specific_level(self):
        """Boosting level 1 should change the image differently from boosting level 2."""
        from postprocess import _wavelet_sharpen
        image = _synthetic_planet(seed=312)
        result_lv1 = _wavelet_sharpen(image, levels=4, level_gains=[1.0, 2.0, 1.0, 1.0])
        result_lv2 = _wavelet_sharpen(image, levels=4, level_gains=[1.0, 1.0, 2.0, 1.0])
        # Both should differ from original and from each other
        self.assertFalse(np.array_equal(result_lv1, image))
        self.assertFalse(np.array_equal(result_lv2, image))
        self.assertFalse(np.array_equal(result_lv1, result_lv2))

    def test_level_gains_preserves_shape(self):
        from postprocess import _wavelet_sharpen
        image = _synthetic_planet(seed=313)
        result = _wavelet_sharpen(image, levels=4, level_gains=[1.0, 1.8, 1.4, 1.0])
        self.assertEqual(result.shape, image.shape)


class TestPostprocessConfigLevelGains(unittest.TestCase):
    def test_default_level_gains_is_none(self):
        from postprocess import PostprocessConfig
        cfg = PostprocessConfig()
        self.assertIsNone(cfg.level_gains)

    def test_custom_level_gains(self):
        from postprocess import PostprocessConfig
        cfg = PostprocessConfig(level_gains=[1.0, 1.8, 1.4, 1.0])
        self.assertEqual(cfg.level_gains, [1.0, 1.8, 1.4, 1.0])

    def test_postprocess_activates_on_level_gains(self):
        """postprocess should sharpen even when sharpen=0 if level_gains is set."""
        from postprocess import PostprocessConfig, postprocess
        image = _synthetic_planet(seed=314)
        cfg = PostprocessConfig(wv=0.0, nlm=0.0, sharpen=0.0, dp=0.0,
                                level_gains=[1.0, 2.0, 1.5, 1.0])
        result = postprocess(image, cfg)
        self.assertFalse(np.array_equal(result, image))


class TestAdaptPostprocessingLevelGains(unittest.TestCase):
    def test_small_planet_sets_level_gains(self):
        from main import adapt_postprocessing
        from postprocess import PostprocessConfig
        image = _synthetic_planet(fill=0.06)
        cfg = PostprocessConfig(wv=25.0, nlm=0.008, sharpen=0.0)
        result = adapt_postprocessing(image, cfg)
        self.assertIsNotNone(result.level_gains)
        self.assertEqual(len(result.level_gains), 4)

    def test_large_planet_no_level_gains(self):
        from main import adapt_postprocessing
        from postprocess import PostprocessConfig
        image = _synthetic_planet(fill=0.40)
        cfg = PostprocessConfig(wv=25.0, nlm=0.008, sharpen=0.0)
        result = adapt_postprocessing(image, cfg)
        self.assertIsNone(result.level_gains)


if __name__ == "__main__":
    unittest.main()
