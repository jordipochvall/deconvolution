"""
Bayesian parameter optimisation for planetary deconvolution (Optuna TPE).

Pipeline
--------
1. Define a continuous search space for PSF + deconvolution parameters.
2. Optuna's TPE sampler proposes combinations based on past results.
3. Each trial: build PSF → deconvolve → compute metrics → return score.
4. Noise floor filter: reject candidates noisier than the input.
5. Sharpness floor: exclude barely-sharpened candidates from ranking.
6. After N trials, normalise scores across the pool and rank.

Typical budget: 80–120 trials (vs 600+ brute-force combinations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna

from psf import build_psf
from deconvolve import deconvolve
from metrics import all_metrics, composite_score, _WEIGHTS



# ---------------------------------------------------------------------------
# Candidate data
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One evaluated parameter combination and its result."""
    psf_type:         str
    psf_params:       dict[str, Any]
    deconv_method:    str
    deconv_params:    dict[str, Any]
    result:           np.ndarray        = field(repr=False)
    metrics:          dict[str, float]  = field(default_factory=dict)
    raw_score:        float             = 0.0
    normalised_score: float             = 0.0
    source:           str               = "optuna"


# ---------------------------------------------------------------------------
# Normalisation and scoring
# ---------------------------------------------------------------------------

# Metrics where LOWER raw values are BETTER (inverted during scoring)
_INVERTED_METRICS = {"laplacian_variance"}


def _default_seed_candidates(small_planet: bool) -> list[dict[str, Any]]:
    """
    Deterministic anchor candidates to stabilise Optuna around known-good regions.

    These are generic priors (not file-dependent) and are evaluated in addition
    to sampled trials.
    """
    if small_planet:
        return [
            {
                "psf_type": "moffat",
                "psf_params": {"fwhm": 1.2, "beta": 2.0, "size": 21},
                "deconv_method": "richardson_lucy",
                "deconv_params": {
                    "iterations": 4,
                    "damping": 5e-4,
                    "tv_lambda": 0.28,
                    "deringing": 0.2,
                    "contrast_boost": 1.0,
                    "limb_suppression": 0.3,
                },
            },
            {
                "psf_type": "gaussian",
                "psf_params": {"fwhm": 1.25, "size": 21},
                "deconv_method": "wiener",
                "deconv_params": {"snr": 45.0},
            },
        ]

    return [
        {
            "psf_type": "moffat",
            "psf_params": {"fwhm": 4.9, "beta": 2.1, "size": 21},
            "deconv_method": "richardson_lucy",
            "deconv_params": {
                "iterations": 232,
                "damping": 2e-4,
                "tv_lambda": 0.20,
                "deringing": 0.455,
                "contrast_boost": 1.446,
                "limb_suppression": 0.85,
            },
        },
        {
            "psf_type": "moffat",
            "psf_params": {"fwhm": 4.7, "beta": 1.9, "size": 21},
            "deconv_method": "richardson_lucy",
            "deconv_params": {
                "iterations": 210,
                "damping": 2e-4,
                "tv_lambda": 0.18,
                "deringing": 0.40,
                "contrast_boost": 1.42,
                "limb_suppression": 0.85,
            },
        },
        {
            "psf_type": "moffat",
            "psf_params": {"fwhm": 4.5, "beta": 1.8, "size": 31},
            "deconv_method": "richardson_lucy",
            "deconv_params": {
                "iterations": 180,
                "damping": 3e-4,
                "tv_lambda": 0.15,
                "deringing": 0.35,
                "contrast_boost": 1.40,
                "limb_suppression": 0.85,
            },
        },
        # Diverse anchor: low beta, high damping.  Extra contrast boost
        # compensates for post-processing tenengrad/brenner reduction.
        {
            "psf_type": "moffat",
            "psf_params": {"fwhm": 4.84, "beta": 1.5, "size": 21},
            "deconv_method": "richardson_lucy",
            "deconv_params": {
                "iterations": 162,
                "damping": 6e-3,
                "tv_lambda": 0.10,
                "deringing": 0.089,
                "contrast_boost": 1.54,
                "limb_suppression": 0.85,
            },
        },
        # Diverse anchor: moderate params, gaussian PSF
        {
            "psf_type": "gaussian",
            "psf_params": {"fwhm": 4.5, "size": 21},
            "deconv_method": "richardson_lucy",
            "deconv_params": {
                "iterations": 120,
                "damping": 1e-3,
                "tv_lambda": 0.15,
                "deringing": 0.30,
                "contrast_boost": 1.35,
                "limb_suppression": 0.85,
            },
        },
    ]


def _normalise_metrics(candidates: list[Candidate], weights: dict) -> None:
    """Min-max normalise each metric across all candidates (in-place)."""
    for name in weights:
        values = np.array([c.metrics[name] for c in candidates])
        lo, hi = values.min(), values.max()
        span = hi - lo if hi > lo else 1.0
        for c in candidates:
            c.metrics[f"{name}_norm"] = (c.metrics[name] - lo) / span


def _composite_normalised(candidate: Candidate, weights: dict) -> float:
    """Weighted composite of normalised metrics (inverted where appropriate)."""
    score = 0.0
    for k, w in weights.items():
        v = candidate.metrics.get(f"{k}_norm", 0.0)
        if k in _INVERTED_METRICS:
            v = 1.0 - v  # lower raw value → higher score
        score += w * v
    return score


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(
    image: np.ndarray,
    n_trials: int = 100,
    progress_callback=None,
    verbose: bool = True,
    search_space: dict | None = None,
    seed_candidates: list[dict[str, Any]] | None = None,
) -> list[Candidate]:
    """
    Find the best deconvolution parameters via Bayesian optimisation.

    Parameters
    ----------
    image              Input planetary image (2-D, float).
    n_trials           Number of Optuna trials (default 100).
    progress_callback  Called as callback(done, total) after each trial.
    verbose            Print progress messages (default True).
    search_space       Unused (kept for API compatibility).
    seed_candidates    Optional fixed candidate configurations to force-evaluate.

    Returns
    -------
    list[Candidate]  Valid candidates sorted by normalised_score (best first).
    """
    from metrics import planet_mask, smoothness as _smoothness, tenengrad as _tenengrad
    seed_candidates = seed_candidates or []

    def _signature(
        psf_type: str,
        psf_params: dict[str, Any],
        method: str,
        deconv_params: dict[str, Any],
    ) -> tuple:
        return (
            psf_type,
            tuple(sorted(psf_params.items())),
            method,
            tuple(sorted(deconv_params.items())),
        )

    # Detect planet size to adapt search space
    pmask = planet_mask(image)
    planet_fraction = pmask.sum() / pmask.size
    small_planet = planet_fraction < 0.20
    seed_candidates = _default_seed_candidates(small_planet) + seed_candidates

    if small_planet and verbose:
        print(f"  Small planet detected ({planet_fraction * 100:.1f}% of frame).")
        print("  Adapting search space: fewer iterations, gentler contrast boost.")

    # Quality baselines from the input image
    input_smoothness = _smoothness(image, mask=pmask)
    input_tenengrad  = _tenengrad(image, mask=pmask)

    # For small planets (Saturn), PixInsight-quality processing only improves
    # tenengrad by ~1%.  A 1.5× sharpness floor would exclude gentle (correct)
    # solutions and force destructive over-deconvolution.
    noise_floor_mult    = 0.70 if small_planet else 0.55
    sharpness_floor_mult = 1.05 if small_planet else 1.50

    noise_floor     = input_smoothness * noise_floor_mult
    sharpness_floor = input_tenengrad * sharpness_floor_mult
    if verbose:
        print(f"  Input smoothness: {input_smoothness:.4f}  (floor: {noise_floor:.4f})")
        print(f"  Input tenengrad:  {input_tenengrad:.0f}  (floor: {sharpness_floor:.0f})")

    # Search space bounds — adapted for small planets to avoid over-deconvolution.
    # Saturn at ~6% of frame: PixInsight-quality processing barely touches
    # sharpness (+1% tenengrad).  The search space must be tight enough that
    # even the most aggressive candidate is gentle.
    if small_planet:
        fwhm_range   = (1.0, 1.5)    # narrow PSF — small planets need very gentle correction
        rl_min_iter  = 2
        rl_max_iter  = 5             # minimal iterations — PI barely sharpens Saturn
        rl_max_cb    = 1.0           # no contrast boost
        rl_min_tv    = 0.25          # strong TV regularisation throughout
        rl_limb_supp = 0.3           # ring edges are features, not artifacts
        wiener_snr   = (30.0, 60.0)  # gentle sharpening only
        tik_reg      = (1e-2, 5e-2)  # heavy regularisation
    else:
        fwhm_range   = (1.0, 5.0)
        rl_min_iter  = 15
        rl_max_iter  = 300
        rl_max_cb    = 1.50
        rl_min_tv    = 0.0
        rl_limb_supp = 0.85
        wiener_snr   = (3.0, 60.0)
        tik_reg      = (1e-4, 1e-1)

    evaluated: list[Candidate] = []
    rejected = 0
    trial_count = 0
    seen_signatures: set[tuple] = set()

    def objective(trial: optuna.Trial) -> float:
        nonlocal rejected, trial_count

        # --- PSF parameters ---
        psf_type = trial.suggest_categorical("psf_type", ["gaussian", "moffat"])
        fwhm = trial.suggest_float("fwhm", fwhm_range[0], fwhm_range[1])
        size = trial.suggest_categorical("psf_size", [15, 21, 31])

        psf_params: dict[str, Any] = {"fwhm": round(fwhm, 2), "size": size}
        if psf_type == "moffat":
            psf_params["beta"] = round(trial.suggest_float("beta", 1.5, 6.0), 2)

        # --- Deconvolution parameters ---
        method = trial.suggest_categorical(
            "deconv_method", ["richardson_lucy", "wiener", "tikhonov"],
        )

        deconv_params: dict[str, Any] = {}
        if method == "richardson_lucy":
            deconv_params = {
                "iterations":       trial.suggest_int("rl_iterations", rl_min_iter, rl_max_iter),
                "damping":          round(trial.suggest_float("rl_damping", 1e-4, 1e-2, log=True), 6),
                "tv_lambda":        round(trial.suggest_float("rl_tv_lambda", rl_min_tv, 0.3), 6),
                "deringing":        round(trial.suggest_float("rl_deringing", 0.0, 0.8), 3),
                "contrast_boost":   round(trial.suggest_float("rl_contrast_boost", 1.0, rl_max_cb), 3),
                "limb_suppression": rl_limb_supp,
            }
        elif method == "wiener":
            deconv_params = {"snr": round(trial.suggest_float("wiener_snr", wiener_snr[0], wiener_snr[1], log=True), 2)}
        elif method == "tikhonov":
            deconv_params = {"regularization": round(trial.suggest_float("tikhonov_reg", tik_reg[0], tik_reg[1], log=True), 6)}

        trial_rejected = False
        candidate: Candidate | None = None

        # --- Evaluate ---
        try:
            psf = build_psf(psf_type, psf_params)
            result = deconvolve(method, image, psf, deconv_params)
        except Exception:
            score = -1.0
        else:
            metrics = all_metrics(result)
            if metrics["smoothness"] < noise_floor:
                trial_rejected = True
                score = -0.5
            else:
                raw_score = composite_score(result, metrics)
                candidate = Candidate(
                    psf_type=psf_type,
                    psf_params=psf_params,
                    deconv_method=method,
                    deconv_params=deconv_params,
                    result=result.astype(np.float32, copy=False),
                    metrics=metrics,
                    raw_score=raw_score,
                    source="optuna",
                )
                score = raw_score

        trial_count += 1
        if trial_rejected:
            rejected += 1
        if candidate is not None:
            sig = _signature(
                candidate.psf_type,
                candidate.psf_params,
                candidate.deconv_method,
                candidate.deconv_params,
            )
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                evaluated.append(candidate)
        if progress_callback:
            progress_callback(trial_count, n_trials)

        return score

    # Run optimisation
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

    # Force-evaluate fixed seed candidates after Optuna trials.
    # This is useful when we already have known-good parameter sets.
    seed_added = 0
    for seed in seed_candidates:
        try:
            psf_type = str(seed["psf_type"])
            psf_params = dict(seed["psf_params"])
            method = str(seed["deconv_method"])
            deconv_params = dict(seed["deconv_params"])
        except Exception:
            continue

        sig = _signature(psf_type, psf_params, method, deconv_params)
        if sig in seen_signatures:
            continue

        try:
            psf = build_psf(psf_type, psf_params)
            result = deconvolve(method, image, psf, deconv_params)
        except Exception:
            continue

        metrics = all_metrics(result)
        if metrics["smoothness"] < noise_floor:
            continue

        evaluated.append(
            Candidate(
                psf_type=psf_type,
                psf_params=psf_params,
                deconv_method=method,
                deconv_params=deconv_params,
                result=result.astype(np.float32, copy=False),
                metrics=metrics,
                raw_score=composite_score(result, metrics),
                source="seed",
            )
        )
        seen_signatures.add(sig)
        seed_added += 1

    if verbose:
        print(f"  {rejected} trials rejected (noisier than input), {len(evaluated)} kept.")
        if seed_added:
            print(f"  Added {seed_added} fixed seed candidates.")

    if not evaluated:
        raise RuntimeError(
            "All candidates were rejected (all noisier than input). "
            "Try increasing n_trials or relaxing the noise floor."
        )

    # Sharpness floor: exclude barely-processed results from ranking
    sharp = [c for c in evaluated if c.metrics["tenengrad"] >= sharpness_floor]
    if sharp:
        n_soft = len(evaluated) - len(sharp)
        if n_soft and verbose:
            print(f"  {n_soft} trials too conservative (below sharpness floor), "
                  f"{len(sharp)} kept for ranking.")
        evaluated = sharp

    # Pool-wide normalisation and final ranking
    _normalise_metrics(evaluated, _WEIGHTS)
    for c in evaluated:
        c.normalised_score = _composite_normalised(c, _WEIGHTS)
    evaluated.sort(key=lambda c: c.normalised_score, reverse=True)

    return evaluated
