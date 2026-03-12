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
from deconvolve import deconvolve, build_rl_masks
from metrics import all_metrics, composite_score, tenengrad as _tenengrad, _WEIGHTS, _INVERTED_METRICS


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
# Search space configuration
# ---------------------------------------------------------------------------

@dataclass
class _SearchBounds:
    """Holds all search space bounds, determined by planet size."""
    fwhm_range:   tuple[float, float]
    rl_min_iter:  int
    rl_max_iter:  int
    rl_max_cb:    float
    rl_min_tv:    float
    rl_max_tv:    float
    rl_limb_supp: float
    wiener_snr:   tuple[float, float]
    tik_reg:      tuple[float, float]

    @staticmethod
    def for_planet(small_planet: bool) -> _SearchBounds:
        if small_planet:
            return _SearchBounds(
                fwhm_range=(1.0, 2.5),
                rl_min_iter=2, rl_max_iter=20,
                rl_max_cb=1.0,
                rl_min_tv=0.30, rl_max_tv=0.80,
                rl_limb_supp=0.3,
                wiener_snr=(30.0, 60.0),
                tik_reg=(1e-2, 5e-2),
            )
        return _SearchBounds(
            fwhm_range=(1.0, 5.0),
            rl_min_iter=15, rl_max_iter=300,
            rl_max_cb=1.50,
            rl_min_tv=0.0, rl_max_tv=0.30,
            rl_limb_supp=0.85,
            wiener_snr=(3.0, 60.0),
            tik_reg=(1e-4, 1e-1),
        )


# ---------------------------------------------------------------------------
# Normalisation and scoring
# ---------------------------------------------------------------------------

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
                "psf_params": {"fwhm": 1.5, "beta": 2.0, "size": 21},
                "deconv_method": "richardson_lucy",
                "deconv_params": {
                    "iterations": 12,
                    "damping": 5e-4,
                    "tv_lambda": 0.50,
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
        # Diverse anchor: low beta, high damping
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


def _candidate_signature(
    psf_type: str,
    psf_params: dict[str, Any],
    method: str,
    deconv_params: dict[str, Any],
) -> tuple:
    """Hashable signature to detect duplicate parameter combinations."""
    return (
        psf_type,
        tuple(sorted(psf_params.items())),
        method,
        tuple(sorted(deconv_params.items())),
    )


# ---------------------------------------------------------------------------
# RL checkpointed evaluation (for MedianPruner)
# ---------------------------------------------------------------------------

_N_SEGMENTS = 3
_RL_MIN_ITER_FOR_PRUNING = 9


def _evaluate_rl_checkpointed(
    trial: optuna.Trial,
    image: np.ndarray,
    psf: np.ndarray,
    deconv_params: dict[str, Any],
    pmask: np.ndarray,
) -> np.ndarray | None:
    """Run Richardson-Lucy with checkpoint-based pruning.

    Splits iterations into segments and reports intermediate tenengrad
    for MedianPruner.  Returns the final result, or None if pruned.
    """
    total_iters = deconv_params["iterations"]

    if total_iters < _RL_MIN_ITER_FOR_PRUNING:
        return deconvolve("richardson_lucy", image, psf, deconv_params)

    dering_m, limb_m = build_rl_masks(
        image,
        deconv_params["deringing"],
        deconv_params["limb_suppression"],
    )
    base = dict(deconv_params)
    base["dering_mask"] = dering_m
    base["limb_mask"] = limb_m

    estimate = None
    for cp in range(_N_SEGMENTS):
        is_last = (cp == _N_SEGMENTS - 1)
        i0 = cp * total_iters // _N_SEGMENTS
        i1 = total_iters if is_last else (cp + 1) * total_iters // _N_SEGMENTS
        cp_iters = i1 - i0
        if cp_iters <= 0:
            continue

        cp_params = dict(base)
        cp_params["iterations"] = cp_iters
        cp_params["initial_estimate"] = estimate
        if not is_last:
            cp_params["contrast_boost"] = 1.0
            cp_params["wavelet_reg"] = 0.0

        estimate = deconvolve("richardson_lucy", image, psf, cp_params)

        if not is_last:
            trial.report(_tenengrad(estimate, mask=pmask), cp)
            if trial.should_prune():
                return None

    return estimate


# ---------------------------------------------------------------------------
# Seed candidate evaluation
# ---------------------------------------------------------------------------

def _evaluate_seeds(
    seed_candidates: list[dict[str, Any]],
    image: np.ndarray,
    noise_floor: float,
    seen_signatures: set[tuple],
) -> list[Candidate]:
    """Evaluate fixed seed candidates and return valid ones."""
    results = []
    for seed in seed_candidates:
        try:
            psf_type = str(seed["psf_type"])
            psf_params = dict(seed["psf_params"])
            method = str(seed["deconv_method"])
            deconv_params = dict(seed["deconv_params"])
        except Exception:
            continue

        sig = _candidate_signature(psf_type, psf_params, method, deconv_params)
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

        results.append(
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

    return results


# ---------------------------------------------------------------------------
# Parameter suggestion
# ---------------------------------------------------------------------------

def _suggest_params(
    trial: optuna.Trial,
    bounds: _SearchBounds,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Sample PSF and deconvolution parameters from the search space."""
    psf_type = trial.suggest_categorical("psf_type", ["gaussian", "moffat"])
    fwhm = trial.suggest_float("fwhm", bounds.fwhm_range[0], bounds.fwhm_range[1])
    size = trial.suggest_categorical("psf_size", [15, 21, 31])

    psf_params: dict[str, Any] = {"fwhm": round(fwhm, 2), "size": size}
    if psf_type == "moffat":
        psf_params["beta"] = round(trial.suggest_float("beta", 1.5, 6.0), 2)

    method = trial.suggest_categorical(
        "deconv_method", ["richardson_lucy", "wiener", "tikhonov"],
    )

    deconv_params: dict[str, Any] = {}
    if method == "richardson_lucy":
        deconv_params = {
            "iterations":       trial.suggest_int("rl_iterations", bounds.rl_min_iter, bounds.rl_max_iter),
            "damping":          round(trial.suggest_float("rl_damping", 1e-4, 1e-2, log=True), 6),
            "tv_lambda":        round(trial.suggest_float("rl_tv_lambda", bounds.rl_min_tv, bounds.rl_max_tv), 6),
            "deringing":        round(trial.suggest_float("rl_deringing", 0.0, 0.8), 3),
            "contrast_boost":   round(trial.suggest_float("rl_contrast_boost", 1.0, bounds.rl_max_cb), 3),
            "limb_suppression": bounds.rl_limb_supp,
        }
    elif method == "wiener":
        deconv_params = {"snr": round(trial.suggest_float("wiener_snr", bounds.wiener_snr[0], bounds.wiener_snr[1], log=True), 2)}
    elif method == "tikhonov":
        deconv_params = {"regularization": round(trial.suggest_float("tikhonov_reg", bounds.tik_reg[0], bounds.tik_reg[1], log=True), 6)}

    return psf_type, psf_params, method, deconv_params


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

@dataclass
class _QualityBaselines:
    """Noise and sharpness thresholds computed from the input image."""
    noise_floor: float
    sharpness_floor: float
    input_smoothness: float
    input_tenengrad: float

    @staticmethod
    def from_image(
        image: np.ndarray, pmask: np.ndarray, small_planet: bool,
    ) -> _QualityBaselines:
        from metrics import smoothness as _smoothness, tenengrad as _tenengrad

        input_smoothness = _smoothness(image, mask=pmask)
        input_tenengrad = _tenengrad(image, mask=pmask)
        noise_mult = 0.70 if small_planet else 0.55
        sharp_mult = 1.05 if small_planet else 1.50
        return _QualityBaselines(
            noise_floor=input_smoothness * noise_mult,
            sharpness_floor=input_tenengrad * sharp_mult,
            input_smoothness=input_smoothness,
            input_tenengrad=input_tenengrad,
        )


def _filter_and_rank(
    evaluated: list[Candidate],
    sharpness_floor: float,
    verbose: bool,
) -> list[Candidate]:
    """Apply sharpness floor, normalise metrics, and sort by score."""
    if not evaluated:
        raise RuntimeError(
            "All candidates were rejected (all noisier than input). "
            "Try increasing n_trials or relaxing the noise floor."
        )

    sharp = [c for c in evaluated if c.metrics["tenengrad"] >= sharpness_floor]
    if sharp:
        n_soft = len(evaluated) - len(sharp)
        if n_soft and verbose:
            print(f"  {n_soft} trials too conservative (below sharpness floor), "
                  f"{len(sharp)} kept for ranking.")
        evaluated = sharp

    _normalise_metrics(evaluated, _WEIGHTS)
    for c in evaluated:
        c.normalised_score = _composite_normalised(c, _WEIGHTS)
    evaluated.sort(key=lambda c: c.normalised_score, reverse=True)
    return evaluated


def _build_objective(
    image: np.ndarray,
    bounds: _SearchBounds,
    pmask: np.ndarray,
    noise_floor: float,
    n_trials: int,
    evaluated: list[Candidate],
    seen_signatures: set[tuple],
    progress_callback,
) -> tuple[callable, dict]:
    """Create the Optuna objective closure and a shared state dict."""
    state = {"rejected": 0, "trial_count": 0}

    def objective(trial: optuna.Trial) -> float:
        psf_type, psf_params, method, deconv_params = _suggest_params(trial, bounds)

        score = -1.0
        candidate: Candidate | None = None
        was_pruned = False

        try:
            psf = build_psf(psf_type, psf_params)

            if method == "richardson_lucy":
                result = _evaluate_rl_checkpointed(
                    trial, image, psf, deconv_params, pmask,
                )
                was_pruned = result is None
            else:
                result = deconvolve(method, image, psf, deconv_params)

            if not was_pruned:
                metrics = all_metrics(result)
                if metrics["smoothness"] < noise_floor:
                    was_pruned = True
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
        except Exception:
            score = -1.0

        state["trial_count"] += 1
        if was_pruned:
            state["rejected"] += 1
        if candidate is not None:
            sig = _candidate_signature(
                candidate.psf_type, candidate.psf_params,
                candidate.deconv_method, candidate.deconv_params,
            )
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                evaluated.append(candidate)
        if progress_callback:
            progress_callback(state["trial_count"], n_trials)

        if was_pruned and score != -0.5:
            raise optuna.TrialPruned()
        return score

    return objective, state


def run_search(
    image: np.ndarray,
    n_trials: int = 100,
    progress_callback=None,
    verbose: bool = True,
    seed_candidates: list[dict[str, Any]] | None = None,
) -> list[Candidate]:
    """Find the best deconvolution parameters via Bayesian optimisation.

    Returns candidates sorted by normalised_score (best first).
    """
    from metrics import planet_mask

    seed_candidates = seed_candidates or []

    # Detect planet size and adapt search space
    pmask = planet_mask(image)
    planet_fraction = pmask.sum() / pmask.size
    small_planet = planet_fraction < 0.20
    seed_candidates = _default_seed_candidates(small_planet) + seed_candidates
    bounds = _SearchBounds.for_planet(small_planet)

    if small_planet and verbose:
        print(f"  Small planet detected ({planet_fraction * 100:.1f}% of frame).")
        print("  Adapting search space: fewer iterations, gentler contrast boost.")

    baselines = _QualityBaselines.from_image(image, pmask, small_planet)
    if verbose:
        print(f"  Input smoothness: {baselines.input_smoothness:.4f}  (floor: {baselines.noise_floor:.4f})")
        print(f"  Input tenengrad:  {baselines.input_tenengrad:.0f}  (floor: {baselines.sharpness_floor:.0f})")

    evaluated: list[Candidate] = []
    seen_signatures: set[tuple] = set()

    objective, state = _build_objective(
        image, bounds, pmask, baselines.noise_floor,
        n_trials, evaluated, seen_signatures, progress_callback,
    )

    # Run optimisation
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

    # Force-evaluate fixed seed candidates
    seed_results = _evaluate_seeds(
        seed_candidates, image, baselines.noise_floor, seen_signatures,
    )
    evaluated.extend(seed_results)

    if verbose:
        print(f"  {state['rejected']} trials rejected (noisier than input), {len(evaluated)} kept.")
        if seed_results:
            print(f"  Added {len(seed_results)} fixed seed candidates.")

    return _filter_and_rank(evaluated, baselines.sharpness_floor, verbose)
