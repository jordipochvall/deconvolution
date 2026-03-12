"""
Post-processing re-ranking of deconvolution candidates.

Optuna scores candidates on raw RL output, but the final image includes
wavelet denoising + NLM which can change the relative ranking.  This module
re-evaluates the top candidates after full post-processing and picks the
best using normalised weighted scoring.
"""

from __future__ import annotations

import numpy as np

from fits_io import to_luminance
from postprocess import PostprocessConfig, postprocess, postprocess_rgb
from rgb import apply_best_to_color


def build_rerank_pool(candidates: list, max_top: int = 20) -> list:
    """Select candidates for re-ranking: top N plus any seed candidates.

    Seed candidates are always included even if they fall outside the
    top N, because they represent domain-knowledge anchors that may
    perform better after post-processing.

    Parameters
    ----------
    candidates  Sorted list of Candidate objects (best first).
    max_top     Maximum number of top candidates to include.

    Returns
    -------
    List of candidates to re-rank (no duplicates).
    """
    rerank_n = min(max_top, len(candidates))
    pool = list(candidates[:rerank_n])
    seen_ids = {id(c) for c in pool}
    for c in candidates:
        if getattr(c, "source", "optuna") == "seed" and id(c) not in seen_ids:
            pool.append(c)
            seen_ids.add(id(c))
    return pool


def evaluate_rerank_pool(
    pool: list,
    input_image: np.ndarray,
    is_color: bool,
    cfg: PostprocessConfig,
    pmask: np.ndarray,
    rgb_jobs: int = 1,
) -> list[tuple[object, np.ndarray, dict]]:
    """Evaluate each candidate with full post-processing and compute metrics.

    Each candidate is deconvolved (RGB if colour), post-processed, and
    scored on the luminance channel using the provided planet mask.

    Parameters
    ----------
    pool         Candidates to evaluate.
    input_image  Original image ((3,H,W) for colour, (H,W) for mono).
    is_color     Whether the input is RGB.
    cfg          Post-processing configuration.
    pmask        Planet mask for metric computation.
    rgb_jobs     Parallel workers for RGB channel deconvolution.

    Returns
    -------
    List of (candidate, post-processed_image, metrics_dict) tuples.
    """
    from metrics import all_metrics

    entries = []
    for c in pool:
        if is_color:
            trial_result = apply_best_to_color(input_image, c, rgb_jobs=rgb_jobs)
            trial_result = postprocess_rgb(trial_result, cfg, jobs=rgb_jobs)
            lum = to_luminance(trial_result)
        else:
            trial_result = postprocess(c.result, cfg)
            lum = trial_result
        entries.append((c, trial_result, all_metrics(lum, mask=pmask)))
    return entries


def pick_best_candidate(
    entries: list[tuple[object, np.ndarray, dict]],
    fallback: object,
    verbose: bool = False,
) -> tuple[object, np.ndarray | None]:
    """Normalise metrics across pool and pick the best candidate.

    Uses min-max normalisation per metric, then computes a weighted
    composite score.  Inverted metrics (like laplacian_variance, where
    lower is better) are flipped so higher normalised = better.

    Parameters
    ----------
    entries   List of (candidate, image, metrics) from evaluate_rerank_pool.
    fallback  Default candidate to return if no better one is found.
    verbose   Print per-candidate scores.

    Returns
    -------
    (best_candidate, cached_result) where cached_result is the
    post-processed image of the winner.
    """
    from metrics import _WEIGHTS, _INVERTED_METRICS

    # Min-max normalise each metric across the pool
    for name in _WEIGHTS:
        values = np.array([m[name] for _, _, m in entries])
        lo, hi = values.min(), values.max()
        span = hi - lo if hi > lo else 1.0
        for _, _, m in entries:
            m[f"{name}_norm"] = (m[name] - lo) / span

    best = fallback
    best_score = -np.inf
    cached_result = None

    for idx, (c, trial_result, metrics) in enumerate(entries, 1):
        score = sum(
            w * (1.0 - metrics.get(f"{k}_norm", 0.0) if k in _INVERTED_METRICS
                 else metrics.get(f"{k}_norm", 0.0))
            for k, w in _WEIGHTS.items()
        )
        if verbose:
            print(
                f"  [{idx}] norm_score={score:.4f} "
                f"qr={metrics['quality_ratio']:.1f} "
                f"lap={metrics['laplacian_variance']:.0f} "
                f"ten={metrics['tenengrad']:.0f}"
            )
        if score > best_score:
            best_score = score
            best = c
            cached_result = trial_result.astype(np.float32, copy=False)

    if verbose and best is not fallback:
        print("  Re-rank selected a different best candidate.")
    return best, cached_result


def rerank_candidates(
    candidates: list,
    input_image: np.ndarray,
    opt_image: np.ndarray,
    is_color: bool,
    cfg: PostprocessConfig,
    rgb_jobs: int = 1,
    verbose: bool = False,
) -> tuple[object, np.ndarray | None]:
    """Re-rank top candidates by post-processed quality.

    Builds a re-ranking pool (top 20 + seeds), evaluates each with full
    post-processing, and picks the best using normalised weighted scoring.

    Parameters
    ----------
    candidates    Sorted list of Candidate objects (best first).
    input_image   Original image for RGB deconvolution.
    opt_image     Luminance image used for planet mask.
    is_color      Whether the input is RGB.
    cfg           Post-processing configuration.
    rgb_jobs      Parallel workers for RGB channels.
    verbose       Print progress and scores.

    Returns
    -------
    (best_candidate, cached_result) where cached_result is the
    post-processed image of the winner (or None if skipped).
    """
    from metrics import planet_mask

    if len(candidates) <= 1:
        return candidates[0], None

    pool = build_rerank_pool(candidates)
    if verbose:
        extra = len(pool) - min(20, len(candidates))
        msg = f"\nRe-ranking top {min(20, len(candidates))} candidates after post-processing preview"
        if extra > 0:
            msg += f" + {extra} seed candidate(s)"
        print(msg + " ...")

    pmask = planet_mask(opt_image)
    entries = evaluate_rerank_pool(pool, input_image, is_color, cfg, pmask, rgb_jobs)
    return pick_best_candidate(entries, candidates[0], verbose)
