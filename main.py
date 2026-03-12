"""
Planetary deconvolution optimizer -- CLI entry point.

Usage
-----
    python main.py images/                      # process all FITS in directory
    python main.py images/ --top 5
    python main.py images/ --no-post            # skip wavelet+NLM post-processing
    python main.py images/ --wv 15 --nlm 0.005  # custom post-processing

Run  python main.py --help  for the full option list.
Output is saved to an ``output/`` subdirectory inside the input directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fits_io import load_fits_image, to_luminance, save_fits
from optimizer import run_search
from postprocess import PostprocessConfig, postprocess, postprocess_rgb
from rerank import rerank_candidates
from rgb import apply_best_to_color
from visualize import plot_best_results, plot_metrics_heatmap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the deconvolution pipeline."""
    p = argparse.ArgumentParser(
        description="Find the optimal deconvolution for all planetary FITS images in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", help="Directory containing FITS files to process.")
    p.add_argument("--channel", type=int, default=None,
                   help="Channel index for 3-D FITS cubes.  Omit for auto RGB detection.")
    p.add_argument("--top",    type=int,   default=1,
                   help="Number of top ranked candidates to report/save.")
    p.add_argument("--trials", type=int,   default=100,  help="Number of Optuna trials.")
    p.add_argument("--file-jobs", type=int, default=1,
                   help="Parallel worker threads across input FITS files.")
    p.add_argument("--rgb-jobs", type=int, default=1,
                   help="Parallel worker threads across RGB channels.")
    p.add_argument("--show",   action="store_true",      help="Display figures interactively.")
    p.add_argument("--save-plots", action="store_true",
                   help="Save summary plots (best results and metrics heatmap).")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce console output to essential messages.")
    p.add_argument("--save-fits", action="store_true",    help="Save top-N as FITS files.")
    p.add_argument("--no-post", action="store_true",      help="Skip wavelet+NLM post-processing.")
    p.add_argument("--wv",  type=float, default=25.0,    help="Wavelet threshold (0 = disabled).")
    p.add_argument("--nlm", type=float, default=0.008,   help="NLM strength (0 = disabled).")
    p.add_argument("--dp",  type=float, default=0.5,
                   help="Disk preservation (0 = uniform, 0.5 = 50%% less denoise on disk, 1 = skip disk).")
    p.add_argument("--sharpen", type=float, default=0.0,
                   help="Wavelet sharpening gain (0 = auto, 1.5 = moderate, 2.0 = strong).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _progress(done: int, total: int) -> None:
    """Print an in-place progress bar for Optuna trials."""
    pct = done / total * 100
    bar_len = 40
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}", end="", flush=True)
    if done == total:
        print()


# ---------------------------------------------------------------------------
# Post-processing adaptation
# ---------------------------------------------------------------------------

def adapt_postprocessing(
    opt_image: np.ndarray,
    cfg: PostprocessConfig,
    verbose: bool = False,
) -> PostprocessConfig:
    """Adapt post-processing strengths for small planets.

    Wavelet denoising destroys fine detail on small targets (<20% of frame).
    For those, wavelet denoising is disabled and wavelet sharpening is
    auto-enabled instead, to enhance medium-scale features (bands, ring gaps).

    Parameters
    ----------
    opt_image  Luminance image for planet size detection.
    cfg        Initial post-processing configuration.
    verbose    Print adaptation info.

    Returns
    -------
    A (possibly modified) PostprocessConfig.
    """
    from metrics import planet_mask

    pmask = planet_mask(opt_image)
    planet_frac = pmask.sum() / pmask.size
    if planet_frac < 0.20:
        # Per-level wavelet sharpening: boost ring/band scales, skip noise scale.
        # Level 0 = finest (~1-2px, noise), level 1 = mid-fine (~2-4px, ring edges),
        # level 2 = mid-coarse (~4-8px, broad structure), level 3 = coarsest (shape).
        small_level_gains = [1.0, 1.8, 1.4, 1.0]
        cfg = PostprocessConfig(
            wv=0.0,                          # skip wavelet denoising
            nlm=min(cfg.nlm, 0.003),         # very gentle NLM only
            sharpen=1.5 if cfg.sharpen == 0.0 else cfg.sharpen,
            dp=cfg.dp,
            level_gains=small_level_gains,
        )
        if verbose:
            print(f"  Small planet ({planet_frac*100:.1f}%): "
                  f"auto-adapting post-processing "
                  f"(wv=0, nlm={cfg.nlm}, sharpen={cfg.sharpen}, "
                  f"level_gains={small_level_gains}).")
    return cfg


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _format_params(params: dict) -> str:
    """Compact, alphabetically sorted parameter string for display."""
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def _finalize_result(
    image_data: np.ndarray,
    candidate,
    cached_result: np.ndarray | None,
    is_color: bool,
    do_post: bool,
    cfg: PostprocessConfig,
    rgb_jobs: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Produce the final (optionally post-processed) image for a candidate.

    If a cached result from re-ranking is available, it is returned
    directly.  Otherwise the candidate is deconvolved and optionally
    post-processed.
    """
    if cached_result is not None:
        return cached_result

    if is_color:
        if verbose:
            print("\nApplying best parameters to all 3 colour channels ...")
        result = apply_best_to_color(image_data, candidate, rgb_jobs=rgb_jobs)
        if do_post:
            if verbose:
                _print_postprocess_info(cfg)
            result = postprocess_rgb(result, cfg, jobs=rgb_jobs)
    else:
        result = candidate.result
        if do_post:
            if verbose:
                _print_postprocess_info(cfg)
            result = postprocess(result, cfg)
    return result


def _print_postprocess_info(cfg: PostprocessConfig) -> None:
    """Print post-processing pipeline summary to stdout."""
    parts = [f"wavelet(wv={cfg.wv})"]
    if cfg.sharpen > 1.0:
        parts.append(f"sharpen(gain={cfg.sharpen})")
    parts.append(f"NLM(h={cfg.nlm})")
    print(f"Post-processing: {' + '.join(parts)} + disk_preservation={cfg.dp} ...")


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------

def _load_and_prepare(
    fits_path: Path, args: argparse.Namespace, verbose: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load a FITS file and prepare the optimisation image.

    Returns (image_data, opt_image, is_color) where opt_image is the
    luminance channel for colour images or the raw image for mono.
    """
    if verbose:
        print(f"\nLoading: {fits_path}")
    image_data, is_color = load_fits_image(str(fits_path), channel=args.channel, verbose=verbose)
    if verbose:
        print(f"  Shape: {image_data.shape}   min={image_data.min():.1f}  max={image_data.max():.1f}")

    if is_color:
        opt_image = to_luminance(image_data)
        if verbose:
            print("  Using luminance channel for parameter optimization.")
    else:
        opt_image = image_data
    return image_data, opt_image, is_color


def _run_optimization(
    opt_image: np.ndarray, args: argparse.Namespace, verbose: bool,
) -> list:
    """Run Bayesian search and return sorted candidates."""
    if verbose:
        print(f"\nBayesian optimisation: {args.trials} trials (Optuna TPE, sequential).")
        print("Evaluating candidates ...")
    t0 = time.time()
    candidates = run_search(
        opt_image,
        n_trials=args.trials,
        progress_callback=_progress if verbose else None,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"Done in {elapsed:.1f} s  ({elapsed / 60:.1f} min)")
    return candidates


def _print_top_results(candidates: list, top_n: int) -> None:
    """Print a summary table of the top N candidates."""
    print(f"\nTop {top_n} results:")
    print(f"  {'#':>3}  {'Score':>7}  {'PSF':>10}  {'PSF params':>24}  {'Method':>15}  Deconv params")
    print("  " + "-" * 110)
    for i, c in enumerate(candidates[:top_n]):
        print(f"  {i+1:>3}  {c.normalised_score:>7.4f}  {c.psf_type:>10}  "
              f"{_format_params(c.psf_params):>24}  {c.deconv_method:>15}  "
              f"{_format_params(c.deconv_params)}")


def _save_all_results(
    image_data: np.ndarray,
    candidates: list,
    best,
    cached_best_result: np.ndarray | None,
    is_color: bool,
    do_post: bool,
    pp_cfg: PostprocessConfig,
    fits_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    verbose: bool,
) -> None:
    """Save best result, optional top-N, and optional plots."""
    stem = fits_path.stem
    extension = fits_path.suffix
    rgb_jobs = max(1, int(args.rgb_jobs))

    best_result = _finalize_result(
        image_data, best, cached_best_result, is_color,
        do_post, pp_cfg, rgb_jobs=rgb_jobs, verbose=verbose,
    )
    best_path = save_fits(best_result, best, stem, extension, output_dir, rank=1)
    print(f"\n{'Best RGB' if is_color else 'Best'} result saved: {best_path}")

    top_n = min(args.top, len(candidates))
    if args.save_fits:
        for i, c in enumerate(candidates[1:top_n]):
            result = _finalize_result(
                image_data, c, None, is_color,
                do_post, pp_cfg, rgb_jobs=rgb_jobs, verbose=False,
            )
            out_path = save_fits(result, c, stem, extension, output_dir, rank=i + 2)
            if verbose:
                print(f"  Saved rank {i+2}: {out_path}")

    if args.save_plots or args.show:
        _save_plots(candidates, image_data if not is_color else to_luminance(image_data),
                     top_n, stem, output_dir, args)


def _save_plots(
    candidates: list, opt_image: np.ndarray, top_n: int,
    stem: str, output_dir: Path, args: argparse.Namespace,
) -> None:
    """Generate and save/show result plots."""
    if args.show:
        matplotlib.use("TkAgg")

    plot_best_results(
        opt_image, candidates, top_n=top_n,
        save_path=str(output_dir / f"{stem}_best_results.png") if args.save_plots else None,
    )
    plot_metrics_heatmap(
        candidates, top_n=min(20, len(candidates)),
        save_path=str(output_dir / f"{stem}_metrics_heatmap.png") if args.save_plots else None,
    )
    if args.show:
        plt.show()
    else:
        plt.close("all")


def _process_one(fits_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Full pipeline for a single FITS file: load, optimise, post-process, save."""
    verbose = not args.quiet

    image_data, opt_image, is_color = _load_and_prepare(fits_path, args, verbose)
    candidates = _run_optimization(opt_image, args, verbose)

    top_n = min(args.top, len(candidates))
    if verbose and top_n < args.top:
        print(f"\nRequested top {args.top}, but only {top_n} candidates passed quality filters.")
    if verbose and top_n > 1:
        _print_top_results(candidates, top_n)

    do_post = not args.no_post
    pp_cfg = PostprocessConfig(wv=args.wv, nlm=args.nlm,
                               sharpen=args.sharpen, dp=args.dp)
    if do_post:
        pp_cfg = adapt_postprocessing(opt_image, pp_cfg, verbose=verbose)

    best = candidates[0]
    cached_best_result: np.ndarray | None = None
    if do_post and len(candidates) > 1:
        best, cached_best_result = rerank_candidates(
            candidates, image_data, opt_image, is_color,
            pp_cfg, rgb_jobs=max(1, int(args.rgb_jobs)), verbose=verbose,
        )

    if verbose:
        print(f"\nBest: PSF={best.psf_type} {best.psf_params}  "
              f"method={best.deconv_method} {best.deconv_params}  "
              f"score={best.normalised_score:.4f}")

    _save_all_results(
        image_data, candidates, best, cached_best_result, is_color,
        do_post, pp_cfg, fits_path, output_dir, args, verbose,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_fits_files(input_dir: Path) -> list[Path]:
    """Find and return sorted FITS files in a directory, or exit if none."""
    fits_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".fits", ".fit")
    )
    if not fits_files:
        sys.exit(f"No FITS files found in {input_dir}")
    return fits_files


def _process_files(
    fits_files: list[Path], output_dir: Path,
    args: argparse.Namespace, file_jobs: int, verbose: bool,
) -> None:
    """Process FITS files sequentially or in parallel."""
    if file_jobs == 1:
        for idx, fits_path in enumerate(fits_files, 1):
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"  [{idx}/{len(fits_files)}]  {fits_path.name}")
                print(f"{'=' * 70}")
            try:
                _process_one(fits_path, output_dir, args)
            except Exception as exc:
                print(f"\n  ERROR processing {fits_path.name}: {exc}")
    else:
        workers = min(file_jobs, len(fits_files))
        if verbose:
            print(f"\nProcessing files in parallel with {workers} workers.")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(_process_one, fp, output_dir, args): fp for fp in fits_files}
            done = 0
            for fut in as_completed(future_map):
                fits_path = future_map[fut]
                done += 1
                try:
                    fut.result()
                except Exception as exc:
                    print(f"\n  ERROR processing {fits_path.name}: {exc}")
                else:
                    if verbose:
                        print(f"  [{done}/{len(fits_files)}] done: {fits_path.name}")


def main() -> None:
    """CLI entry point: parse args, find FITS files, run pipeline."""
    args = parse_args()
    verbose = not args.quiet
    file_jobs = max(1, int(args.file_jobs))

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")

    output_dir = input_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_files = _find_fits_files(input_dir)
    if verbose:
        print(f"Found {len(fits_files)} FITS file(s) in {input_dir}")
        for f in fits_files:
            print(f"  {f.name}")

    if args.show and file_jobs > 1:
        print("WARNING: --show is not compatible with --file-jobs > 1. Falling back to file_jobs=1.")
        file_jobs = 1

    total_t0 = time.time()
    _process_files(fits_files, output_dir, args, file_jobs, verbose)
    total_elapsed = time.time() - total_t0

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"All done. {len(fits_files)} images processed in "
              f"{total_elapsed:.1f} s  ({total_elapsed / 60:.1f} min)")
        print(f"Outputs written to: {output_dir}/")


if __name__ == "__main__":
    main()
