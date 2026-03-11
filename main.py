"""
Planetary deconvolution optimizer â€” CLI entry point.

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

try:
    from astropy.io import fits
except ImportError:
    sys.exit("astropy is required.  Install with:\n    pip install astropy\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizer import run_search
from postprocess import postprocess, postprocess_rgb
from visualize import plot_best_results, plot_metrics_heatmap


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def load_fits_image(path: str, channel: int | None = None, verbose: bool = True) -> tuple[np.ndarray, bool]:
    """
    Load a FITS image and return (data, is_color).

    Returns (H, W) for mono or (3, H, W) for RGB.
    NaN/Inf pixels are replaced with the image median.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data

    # Fallback: search other HDUs if primary has no data
    if data is None:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break

    if data is None:
        raise ValueError(f"No image data found in {path}")

    data = data.astype(np.float64)

    # Sanitise invalid pixels
    bad = ~np.isfinite(data)
    if bad.any():
        data[bad] = np.nanmedian(data)

    # (3, H, W) â†’ treat as RGB unless a specific channel was requested
    if data.ndim == 3 and data.shape[0] == 3 and channel is None:
        if verbose:
            print(f"  RGB FITS detected ({data.shape}). Processing all 3 channels.")
        return data, True

    if data.ndim == 3:
        idx = channel if channel is not None else 0
        if idx < 0 or idx >= data.shape[0]:
            raise ValueError(
                f"Channel index {idx} out of range for FITS cube with shape {data.shape}."
            )
        if verbose:
            print(f"  3-D FITS cube ({data.shape}). Using channel {idx}.")
        return data[idx], False

    if data.ndim == 2:
        return data, False

    raise ValueError(f"Unsupported FITS data shape: {data.shape}")


def _to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) RGB to luminance (H, W) using BT.601 weights."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


# ---------------------------------------------------------------------------
# Output filename helpers
# ---------------------------------------------------------------------------

def _output_fits_name(stem: str, extension: str, rank: int) -> str:
    """
    Build output FITS names from the input filename.

    Rank 1 uses the exact requested convention: <input>_DC<ext>
    Additional ranks use <input>_DC_rN<ext> to avoid overwriting.
    """
    if rank == 1:
        return f"{stem}_DC{extension}"
    return f"{stem}_DC_r{rank}{extension}"


def _save_fits(
    data: np.ndarray,
    candidate,
    stem: str,
    extension: str,
    output_dir: Path,
    rank: int,
) -> Path:
    """Save a deconvolved result (mono or RGB) as FITS with metadata headers."""
    out_path = output_dir / _output_fits_name(stem, extension, rank)

    hdr = fits.Header()
    hdr["RANK"]     = rank
    hdr["SCORE"]    = round(candidate.normalised_score, 6)
    hdr["PSF_TYPE"] = candidate.psf_type
    hdr["FWHM"]     = candidate.psf_params.get("fwhm", -1)
    hdr["DMETHOD"]  = candidate.deconv_method

    fits.writeto(str(out_path), data.astype(np.float32), header=hdr, overwrite=True)
    return out_path


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _progress(done: int, total: int) -> None:
    """Print an in-place progress bar."""
    pct = done / total * 100
    bar_len = 40
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}", end="", flush=True)
    if done == total:
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
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
# Core processing
# ---------------------------------------------------------------------------

def _apply_best_to_color(rgb: np.ndarray, candidate, rgb_jobs: int = 1) -> np.ndarray:
    """Apply a candidate's PSF + deconvolution to each RGB channel independently."""
    from psf import build_psf
    from deconvolve import deconvolve

    psf = build_psf(candidate.psf_type, candidate.psf_params)

    n_channels = rgb.shape[0]
    workers = max(1, min(int(rgb_jobs), n_channels))

    def _run_channel(ch: int) -> np.ndarray:
        return deconvolve(candidate.deconv_method, rgb[ch], psf, candidate.deconv_params)

    if workers == 1:
        channels = [_run_channel(ch) for ch in range(n_channels)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            channels = list(ex.map(_run_channel, range(n_channels)))

    return np.stack(channels, axis=0)


def _format_params(params: dict) -> str:
    """Compact, alphabetically sorted parameter string."""
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def _process_one(fits_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Full pipeline for a single FITS file: load â†’ optimise â†’ post-process â†’ save."""
    verbose = not args.quiet
    rgb_jobs = max(1, int(args.rgb_jobs))

    # Load
    if verbose:
        print(f"\nLoading: {fits_path}")
    image_data, is_color = load_fits_image(str(fits_path), channel=args.channel, verbose=verbose)
    if verbose:
        print(f"  Shape: {image_data.shape}   min={image_data.min():.1f}  max={image_data.max():.1f}")

    # Optimise on luminance (2-D)
    if is_color:
        opt_image = _to_luminance(image_data)
        if verbose:
            print("  Using luminance channel for parameter optimization.")
    else:
        opt_image = image_data

    # Bayesian search
    if verbose:
        print(f"\nBayesian optimisation: {args.trials} trials (Optuna TPE, sequential).")
    t0 = time.time()
    if verbose:
        print("Evaluating candidates ...")
    candidates = run_search(
        opt_image,
        n_trials=args.trials,
        progress_callback=_progress if verbose else None,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"Done in {elapsed:.1f} s  ({elapsed / 60:.1f} min)")

    # Report top results
    top_n = min(args.top, len(candidates))
    if verbose and top_n < args.top:
        print(f"\nRequested top {args.top}, but only {top_n} candidates passed quality filters.")

    if verbose and top_n > 1:
        print(f"\nTop {top_n} results:")
        print(f"  {'#':>3}  {'Score':>7}  {'PSF':>10}  {'PSF params':>24}  {'Method':>15}  Deconv params")
        print("  " + "-" * 110)
        for i, c in enumerate(candidates[:top_n]):
            print(f"  {i+1:>3}  {c.normalised_score:>7.4f}  {c.psf_type:>10}  "
                  f"{_format_params(c.psf_params):>24}  {c.deconv_method:>15}  "
                  f"{_format_params(c.deconv_params)}")

    # Save results
    stem = fits_path.stem
    extension = fits_path.suffix
    do_post = not args.no_post
    best = candidates[0]
    cached_best_result: np.ndarray | None = None

    # Adapt post-processing for small planets: wavelet denoising destroys fine
    # detail on small targets.  Instead, use wavelet sharpening (frequency-
    # selective) to enhance medium-scale features (bands, ring gaps).
    wv_strength = args.wv
    nlm_strength = args.nlm
    sharpen_gain = args.sharpen
    if do_post:
        from metrics import planet_mask
        pmask = planet_mask(opt_image)
        planet_frac = pmask.sum() / pmask.size
        if planet_frac < 0.20:
            wv_strength = 0.0      # skip wavelet denoising (destroys detail)
            nlm_strength = min(nlm_strength, 0.003)  # very gentle NLM only
            if sharpen_gain == 0.0:
                sharpen_gain = 1.5  # auto-enable wavelet sharpening
            if verbose:
                print(f"  Small planet ({planet_frac*100:.1f}%): "
                      f"auto-adapting post-processing "
                      f"(wv=0, nlm={nlm_strength}, sharpen={sharpen_gain}).")

    # Re-rank top candidates by final (post-processed) quality, not only the
    # pre-post luminance optimisation score.
    # Use planet mask from the INPUT image consistently (same as optimizer).
    if do_post and len(candidates) > 1:
        from metrics import all_metrics, _WEIGHTS
        from optimizer import _INVERTED_METRICS
        rerank_n = min(20, len(candidates))
        rerank_pool = list(candidates[:rerank_n])
        seen_ids = {id(c) for c in rerank_pool}
        for c in candidates:
            if getattr(c, "source", "optuna") == "seed" and id(c) not in seen_ids:
                rerank_pool.append(c)
                seen_ids.add(id(c))

        if verbose:
            extra = len(rerank_pool) - rerank_n
            msg = f"\nRe-ranking top {rerank_n} candidates after post-processing preview"
            if extra > 0:
                msg += f" + {extra} seed candidate(s)"
            print(msg + " ...")

        # Phase 1: compute post-processed metrics for each candidate
        rerank_entries = []  # list of (candidate, trial_result, metrics)
        for idx, c in enumerate(rerank_pool, 1):
            if is_color:
                trial_result = _apply_best_to_color(image_data, c, rgb_jobs=rgb_jobs)
                trial_result = postprocess_rgb(
                    trial_result,
                    wavelet_threshold=wv_strength,
                    nlm_h=nlm_strength,
                    disk_preservation=args.dp,
                    sharpen_gain=sharpen_gain,
                    jobs=rgb_jobs,
                )
                lum = _to_luminance(trial_result)
            else:
                trial_result = postprocess(
                    c.result,
                    wavelet_threshold=wv_strength,
                    nlm_h=nlm_strength,
                    disk_preservation=args.dp,
                    sharpen_gain=sharpen_gain,
                )
                lum = trial_result

            metrics = all_metrics(lum, mask=pmask)
            rerank_entries.append((c, trial_result, metrics))

        # Phase 2: normalise metrics across the re-ranking pool (min-max),
        # same approach the optimizer uses, so all metrics contribute equally.
        for name in _WEIGHTS:
            values = np.array([m[name] for _, _, m in rerank_entries])
            lo, hi = values.min(), values.max()
            span = hi - lo if hi > lo else 1.0
            for _, _, m in rerank_entries:
                m[f"{name}_norm"] = (m[name] - lo) / span

        # Phase 3: pick the best using normalised weighted composite
        best_final_score = -np.inf
        for idx, (c, trial_result, metrics) in enumerate(rerank_entries, 1):
            final_score = 0.0
            for k, w in _WEIGHTS.items():
                v = metrics.get(f"{k}_norm", 0.0)
                if k in _INVERTED_METRICS:
                    v = 1.0 - v
                final_score += w * v

            if verbose:
                print(
                    f"  [{idx}] norm_score={final_score:.4f} "
                    f"qr={metrics['quality_ratio']:.1f} "
                    f"lap={metrics['laplacian_variance']:.0f} "
                    f"ten={metrics['tenengrad']:.0f}"
                )
            if final_score > best_final_score:
                best_final_score = final_score
                best = c
                cached_best_result = trial_result.astype(np.float32, copy=False)

        if verbose and best is not candidates[0]:
            print("  Re-rank selected a different best candidate.")

    if verbose:
        print(f"\nBest: PSF={best.psf_type} {best.psf_params}  "
              f"method={best.deconv_method} {best.deconv_params}  "
              f"score={best.normalised_score:.4f}")

    if is_color:
        if verbose:
            print("\nApplying best parameters to all 3 colour channels ...")
        color_result = cached_best_result
        if color_result is None:
            color_result = _apply_best_to_color(image_data, best, rgb_jobs=rgb_jobs)
        if do_post and cached_best_result is None:
            parts = [f"wavelet(wv={wv_strength})"]
            if sharpen_gain > 1.0:
                parts.append(f"sharpen(gain={sharpen_gain})")
            parts.append(f"NLM(h={nlm_strength})")
            if verbose:
                print(f"Post-processing: {' + '.join(parts)} + disk_preservation={args.dp} ...")
            color_result = postprocess_rgb(color_result, wavelet_threshold=wv_strength,
                                           nlm_h=nlm_strength, disk_preservation=args.dp,
                                           sharpen_gain=sharpen_gain, jobs=rgb_jobs)
        best_path = _save_fits(color_result, best, stem, extension, output_dir, rank=1)
        print(f"Best RGB result saved: {best_path}")
    else:
        mono_result = cached_best_result
        if mono_result is None:
            mono_result = best.result
        if do_post and cached_best_result is None:
            parts = [f"wavelet(wv={wv_strength})"]
            if sharpen_gain > 1.0:
                parts.append(f"sharpen(gain={sharpen_gain})")
            parts.append(f"NLM(h={nlm_strength})")
            if verbose:
                print(f"\nPost-processing: {' + '.join(parts)} + disk_preservation={args.dp} ...")
            mono_result = postprocess(
                mono_result,
                wavelet_threshold=wv_strength,
                nlm_h=nlm_strength,
                disk_preservation=args.dp,
                sharpen_gain=sharpen_gain,
            )
        best_path = _save_fits(mono_result, best, stem, extension, output_dir, rank=1)
        print(f"\nBest result saved: {best_path}")

    # Save additional top-N
    if args.save_fits:
        for i, c in enumerate(candidates[1:top_n]):
            if is_color:
                result = _apply_best_to_color(image_data, c, rgb_jobs=rgb_jobs)
                if do_post:
                    result = postprocess_rgb(result, wavelet_threshold=wv_strength,
                                             nlm_h=nlm_strength, disk_preservation=args.dp,
                                             sharpen_gain=sharpen_gain, jobs=rgb_jobs)
            else:
                result = c.result
                if do_post:
                    result = postprocess(result, wavelet_threshold=wv_strength,
                                         nlm_h=nlm_strength, disk_preservation=args.dp,
                                         sharpen_gain=sharpen_gain)
                    c.result = result
            out_path = _save_fits(result, c, stem, extension, output_dir, rank=i + 2)
            if verbose:
                print(f"  Saved rank {i+2}: {out_path}")

    # Figures are optional in default mode
    if args.save_plots or args.show:
        if args.show:
            matplotlib.use("TkAgg")

        plot_best_results(
            opt_image,
            candidates,
            top_n=top_n,
            save_path=str(output_dir / f"{stem}_best_results.png") if args.save_plots else None,
        )
        plot_metrics_heatmap(
            candidates,
            top_n=min(20, len(candidates)),
            save_path=str(output_dir / f"{stem}_metrics_heatmap.png") if args.save_plots else None,
        )

        if args.show:
            plt.show()
        else:
            plt.close("all")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    file_jobs = max(1, int(args.file_jobs))

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")

    output_dir = input_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".fits", ".fit")
    )

    if not fits_files:
        sys.exit(f"No FITS files found in {input_dir}")

    if verbose:
        print(f"Found {len(fits_files)} FITS file(s) in {input_dir}")
        for f in fits_files:
            print(f"  {f.name}")

    total_t0 = time.time()

    if args.show and file_jobs > 1:
        print("WARNING: --show is not compatible with --file-jobs > 1. Falling back to file_jobs=1.")
        file_jobs = 1

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
                continue
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

    total_elapsed = time.time() - total_t0
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"All done. {len(fits_files)} images processed in "
              f"{total_elapsed:.1f} s  ({total_elapsed / 60:.1f} min)")
        print(f"Outputs written to: {output_dir}/")


if __name__ == "__main__":
    main()

