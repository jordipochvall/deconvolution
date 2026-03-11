# Planetary Deconvolution Optimizer

Automatic optimization of planetary FITS deconvolution using:
- Bayesian search (Optuna TPE)
- adaptive PSF and method parameters
- optional post-processing (wavelet + NLM)

The pipeline is designed for both large targets (for example Jupiter) and small targets (for example Saturn).

## What It Does

For each FITS image in an input directory:
1. Load image data (mono, cube channel, or RGB cube).
2. Detect the planet region with an adaptive mask.
3. Run sequential Bayesian optimization over PSF + deconvolution parameters.
4. Rank candidates with no-reference quality metrics.
5. Re-rank top 20 candidates after post-processing using the input planet mask.
6. Save best result (and optional top-N), plus optional summary figures.

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:
- `numpy`
- `scipy`
- `astropy`
- `matplotlib`
- `optuna`
- `PyWavelets`

Optional:
- `scikit-image` (for NLM denoising in post-processing)

## Usage

```bash
# Process all FITS files in a directory (recommended defaults)
python main.py images/

# More optimization budget
python main.py images/ --trials 150

# Parallel files and RGB channels
python main.py images/ --file-jobs 2 --rgb-jobs 3

# Skip post-processing
python main.py images/ --no-post

# Custom post-processing
python main.py images/ --wv 15 --nlm 0.005 --dp 0.6 --sharpen 1.5

# Save full top-N as FITS (best is always saved)
python main.py images/ --top 5 --save-fits

# Save summary plots (disabled by default)
python main.py images/ --save-plots

# Minimal console output
python main.py images/ --quiet

# Select channel from generic 3D cube
python main.py images/ --channel 1
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `input_dir` | required | Directory containing `.fits` or `.fit` files |
| `--channel N` | `None` | Channel index for 3D cubes; omitted means RGB auto-detect for `(3,H,W)` |
| `--top N` | `1` | Number of ranked results to report/save |
| `--trials N` | `100` | Number of Optuna trials |
| `--file-jobs N` | `1` | Parallel worker threads across FITS files |
| `--rgb-jobs N` | `1` | Parallel worker threads across RGB channels |
| `--save-fits` | off | Save top-N FITS outputs (rank #1 always saved) |
| `--save-plots` | off | Save `best_results` and `metrics_heatmap` PNG files |
| `--quiet` | off | Reduce console output to essential messages |
| `--show` | off | Show figures interactively |
| `--no-post` | off | Disable post-processing |
| `--wv FLOAT` | `25.0` | Wavelet denoise threshold (`0` disables) |
| `--nlm FLOAT` | `0.008` | NLM strength (`0` disables) |
| `--dp FLOAT` | `0.5` | Disk preservation factor for adaptive denoising |
| `--sharpen FLOAT` | `0.0` | Wavelet sharpening gain (`0` means auto on small planets) |

## Output

Outputs are written to:
- `<input_dir>/output/`

Per input image:
- `NAME_DC.<ext>` (best candidate, always)
- `NAME_DC_r2.<ext>`, `NAME_DC_r3.<ext>`, ... (rank 2..N, when `--save-fits`)
- `NAME_best_results.png` (only with `--save-plots`)
- `NAME_metrics_heatmap.png` (only with `--save-plots`)

FITS headers include:
- `RANK`
- `SCORE`
- `PSF_TYPE`
- `FWHM`
- `DMETHOD`

## Optimization Strategy

The optimizer uses Optuna TPE with adaptive bounds:
- large planets: broader, more aggressive search
- small planets: safer search to avoid over-deconvolution

Current runtime improvements:
- precomputed RL masks reused across trials
- sequential TPE trials for maximum Bayesian learning quality
- noise floor filter rejects candidates noisier than the input
- sharpness floor excludes barely-sharpened candidates from ranking
- optional parallel file processing with `--file-jobs`
- optional parallel RGB channel processing with `--rgb-jobs`
- reduced per-candidate memory (`float32` result storage)

## Quality Metrics

No-reference metrics are computed on the detected planet mask:
- `tenengrad`
- `laplacian_variance` (inverted in normalized scoring)
- `brenner`
- `smoothness`
- `quality_ratio`
- `normalised_power_hf`

Candidates are ranked by weighted normalized composite score.

## Deconvolution Methods

- `richardson_lucy`
  - FFT-based RL
  - optional TV regularization
  - deringing and limb suppression
- `wiener`
  - frequency-domain Wiener filter
- `tikhonov`
  - regularized inverse filter

## Code Structure

```text
deconvolution/
  main.py          # CLI entry point, RGB support, post-processing
  optimizer.py     # Optuna TPE search, candidate ranking
  deconvolve.py    # RL with TV reg + deringing, Wiener, Tikhonov
  psf.py           # Gaussian, Moffat, Airy PSF generation
  metrics.py       # No-reference sharpness metrics + planet mask
  postprocess.py   # Wavelet soft-thresholding + NLM post-processing
  visualize.py     # Summary plots (best results, metrics heatmap)
  test_compare.py  # Compare pipeline output vs reference images
  tests/
    test_project.py      # Unit tests
    test_integration.py  # Full pipeline vs reference (real images)
  test_images/           # Test data (input + reference per image)
  requirements.txt
```

## Tests

Run unit tests:

```bash
python -m pytest tests/test_project.py -v
```

Run integration tests (requires `test_images/` with real data):

```bash
python -m pytest tests/test_integration.py -v -s
```

Run all tests:

```bash
python -m pytest tests/ -v
```

Unit tests cover:
- PSF generation and dispatch
- deconvolution output shape/non-negativity
- mask-aware deconvolution parameters
- FITS loading behavior and channel bounds
- optimizer ranking
- post-processing shape/no-op behavior

Integration tests cover:
- full pipeline (100 trials) on real planetary images
- comparison against external reference deconvolutions (e.g. PixInsight)
- pass criterion: weighted metric ratio >= 100% of reference
