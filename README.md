# Planetary Image Deconvolution Pipeline

Automated deconvolution of planetary FITS images using Bayesian optimisation
(Optuna TPE) to find optimal PSF + algorithm parameters, ranked by
no-reference sharpness metrics. Achieves results comparable to or better than
PixInsight on Jupiter and Saturn images.

## Quick start

```bash
# Install dependencies
pip install numpy scipy astropy pywt optuna scikit-image matplotlib

# Run on a directory of FITS files
python main.py images/

# Custom settings
python main.py images/ --trials 150 --top 5 --save-fits
python main.py images/ --wv 15 --nlm 0.005      # custom post-processing
python main.py images/ --no-post                 # skip post-processing
```

## How it works

The pipeline processes each FITS file through five stages:

```
Input FITS → 1. PSF modelling → 2. Bayesian search → 3. Re-ranking → 4. Post-processing → Output FITS
```

### Stage 1: PSF modelling (`psf.py`)

The Point Spread Function models atmospheric seeing — the blur introduced by
Earth's atmosphere. Three models are available:

| Model    | Use case                | Parameters                    |
|----------|-------------------------|-------------------------------|
| Gaussian | Simple seeing           | `fwhm`, `size`                |
| Moffat   | Standard atmospheric    | `fwhm`, `beta`, `size`        |
| Airy     | Diffraction-limited     | `diameter_pixels`, `size`     |

**Moffat** is the primary model: it has heavier tails than Gaussian, matching
real atmospheric turbulence. The `beta` parameter (1.5–6.0) controls how
quickly the PSF wings fall off — lower values model worse seeing.

All PSFs are normalised to unit sum and forced to odd dimensions for centred
FFT convolution.

### Stage 2: Bayesian search (`optimizer.py`)

Instead of brute-force grid search (600+ combinations, ~60 min), Optuna's
Tree-structured Parzen Estimator (TPE) samples the parameter space
intelligently: it builds a model of which regions produce good results and
focuses exploration there.

**Search space** (adapted for planet size):

| Parameter         | Large planet (>20%)    | Small planet (<20%)    |
|-------------------|------------------------|------------------------|
| PSF FWHM          | 1.0–5.0 px             | 1.0–1.5 px             |
| RL iterations     | 15–300                 | 2–5                    |
| RL damping        | 1e-4–1e-2 (log)        | 1e-4–1e-2 (log)        |
| TV lambda         | 0.0–0.3                | 0.25–0.3               |
| Contrast boost    | 1.0–1.5                | 1.0                    |
| Limb suppression  | 0.85                   | 0.3                    |

**Quality filters** reject bad candidates early:
- **Noise floor**: smoothness < 55% of input → rejected (too noisy).
- **Sharpness floor**: tenengrad < 1.5× input → excluded from ranking
  (barely sharpened, wastes a ranking slot).

**MedianPruner** provides 8.5× speedup: RL iterations are split into 3
segments with intermediate tenengrad checks. Unpromising trials are stopped
early, saving compute on doomed candidates.

**Seed candidates**: 5 pre-defined parameter sets (domain knowledge) are
evaluated alongside Optuna trials to anchor the search around known-good
regions and ensure diverse coverage.

**Budget**: 100 trials ≈ 8 min (vs 600+ brute-force ≈ 60 min).

### Stage 2a: Deconvolution algorithms (`deconvolve.py`)

Three deconvolution methods compete in the search:

#### Richardson-Lucy with TV regularisation (primary)

The iterative RL algorithm models the imaging process (blur + Poisson noise)
and iteratively refines an estimate of the true scene. Each iteration:

1. **Forward model**: convolve current estimate with PSF (FFT-based).
2. **Ratio**: observed / forward model.
3. **Back-projection**: correlate ratio with PSF transpose.
4. **Corrections applied** in sequence:
   - **Deringing mask**: dampen corrections in dark sky regions (prevents
     Gibbs ringing from the planet boundary propagating outward).
   - **Limb clamping**: dampen corrections toward 1.0 at the planet limb
     (prevents Gibbs overshoot from accumulating over iterations).
5. **Multiplicative update**: estimate × correction.
6. **TV diffusion**: one step of anisotropic total variation smoothing
   (edge-preserving noise suppression each iteration, stable at 100–200
   iterations without noise explosion).

Optional Nesterov momentum (`use_nesterov=True`) accelerates convergence
with a k/(k+3) annealing schedule.

#### Wiener filter

Single-pass frequency-domain filter: `F̂ = H* / (|H|² + 1/SNR²) · G`.
Fast but cannot model Poisson noise correctly. Higher SNR → sharper but
noisier. Useful as a quick baseline.

#### Tikhonov (regularised inverse)

Similar to Wiener but with a direct regularisation parameter λ instead of
SNR. `F̂ = H* / (|H|² + λ) · G`. Higher λ → smoother result.

### Stage 2b: Sharpness metrics (`metrics.py`)

Six no-reference metrics evaluate each deconvolution result. All metrics are
computed on the **planet disk only** (detected via adaptive sky-statistics
thresholding) to avoid background sky from skewing results:

| Metric                | Weight | Direction | What it measures                           |
|-----------------------|--------|-----------|---------------------------------------------|
| `tenengrad`           | 30%    | Higher=better | Mean squared Sobel gradient (edge energy)  |
| `laplacian_variance`  | 25%    | **Lower=better** | Variance of Laplacian (ringing indicator)  |
| `brenner`             | 20%    | Higher=better | Horizontal gradient energy (focus measure) |
| `smoothness`          | 15%    | Higher=better | Structure-to-noise ratio (clean ≈ 1.0)     |
| `quality_ratio`       | 5%     | Higher=better | tenengrad/√(laplacian_var) — clean sharpening |
| `normalised_power_hf` | 5%     | Higher=better | High-frequency spectral power fraction     |

**`quality_ratio`** is the key balance metric: it rewards sharpening
(tenengrad↑) that does not produce ringing (laplacian_variance↓).

**Scoring**: candidates are ranked by pool-wide min-max normalised weighted
composite. `laplacian_variance` is inverted (lower raw value = higher score).
This normalisation ensures that no single metric dominates.

**Planet mask**: adaptive threshold at `sky_level + 5σ` (estimated from 25th
percentile). Critical for small planets like Saturn (~6% of frame) where a
fixed percentile would include sky pixels.

### Stage 3: Re-ranking (`main.py: rerank_candidates`)

The top 20 Optuna candidates (plus seed candidates) are re-evaluated **after
full post-processing**. This is important because Optuna scores are computed
on raw RL output, but the final image includes wavelet denoising + NLM which
can change the relative ranking.

Each candidate gets:
1. Full RGB deconvolution (if colour image).
2. Full post-processing pipeline.
3. Re-scored with normalised weighted composite on the post-processed result.

### Stage 4: Post-processing (`postprocess.py`)

Applied after RL to reduce residual noise and enhance planetary detail:

```
Deconvolved → Limb deringing → Wavelet denoising → Wavelet sharpening → NLM denoising → Output
```

#### 4.1 Limb deringing (optional)

Iterated median filtering in a narrow annulus around the planet limb.
Median filters eliminate oscillatory Gibbs artifacts while preserving the
sharp limb edge (idempotent on monotonic regions).

#### 4.2 Wavelet soft-thresholding

Stationary Wavelet Transform (SWT) decomposition into frequency layers.
Detail coefficients are soft-thresholded with geometrically decreasing
strength: fine scales (noise-dominated) get full thresholding, coarse scales
(real features) get almost none.

Noise level estimated via MAD (Median Absolute Deviation) of the finest
level — robust to outliers.

**Critical for large planets** (Jupiter): `wv=25` removes RL residual noise
while preserving cloud bands. Without it, results are ~80% of PixInsight
quality.

**Destroys small planet detail**: even `wv=5` crashes Saturn's quality_ratio
to 61% of PI. Auto-disabled for small planets (<20% of frame).

#### 4.3 Wavelet sharpening

Amplifies **medium-scale** SWT detail coefficients (planetary bands, ring
gaps, cloud features) with a bell-curve gain profile:
- Finest scale (noise) → gain = 1.0 (untouched).
- Medium scales (detail) → gain = user-specified (1.5–2.0).
- Coarsest scale (shape) → gain = 1.0 (untouched).

Auto-enabled for small planets at `gain=1.5` when wavelet denoising is
disabled.

#### 4.4 Non-Local Means denoising (NLM)

Patch-based denoising (scikit-image) that preserves edges by averaging
similar patches across the image. Applied after wavelet processing.

#### Adaptive disk preservation

All denoising steps use a spatial blend mask:
- Planet disk interior → lighter denoising (preserve detail).
- Limb and sky → full denoising (suppress artifacts).

Controlled by `dp` (disk preservation): 0.0 = uniform everywhere,
0.5 = disk gets 50% less denoising, 1.0 = disk untouched.

#### Small planet adaptation (`adapt_postprocessing`)

For planets covering <20% of the frame (e.g. Saturn):
- Wavelet denoising disabled (`wv=0`).
- NLM limited to `h=0.003` (very gentle).
- Wavelet sharpening auto-enabled (`gain=1.5`).

### Stage 5: RGB handling (`main.py`)

For colour FITS images (3, H, W):

1. **Optimise on luminance**: the Bayesian search runs on
   `L = 0.299·R + 0.587·G + 0.114·B` (BT.601) to find one set of optimal
   parameters.

2. **Apply to each channel**: the best PSF + deconvolution parameters are
   applied independently to R, G, B.

3. **White balance correction** (`_correct_white_balance`): RL with
   identical parameters on channels with different SNR (Bayer: R and B have
   lower SNR than G) causes differential noise amplification → colour cast.
   Each channel is scaled so its mean ratio to luminance on the planet disk
   matches the input.

4. **Global contrast boost** (`_apply_global_contrast_boost`): instead of
   stretching each channel independently (which shifts colour), a single
   global linear transform (scale + offset) is derived from the target
   luminance and applied equally to all channels, preserving colour ratios.

## Configuration (`PostprocessConfig`)

Post-processing parameters are grouped in a `PostprocessConfig` dataclass:

```python
from postprocess import PostprocessConfig

cfg = PostprocessConfig(
    wv=25.0,       # wavelet threshold (0 = disabled)
    nlm=0.008,     # NLM strength (0 = disabled)
    sharpen=0.0,   # wavelet sharpening gain (0 = auto, 1.5 = moderate)
    dp=0.5,        # disk preservation (0 = uniform, 1 = skip disk)
)
```

## Project structure

```
main.py              CLI entry point, RGB pipeline, re-ranking
optimizer.py         Optuna TPE search, seed candidates, scoring
deconvolve.py        RL+TV, Wiener, Tikhonov deconvolution
postprocess.py       Wavelet + NLM post-processing, PostprocessConfig
metrics.py           Sharpness metrics, planet mask, weights
psf.py               Gaussian, Moffat, Airy PSF models
wavelet_utils.py     Shared SWT padding helpers
visualize.py         Result plots and metrics heatmaps

tests/
  test_project.py          Unit tests (32)
  test_characterization.py Characterization tests (21) — refactoring safety net
  test_refactored.py       Tests for extracted/refactored functions (62)
  test_integration.py      Full pipeline vs reference images (4)

test_images/               Real test images with input/ and reference/
  image1/                  Jupiter
  image2/                  Jupiter
  image3/                  Jupiter
  image4/                  Saturn
```

## Test suite

```bash
# Fast tests (~2 s)
python -m pytest tests/test_project.py tests/test_characterization.py tests/test_refactored.py -v

# Full pipeline integration tests (~55 min, needs test_images/)
python -m pytest tests/test_integration.py -v -s

# Integration tests in parallel (requires pytest-xdist: pip install pytest-xdist)
python -m pytest tests/test_integration.py -v -n 4

# All tests
python -m pytest tests/ -v
```

**115 fast tests** + **4 integration tests** = 119 total.

## Results

Tested against PixInsight reference deconvolutions:

| Image  | Target  | Score vs reference |
|--------|---------|-------------------|
| image1 | Jupiter | 110.7%            |
| image2 | Jupiter | 100.3%            |
| image3 | Jupiter | 111.7%            |
| image4 | Saturn  | 135.9%            |

All 4 images pass (score >= 100% of reference).

## Output

Outputs are written to `<input_dir>/output/`:

- `NAME_DC.<ext>` — best candidate (always saved).
- `NAME_DC_r2.<ext>`, `NAME_DC_r3.<ext>`, ... — rank 2..N (with `--save-fits`).
- `NAME_best_results.png` — visual comparison (with `--save-plots`).
- `NAME_metrics_heatmap.png` — metric scores grid (with `--save-plots`).

FITS headers include: `RANK`, `SCORE`, `PSF_TYPE`, `FWHM`, `DMETHOD`.

## CLI reference

```
python main.py <input_dir> [options]

Positional:
  input_dir          Directory containing FITS files to process

Options:
  --channel N        Channel index for 3-D FITS cubes (omit for auto RGB)
  --top N            Number of top candidates to report/save (default: 1)
  --trials N         Number of Optuna trials (default: 100)
  --file-jobs N      Parallel workers across files (default: 1)
  --rgb-jobs N       Parallel workers across RGB channels (default: 1)
  --show             Display figures interactively
  --save-plots       Save summary plots (PNG)
  --save-fits        Save top-N results as FITS files
  --quiet            Reduce console output
  --no-post          Skip wavelet+NLM post-processing
  --wv FLOAT         Wavelet threshold (default: 25.0, 0 = disabled)
  --nlm FLOAT        NLM strength (default: 0.008, 0 = disabled)
  --dp FLOAT         Disk preservation (default: 0.5)
  --sharpen FLOAT    Wavelet sharpening gain (default: 0.0 = auto)
```

## Key design decisions

- **Optuna TPE** over grid search: 100 trials ≈ 8 min vs 648+ ≈ 60 min.
- **TV regularisation inside RL**: allows 100–200 iterations without noise explosion.
- **Wavelet as post-processing only**: wavelet inside the RL loop was 92 min and worse quality.
- **Post-processing is critical**: without it, results are ~80% of PI. With `wv=25, nlm=0.008`, results reach ~100–110%.
- **Adaptive planet mask**: sky-statistics threshold instead of fixed percentile, essential for small targets.
- **Sequential Optuna** (`n_jobs=1`): parallel TPE with `n_jobs=80` made all 100 trials random (`n_startup_trials=15 < 80`). Sequential gives maximum TPE quality.

## Requirements

- Python 3.10+
- numpy, scipy, astropy, pywt, optuna, matplotlib
- scikit-image (optional, for NLM denoising)
