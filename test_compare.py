"""
Compare deconvolution outputs against reference images.

For each test_images/imageN/:
  - input/  contains the original FITS
  - input/output/ contains the pipeline's deconvolved result (*_DC.fit)
  - reference/ contains a reference deconvolution from an external process

The pipeline PASSES for an image when its composite_score >= reference's score.
Overall PASS requires all images to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from metrics import all_metrics, composite_score, planet_mask, _WEIGHTS
from optimizer import _INVERTED_METRICS


TEST_DIR = Path(__file__).parent / "test_images"


def _load_fits(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
    if data is None:
        raise ValueError(f"No image data in {path}")
    arr = np.asarray(data, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = np.nanmedian(arr)
    return arr


def _to_luminance(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 3:
        return 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"Unsupported shape: {arr.shape}")


def main() -> None:
    if not TEST_DIR.is_dir():
        sys.exit(f"Test directory not found: {TEST_DIR}")

    image_dirs = sorted(d for d in TEST_DIR.iterdir() if d.is_dir())
    if not image_dirs:
        sys.exit("No image directories found in test_images/")

    all_pass = True
    results = []

    for img_dir in image_dirs:
        name = img_dir.name
        input_dir = img_dir / "input"
        output_dir = input_dir / "output"
        ref_dir = img_dir / "reference"

        # Find output file
        out_files = sorted(
            f for f in output_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".fit", ".fits")
        ) if output_dir.is_dir() else []

        if not out_files:
            print(f"[{name}] SKIP - no output found in {output_dir}")
            all_pass = False
            continue

        # Find reference file
        ref_files = sorted(
            f for f in ref_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".fit", ".fits")
        ) if ref_dir.is_dir() else []

        if not ref_files:
            print(f"[{name}] SKIP - no reference found in {ref_dir}")
            all_pass = False
            continue

        # Load and convert to luminance
        out_img = _to_luminance(_load_fits(out_files[0]))
        ref_img = _to_luminance(_load_fits(ref_files[0]))

        # Also load original input for context
        input_files = sorted(
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".fit", ".fits")
        )
        if input_files:
            orig_img = _to_luminance(_load_fits(input_files[0]))
            orig_metrics = all_metrics(orig_img)
            orig_score = composite_score(orig_img, metrics=orig_metrics)
        else:
            orig_metrics = None
            orig_score = None

        # Compute metrics
        out_metrics = all_metrics(out_img)
        ref_metrics = all_metrics(ref_img)

        # Proportional scoring: for each metric compute the ratio output/reference
        # (or ref/output for inverted metrics where lower is better).  The
        # weighted average of ratios gives a single number where >= 1.0 means
        # our output matches or exceeds the reference.
        weighted_ratio = 0.0
        for metric_name, weight in _WEIGHTS.items():
            ov, rv = out_metrics[metric_name], ref_metrics[metric_name]
            if metric_name in _INVERTED_METRICS:
                # Lower is better → ratio = ref/out (>1 when ours is lower)
                ratio = rv / ov if ov > 1e-30 else 1.0
            else:
                ratio = ov / rv if rv > 1e-30 else 1.0
            weighted_ratio += weight * ratio

        out_score = weighted_ratio
        ref_score = 1.0  # reference is always 1.0 by definition

        passed = out_score >= ref_score
        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        pct = (out_score / ref_score * 100) if ref_score != 0 else 0

        results.append({
            "name": name,
            "status": status,
            "out_score": out_score,
            "ref_score": ref_score,
            "pct": pct,
            "out_metrics": out_metrics,
            "ref_metrics": ref_metrics,
            "orig_score": orig_score,
            "orig_metrics": orig_metrics,
            "out_file": out_files[0].name,
            "ref_file": ref_files[0].name,
        })

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS: Deconvolution Output vs Reference")
    print("=" * 80)

    for r in results:
        print(f"\n--- {r['name']} [{r['status']}] ---")
        print(f"  Output:    {r['out_file']}")
        print(f"  Reference: {r['ref_file']}")
        print(f"  Score: {r['out_score']:.4f} vs ref {r['ref_score']:.4f}  "
              f"({r['pct']:.1f}% of reference)")
        if r['orig_score'] is not None:
            print(f"  Original input score: {r['orig_score']:.4f}")
        print(f"  Metric breakdown:")
        print(f"    {'Metric':<22s} {'Output':>12s} {'Reference':>12s} {'Ratio':>8s}")
        print(f"    {'-'*56}")
        for k in sorted(r['out_metrics'].keys()):
            ov = r['out_metrics'][k]
            rv = r['ref_metrics'][k]
            ratio = ov / rv * 100 if rv != 0 else 0
            print(f"    {k:<22s} {ov:>12.2f} {rv:>12.2f} {ratio:>7.1f}%")

    # Summary
    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_total = len(results)
    print(f"\n{'=' * 80}")
    overall = "PASS" if all_pass else "FAIL"
    print(f"OVERALL: {overall}  ({n_pass}/{n_total} images passed)")
    print(f"{'=' * 80}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
