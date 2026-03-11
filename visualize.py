"""
Visualisation utilities for deconvolution results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from optimizer import Candidate


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _percentile_stretch(image: np.ndarray,
                        low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Stretch pixel values to [0, 1] using percentile clipping."""
    lo = np.percentile(image, low)
    hi = np.percentile(image, high)
    if hi == lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)


def _deconv_label(c: Candidate) -> str:
    """One-line summary of a candidate's deconvolution parameters."""
    p = c.deconv_params
    if "iterations" in p:
        return f"iter={p['iterations']} damp={p.get('damping', 0):.0e}"
    if "snr" in p:
        return f"snr={p['snr']}"
    if "regularization" in p:
        return f"λ={p['regularization']:.0e}"
    return str(p)


def _planet_crop_box(image: np.ndarray, margin: float = 0.15) -> tuple | None:
    """
    Return (r0, r1, c0, c1) cropping the image to the planet region
    with a margin, or None if the planet fills most of the frame.
    """
    from metrics import planet_mask
    mask = planet_mask(image)
    fill = mask.sum() / mask.size
    if fill > 0.20:
        return None  # planet fills the frame, no crop needed

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    # Add margin (fraction of bounding box size)
    h, w = image.shape[:2]
    rh = r1 - r0
    rw = c1 - c0
    pad_h = int(rh * margin)
    pad_w = int(rw * margin)
    r0 = max(0, r0 - pad_h)
    r1 = min(h, r1 + pad_h)
    c0 = max(0, c0 - pad_w)
    c1 = min(w, c1 + pad_w)
    return r0, r1, c0, c1


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def plot_best_results(
    original: np.ndarray,
    candidates: list[Candidate],
    top_n: int = 5,
    cmap: str = "gray",
    save_path: str | None = None,
) -> plt.Figure:
    """Show the original image alongside the top-N deconvolution results."""
    top = candidates[:top_n]
    ncols = top_n + 1  # original + results

    # For small planets, crop to planet region so detail is visible
    crop = _planet_crop_box(original)

    fig = plt.figure(figsize=(4 * ncols, 5), constrained_layout=True)
    gs = gridspec.GridSpec(2, ncols, figure=fig, height_ratios=[4, 1], hspace=0.05)

    def _show(ax, img, title):
        view = img[crop[0]:crop[1], crop[2]:crop[3]] if crop else img
        ax.imshow(_percentile_stretch(view), cmap=cmap, origin="upper", aspect="equal")
        ax.set_title(title, fontsize=7, pad=3)
        ax.axis("off")

    # Original
    _show(fig.add_subplot(gs[0, 0]), original, "Original")

    # Top candidates
    for col, c in enumerate(top, start=1):
        label = (
            f"#{col}  score={c.normalised_score:.3f}\n"
            f"PSF: {c.psf_type} fwhm={c.psf_params.get('fwhm', '?')}\n"
            f"Method: {c.deconv_method}\n"
            f"{_deconv_label(c)}"
        )
        _show(fig.add_subplot(gs[0, col]), c.result, label)

    # Score bar chart
    ax_bar = fig.add_subplot(gs[1, 1:])
    scores = [c.normalised_score for c in top]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scores)))
    ax_bar.barh(range(len(scores)), scores, color=colors)
    ax_bar.set_yticks(range(len(scores)))
    ax_bar.set_yticklabels([f"#{i+1}" for i in range(len(scores))], fontsize=7)
    ax_bar.set_xlabel("Normalised score", fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.tick_params(labelsize=6)

    fig.suptitle("Planetary Deconvolution — Best Results", fontsize=10, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Metrics heatmap
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(
    candidates: list[Candidate],
    top_n: int = 20,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of normalised metric values for the top-N candidates."""
    from metrics import _WEIGHTS

    top = candidates[:top_n]
    metric_names = list(_WEIGHTS.keys())

    data = np.array([
        [c.metrics.get(f"{m}_norm", 0.0) for m in metric_names]
        for c in top
    ])

    row_labels = [
        f"#{i+1} {c.psf_type}/{c.deconv_method} fwhm={c.psf_params.get('fwhm', '?')}"
        for i, c in enumerate(top)
    ]

    fig, ax = plt.subplots(
        figsize=(len(metric_names) * 1.8 + 2, max(4, top_n * 0.4 + 1)),
        constrained_layout=True,
    )
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Normalised metric scores (top candidates)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Quick single-image display
# ---------------------------------------------------------------------------

def plot_single(image: np.ndarray, title: str = "", cmap: str = "gray") -> plt.Figure:
    """Display a single image with percentile stretch."""
    fig, ax = plt.subplots()
    ax.imshow(_percentile_stretch(image), cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig
