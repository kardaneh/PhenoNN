"""Shared plotting style for the paper figures.

Mirrors the IPSL-AID conventions: DejaVu Sans, 12 pt base, 15 pt axis labels,
frameless 10 pt legends, linewidth 2.  Every figure is saved as PDF and 300 dpi
PNG.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["apply_style", "savefig", "PALETTE", "MODEL_COLORS", "annotate_metrics"]

PALETTE = {
    "blue": "#1f4e79",
    "orange": "#c14b30",
    "green": "#2e7d32",
    "purple": "#6a3d9a",
    "grey": "#555555",
    "lightgrey": "#bdbdbd",
    "teal": "#008b8b",
    "gold": "#b8860b",
}

MODEL_COLORS = {
    "lstm": PALETTE["blue"],
    "aelstm": PALETTE["orange"],
    "bitransformer": PALETTE["green"],
    "attnlstm": PALETTE["purple"],
    "climatology": PALETTE["grey"],
}

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply the shared rcParams."""
    plt.rcParams.update(_RC)


def savefig(fig, path: str) -> None:
    """Save ``fig`` as PDF and PNG, deriving the sibling path from ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    root, ext = os.path.splitext(path)
    if ext.lower() not in (".pdf", ".png"):
        root = path
    fig.savefig(root + ".pdf")
    fig.savefig(root + ".png", dpi=300)
    plt.close(fig)


def annotate_metrics(ax, m: dict, loc: str = "upper left", fontsize: int = 10) -> None:
    """Write the headline metrics in a corner of a scatter panel."""
    def f(key, fmt=".4f"):
        v = m.get(key)
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "n/a"

    lines = [
        f"R² = {f('overall_r2')}",
        f"RMSE = {f('rmse')}",
        f"MAE = {f('mae')}",
        f"bias = {f('bias', '+.4f')}",
        f"slope = {f('slope')}",
        f"N = {int(m.get('n_points', 0)):,}",
    ]
    ax.text(
        0.03 if "left" in loc else 0.97,
        0.97 if "upper" in loc else 0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )
