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

from analysis.figures.ipsl_aid import annotate_metrics, apply_ipsl_style

__all__ = ["apply_style", "savefig", "PALETTE", "MODEL_COLORS", "annotate_metrics"]

PALETTE = {
    "blue": "#1f77b4",
    "red": "#d62728",
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


def apply_style() -> None:
    """Apply the shared IPSL-AID rcParams."""
    apply_ipsl_style()


def savefig(fig, path: str) -> None:
    """Save ``fig`` as PDF and PNG, deriving the sibling path from ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    root, ext = os.path.splitext(path)
    if ext.lower() not in (".pdf", ".png"):
        root = path
    fig.savefig(root + ".pdf")
    fig.savefig(root + ".png", dpi=300)
    plt.close(fig)
