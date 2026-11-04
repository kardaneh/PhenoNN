# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Curated plotting primitives adapted from IPSL-AID diagnostics.

The original file lives in ``IPSL-AID/src/IPSL_AID/diagnostics.py``
(Copyright 2026 CNRS; authors Kazem Ardaneh, Kishanthan Kingston,
Pierre Chapel, Rosie Eade; CC BY-NC-SA 4.0).  Only the style configuration
and the generic hexbin / colourbar / line-style helpers are retained here;
the weather-map, ensemble and torch-specific code is intentionally not
vendored.

The helpers are deliberately small so the PhenoNN figure code can share the
same visual language without importing cartopy, torch or seaborn.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional

import matplotlib as mpl
import matplotlib.ticker as ticker

__all__ = [
    "IPSL_RC",
    "HEXBIN_CMAP",
    "MARKERS",
    "apply_ipsl_style",
    "hexbin_density",
    "add_log_count_colorbar",
    "identity_line",
    "annotate_metrics",
    "integer_epoch_axis",
    "linestyle_cycle",
    "next_style",
    "collect_handles",
    "figure_legend",
    "metrics_line",
    "figure_note",
    "model_label",
]

# Marker sequence used when a line plot needs markers as well as dash patterns.
# mpltex requires an explicit list: ``markers=None`` raises inside its
# ``itertools.product`` call.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Matched to IPSL-AID diagnostics.py, with mathtext kept on the DejaVu
# family so the PDF does not depend on an Arial installation.
IPSL_RC: Dict[str, object] = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "font.style": "normal",
    "font.weight": "normal",
    "lines.linewidth": 2,
    "lines.dashed_pattern": [4, 2],
    "lines.dashdot_pattern": [6, 3, 2, 3],
    "lines.dotted_pattern": [2, 3],
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 10,
    "legend.loc": "best",
    "legend.frameon": False,
    "text.color": "black",
    "text.usetex": False,
    "text.hinting": "auto",
    "text.antialiased": True,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# The reference diagnostics.py uses jet for density panels.
HEXBIN_CMAP = "jet"
IDENTITY_COLOR = "red"

_MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "aelstm": "AELSTM",
    "bitransformer": "BiTransformer",
    "bitransformer_v2": "BiTransformer",
    "bittransformer_v2": "BiTransformer",
    "attnlstm": "AttnLSTM",
    "climatology": "Climatology",
}


def apply_ipsl_style() -> None:
    """Apply the shared IPSL-AID rcParams."""
    mpl.rcParams.update(IPSL_RC)


def model_label(name: Optional[str]) -> str:
    """Return the publication spelling of an architecture name."""
    if not name:
        return ""
    key = str(name).lower()
    return _MODEL_LABELS.get(key, str(name))


def hexbin_density(
    ax,
    x,
    y,
    *,
    gridsize: int = 100,
    cmap: str = HEXBIN_CMAP,
    mincnt: int = 1,
    linewidths: float = 0.0,
    **kwargs,
):
    """Draw a log-scaled density hexbin and return the collection."""
    return ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        bins="log",
        cmap=cmap,
        mincnt=mincnt,
        linewidths=linewidths,
        **kwargs,
    )


def add_log_count_colorbar(
    fig,
    collection,
    ax,
    *,
    pad: float = 0.02,
    label: str = r"$\log_{10}[\mathrm{count}]$",
):
    """Add the IPSL-AID log-count colourbar."""
    cbar = fig.colorbar(collection, ax=ax, pad=pad)
    cbar.set_label(label)
    return cbar


def identity_line(ax, lo: float, hi: float, *, lw: float = 1.2):
    """Draw the red dashed 1:1 line used in the reference figures."""
    return ax.plot(
        [lo, hi],
        [lo, hi],
        color=IDENTITY_COLOR,
        linestyle="--",
        linewidth=lw,
        zorder=3,
    )


def annotate_metrics(
    ax,
    metrics: Optional[dict],
    *,
    keys: Optional[Iterable[str]] = None,
    loc: str = "upper left",
    fontsize: int = 10,
) -> None:
    """Write a plain, frame-less metric annotation in a corner.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    metrics : dict or None
        Metric mapping.  ``overall_r2`` / ``r2``, ``rmse``, ``mae``,
        ``bias``, ``slope`` and ``n_points`` / ``n`` are recognised.
    keys : iterable of str, optional
        Explicit keys, in display order.  Defaults to R², RMSE, MAE, bias,
        slope and N when present.
    loc : {"upper left", "upper right", "lower left", "lower right"}
    fontsize : int
    """
    if not metrics:
        return

    aliases = {
        "r2": "overall_r2",
        "n": "n_points",
        "N": "n_points",
    }
    labels = {
        "overall_r2": "R²",
        "rmse": "RMSE",
        "mae": "MAE",
        "bias": "bias",
        "slope": "slope",
        "n_points": "N",
    }
    formats = {
        "overall_r2": ".4f",
        "rmse": ".4f",
        "mae": ".4f",
        "bias": "+.4f",
        "slope": ".4f",
        "n_points": ",",
    }

    if keys is None:
        keys = ("overall_r2", "rmse", "mae", "bias", "slope", "n_points")

    lines = []
    for key in keys:
        canonical = aliases.get(key, key)
        if canonical not in labels:
            continue
        value = metrics.get(key, metrics.get(canonical))
        if value is None:
            continue
        try:
            if canonical == "n_points":
                text = f"{int(value):,}"
            else:
                text = format(float(value), formats[canonical])
        except (TypeError, ValueError):
            continue
        lines.append(f"{labels[canonical]} = {text}")

    if not lines:
        return

    left = "left" if "left" in loc else "right"
    top = "top" if "upper" in loc else "bottom"
    ax.text(
        0.03 if left == "left" else 0.97,
        0.97 if top == "top" else 0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=left,
        va=top,
        fontsize=fontsize,
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )


def integer_epoch_axis(ax) -> None:
    """Force integer tick labels on an epoch axis."""
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def linestyle_cycle(markers: bool = False):
    """Return an iterator of mpltex line kwargs.

    ``markers=True`` also cycles markers; mpltex needs an explicit marker
    list, so ``MARKERS`` is used instead of ``None``.
    """
    try:
        import mpltex

        return mpltex.linestyle_generator(markers=MARKERS if markers else [])
    except Exception:
        styles = [
            {"linestyle": "-"},
            {"linestyle": "--"},
            {"linestyle": "-."},
            {"linestyle": ":"},
        ]
        if markers:
            styles = [
                dict(style, marker=marker) for style, marker in zip(styles, MARKERS)
            ]
        return itertools.cycle(styles)


def next_style(cycle, *, color=None, marker=None):
    """Return one style dict from ``cycle``, with optional colour/marker override.

    ``color`` also sets ``mec``/``mfc``, so the marker matches the line.
    """
    style = dict(next(cycle))
    if color is not None:
        style["color"] = color
        style["mec"] = color
        style["mfc"] = color
    if marker is not None:
        style["marker"] = marker
    return style


def collect_handles(axes):
    """Collect legend handles/labels from one or more axes, without duplicates."""
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if not label or label.startswith("_"):
                continue
            if label in labels:
                continue
            handles.append(handle)
            labels.append(label)
    return handles, labels


_METRIC_LABELS = {
    "overall_r2": "R²",
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "bias",
    "slope": "slope",
    "n_points": "N",
}


def metrics_line(
    metrics,
    keys=("overall_r2", "rmse", "mae", "bias", "slope", "n_points"),
    *,
    sep: str = " · ",
) -> str:
    """Format metrics as one footnote line, e.g. ``R² = 0.93 · RMSE = 0.42``.

    Missing or non-numeric entries are skipped.
    """
    if not isinstance(metrics, dict):
        return ""
    parts = []
    for key in keys:
        value = metrics.get(key)
        label = _METRIC_LABELS.get(key, key)
        try:
            if key == "n_points":
                parts.append(f"{label} = {int(value):,}")
            elif key == "bias":
                parts.append(f"{label} = {float(value):+.4f}")
            else:
                parts.append(f"{label} = {float(value):.4f}")
        except (TypeError, ValueError):
            continue
    return sep.join(parts)


def figure_note(fig, text: str, *, fontsize: float = 8.5) -> None:
    """Write a footnote under the axes; constrained layout reserves the space."""
    if text:
        fig.supxlabel(text, fontsize=fontsize)


def figure_legend(
    fig,
    handles,
    labels,
    *,
    ncol: int = 3,
    loc: str = "outside lower center",
    fontsize: int = 9,
):
    """Place a multi-column legend outside the axes.

    The figure must use ``layout="constrained"`` so matplotlib reserves the
    space. ``ncol`` is clamped to at least 2.
    """
    if not handles:
        return None
    return fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=max(2, int(ncol)),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.6,
        columnspacing=1.4,
    )
