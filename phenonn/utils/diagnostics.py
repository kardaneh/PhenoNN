# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

# PhenoCam diagnostic plots
#
# Three figures for monitoring training and evaluating results:
#   - plot_loss_histories:   train/val loss curve over epochs
#   - plot_metric_histories: multi-metric panel (R², RMSE, ...) over epochs
#   - plot_pred_vs_obs:      predicted vs observed hexbin scatter with R² / RMSE

import math
import os
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as mpl
import mpltex
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import pandas as pd
from phenonn.utils.config import (
    DYNAMIC_FEATURES,
    CYCLIC_FEATURES,
    PFT_COLS,
    LOG_TRANSFORM_FEATURES,
)
# `add_derived_features` is now an xarray helper. The few diagnostic plots
# that need it import it lazily inside the function body to keep this
# module light when only basic plots are needed.

params = {
    "font.family": "DejaVu Sans",
    #    'figure.dpi': 300,
    #    'savefig.dpi': 300,
    "lines.linewidth": 2,
    "lines.dashed_pattern": [4, 2],
    "lines.dashdot_pattern": [6, 3, 2, 3],
    "lines.dotted_pattern": [2, 3],
    "mathtext.rm": "arial",
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 15,
    "legend.loc": "best",
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
mpl.update(params)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _compute_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, bias, and R² on 1-D arrays. Ignores NaNs."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]
    if len(pred) == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "n": 0}

    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "n": len(pred)}


def _save(fig, filename: str, logger=None):
    """Save a figure, create parent directories if needed, and log."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    msg = f"Saved plot to {filename}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ── 1. Loss history ──────────────────────────────────────────────────────────
def plot_loss_histories(
    train_loss: Sequence[float],
    valid_loss: Sequence[float],
    filename: str = "loss_history.png",
    logger=None,
    log_scale: bool = True,
    title: str = "Training / Validation Loss",
) -> None:
    """
    Plot training and validation loss curves over epochs.

    Parameters
    ----------
    train_loss : sequence of float
        Per-epoch training loss.
    valid_loss : sequence of float
        Per-epoch validation loss (same length as train_loss).
    filename : str
        Output path.
    logger : phenonn.utils.logger.Logger, optional
        If provided, log the save location instead of printing.
    log_scale : bool
        Use log y-axis (default True — losses usually span orders of magnitude).
    title : str
        Figure title.

    Notes
    -----
    - Uses mpltex linestyles for consistent styling
    - Gray vertical dashed line: best validation epoch
    - Includes grid for better readability
    """

    epochs = np.arange(1, len(train_loss) + 1)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    # Generate linestyles
    linestyles = mpltex.linestyle_generator(markers=[])

    # Plot training loss
    ax.plot(epochs, train_loss, label="train", **next(linestyles))

    # Plot validation loss
    ax.plot(epochs, valid_loss, label="validation", **next(linestyles))

    # Set y-scale (log or linear)
    if log_scale and min(min(train_loss), min(valid_loss)) > 0:
        ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)

    # Grid for better readability
    ax.grid(True)

    # Mark the best validation epoch with a vertical dashed line
    best_epoch = int(np.argmin(valid_loss)) + 1
    ax.axvline(best_epoch, label=f"Best epoch: {best_epoch}", **next(linestyles))

    # Legend
    ax.legend()

    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    # Log or print the save location
    if logger:
        logger.info(f"Saved loss history plot to {filename}")
    else:
        print(f"Saved loss history plot to {filename}")


def plot_metric_histories(
    train_history: Dict[str, Sequence[float]],
    valid_history: Dict[str, Sequence[float]],
    filename: str = "metric_history.png",
    logger=None,
    log_metrics: Optional[Sequence[str]] = None,
    cols: int = 3,
) -> None:
    """
    Multi-panel figure of metric evolution over epochs.

    One panel per metric. Train and validation plotted together on each panel.

    Parameters
    ----------
    train_history : dict
        {metric_name: per-epoch values}. Example: {"rmse": [...], "r2": [...]}.
    valid_history : dict
        Same keys as train_history, same lengths.
    filename : str
        Output path.
    logger : phenonn.utils.logger.Logger, optional
    log_metrics : sequence of str, optional
        Metric names that should use log y-scale. Default: all except R²-like
        metrics (anything with "r2" or "r²" in the name).
    cols : int
        Number of columns in the grid.

    Notes
    -----
    - Uses mpltex linestyles for consistent styling
    - Missing/NaN values in a metric series are skipped — useful when you record
      metrics conditionally and some epochs lack certain values.
    - Includes grid for better readability
    """

    metric_names = list(train_history.keys())
    if not metric_names:
        raise ValueError("train_history is empty — nothing to plot")
    if set(valid_history.keys()) != set(metric_names):
        raise ValueError(
            f"train/valid histories must have same keys. "
            f"Train: {sorted(metric_names)}. "
            f"Valid: {sorted(valid_history.keys())}"
        )

    if log_metrics is None:
        log_metrics = [m for m in metric_names]

    n = len(metric_names)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(5 * cols, 4 * rows), tight_layout=True)
    gs = gridspec.GridSpec(rows, cols)

    for i, name in enumerate(metric_names):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])

        # Generate linestyles for this subplot
        linestyles = mpltex.linestyle_generator(markers=[])

        tr = np.asarray(train_history[name], dtype=float)
        vl = np.asarray(valid_history[name], dtype=float)
        ep = np.arange(1, max(len(tr), len(vl)) + 1)

        # Skip NaNs — plot only finite values
        tr_mask = np.isfinite(tr)
        vl_mask = np.isfinite(vl)

        ax.plot(ep[: len(tr)][tr_mask], tr[tr_mask], label="train", **next(linestyles))
        ax.plot(ep[: len(vl)][vl_mask], vl[vl_mask], label="valid", **next(linestyles))

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name.replace("_", " ").upper())

        # Log-scale only if strictly positive and metric in log_metrics
        all_vals = np.concatenate([tr[tr_mask], vl[vl_mask]])
        if name in log_metrics and len(all_vals) > 0 and all_vals.min() > 0:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True)

    # Hide unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    # Save the figure
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Log or print the save location
    if logger:
        logger.info(f"Saved metric history plot to {filename}")
    else:
        print(f"Saved metric history plot to {filename}")


# ── 3. Predicted vs observed hexbin ──────────────────────────────────────────


def plot_pred_vs_obs(
    pred: Union[np.ndarray, Sequence[float]],
    obs: Union[np.ndarray, Sequence[float]],
    filename: str = "pred_vs_obs.png",
    logger=None,
    title: str = "Predicted vs Observed LAI",
    xlabel: str = "Observed LAI",
    ylabel: str = "Predicted LAI",
    gridsize: int = 40,
    hexbin: bool = True,
) -> Dict[str, float]:
    """
    Scatter plot of predictions vs observations with y=x reference and metrics.

    Uses hexbin density by default (fast, readable for >10k points). Falls back
    to a regular scatter for small sample sets where hexbin would be mostly empty.

    Parameters
    ----------
    pred : array-like
        Predicted values (any shape — will be flattened).
    obs : array-like
        Observed values (same shape).
    filename : str
        Output path.
    logger : phenonn.utils.logger.Logger, optional
    title, xlabel, ylabel : str
        Plot labels.
    gridsize : int
        Hexbin resolution (number of hexagons along each axis).
    hexbin : bool
        If True, use hexbin density. If False, use scatter. Default True.

    Returns
    -------
    dict
        Computed metrics: rmse, mae, bias, r2, n.

    Notes
    -----
    Errors ignored in computation: NaN and Inf pairs are dropped.
    Plot axis limits are set to cover both predictions and observations with
    a small margin, so the y=x line always appears diagonal.
    """

    pred = np.asarray(pred, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[mask], obs[mask]

    if len(pred) == 0:
        raise ValueError("No finite pred/obs pairs to plot")

    metrics = _compute_metrics(pred, obs)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot density or scatter
    if hexbin and len(pred) >= 200:
        hb = ax.hexbin(
            obs, pred, gridsize=gridsize, cmap="viridis", mincnt=1, bins="log"
        )
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(hb, cax=cbar_ax, label=r"$\mathrm{\log_{10}[Count]}$")
    else:
        ax.scatter(obs, pred, alpha=0.5, s=12, color="#1f77b4", edgecolor="none")

    # y=x reference line — span the combined data range
    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    span = hi - lo
    margin = 0.05 * span if span > 0 else 0.01
    axmin, axmax = lo - margin, hi + margin
    ax.plot([axmin, axmax], [axmin, axmax], "k--")
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(span / 4))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(span / 4))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Metrics text box (upper left)
    txt = (
        f"$R^2$ = {metrics['r2']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"MAE  = {metrics['mae']:.4f}\n"
        f"Bias = {metrics['bias']:+.4f}\n"
        f"N    = {metrics['n']:,}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved plot to {filename}")
    else:
        print(f"Saved plot to {filename}")
    return metrics


# ── 4. LAI annual curves for low / medium / high R² sites ────────────────────


def plot_gcc_curves(
    df: "pd.DataFrame",
    filename: str = "gcc_curves_by_r2.png",
    logger=None,
    seed: int = 42,
    site_col: str = "site",
    year_col: str = "year",
    doy_col: str = "day_index",
    pred_col: str = "lai_pred",
    obs_col: str = "lai_obs",
    extra_col: Optional[str] = None,
    extra_label: str = "extra",
) -> Dict[str, float]:
    """
    Plot observed vs predicted annual LAI curves for three representative sites.

    When `extra_col` is given, a third (dotted) line per year is overlaid —
    used e.g. to add the climatology curve alongside obs and model prediction.

    Picks one random site with low R², one with medium R², and one with high R²
    relative to the mean across all sites. Each site gets a subplot showing all
    validation years overlaid, with observed LAI as a solid line and predicted
    LAI as a dashed line.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions dataframe as produced by predict.py, with columns for
        site, year, day index, lai_pred, and lai_obs.
    filename : str
        Output path for the figure.
    logger : phenonn.utils.logger.Logger, optional
    seed : int
        Random seed for reproducible site selection.
    site_col, year_col, doy_col, pred_col, obs_col : str
        Column names in df. Defaults match predict.py output.

    Returns
    -------
    dict
        {site_name: r2_value} for the three selected sites.

    Notes
    -----
    Site selection: all sites are ranked by R². The site pool is split into
    three terciles (low / mid / high). One site is drawn randomly from each
    tercile. Using terciles rather than the absolute best/worst avoids
    always showing the same outlier sites.
    """

    # ── Compute per-site R², MAE, MSE ──
    site_metrics = {}
    for site, g in df.groupby(site_col):
        p = g[pred_col].values
        o = g[obs_col].values
        mask = np.isfinite(p) & np.isfinite(o)
        if mask.sum() < 10:
            continue
        p, o = p[mask], o[mask]
        ss_res = np.sum((o - p) ** 2)
        ss_tot = np.sum((o - o.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(o - p)))
        mse = float(np.mean((o - p) ** 2))
        site_metrics[site] = (r2, mae, mse)

    if len(site_metrics) < 3:
        raise ValueError(
            f"Need at least 3 sites with enough data, got {len(site_metrics)}"
        )

    # ── Rank sites by R² and pick one from each tercile ──
    sorted_sites = sorted(site_metrics.items(), key=lambda x: x[1][0])
    n = len(sorted_sites)
    tercile_size = max(1, n // 3)

    low_pool = sorted_sites[:tercile_size]
    mid_pool = sorted_sites[tercile_size : 2 * tercile_size]
    high_pool = sorted_sites[2 * tercile_size :]

    rng = np.random.RandomState(seed)
    pick_low = low_pool[rng.randint(len(low_pool))]
    pick_mid = mid_pool[rng.randint(len(mid_pool))]
    pick_high = high_pool[rng.randint(len(high_pool))]

    picks = [
        ("Low R²", pick_low[0], pick_low[1]),
        ("Medium R²", pick_mid[0], pick_mid[1]),
        ("High R²", pick_high[0], pick_high[1]),
    ]

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    plt.subplots_adjust(hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle("LAI annual curves — low / medium / high R² sites", fontsize=14)

    for ax, (label, site_name, metrics) in zip(axes, picks):
        r2_val, mae_val, mse_val = metrics
        site_df = df[df[site_col] == site_name].copy()
        years = sorted(site_df[year_col].unique())
        linestyles = mpltex.linestyle_generator(markers=[])
        for i, year in enumerate(years):
            ydf = site_df[site_df[year_col] == year].sort_values(doy_col)

            # DOY axis: use day_index modulo 365 if it's an absolute index,
            # or use it directly if it's already 1-365
            doys = ydf[doy_col].values
            if doys.max() > 400:
                # Absolute index — convert to within-year DOY
                doys = doys - doys.min() + 1

            base_style = next(linestyles)

            obs_style = base_style.copy()
            obs_style["linestyle"] = "-"
            ax.plot(
                doys,
                ydf[obs_col].values,
                label=f"{year} obs" if i == 0 else f"{year} obs",
                **obs_style,
            )

            pred_style = base_style.copy()
            pred_style["linestyle"] = "--"

            ax.plot(
                doys,
                ydf[pred_col].values,
                label=f"{year} pred" if i == 0 else f"{year} pred",
                **pred_style,
            )

            if extra_col is not None:
                extra_style = base_style.copy()
                extra_style["linestyle"] = ":"
                ax.plot(
                    doys,
                    ydf[extra_col].values,
                    label=f"{year} {extra_label}",
                    **extra_style,
                )

        ax.set_title(
            f"{label}: {site_name}  "
            f"(R² = {r2_val:.4f}  |  MAE = {mae_val:.4f}  |  MSE = {mse_val:.4f})",
            fontsize=12,
        )
        ax.set_ylabel("LAI")
        ax.legend(fontsize=8, ncol=min(len(years), 4), loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Day of year")
    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    # Log or print the save location
    if logger:
        logger.info(f"Saved GCC curves plot to {filename}")
    else:
        print(f"Saved GCC curves plot to {filename}")

    return {site: metrics for _, site, metrics in picks}


# ── 5. LAI annual curves for ALL sites ───────────────────────────────────────


def plot_gcc_curves_all(
    df: "pd.DataFrame",
    filename: str = "gcc_curves_all.png",
    logger=None,
    cols: int = 4,
    site_col: str = "site",
    year_col: str = "year",
    doy_col: str = "day_index",
    pred_col: str = "lai_pred",
    obs_col: str = "lai_obs",
    extra_col: Optional[str] = None,
    extra_label: str = "extra",
    pft_by_site: Optional[Dict] = None,
    pft_names: Optional[List[str]] = None,
    pft_min_frac: float = 0.05,
) -> Dict[str, float]:
    """
    Plot observed vs predicted annual LAI curves for every site.

    When `extra_col` is given, a third line per year is overlaid (e.g. the
    climatology curve alongside obs and model prediction).

    When `pft_by_site` ({site: (N_PFT,) fraction vector}) is given, each subplot
    title also lists the PFTs whose fraction is ≥ `pft_min_frac` (named via
    `pft_names` if provided, else PFT1..N), sorted by decreasing fraction.

    One small subplot per site, arranged in a grid, sorted by R² from best
    (top-left) to worst (bottom-right). Each subplot overlays all validation
    years with observed (solid) and predicted (dashed) lines.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions dataframe with columns for site, year, day index,
        lai_pred, and lai_obs.
    filename : str
        Output path.
    logger : phenonn.utils.logger.Logger, optional
    cols : int
        Number of columns in the grid. Default 4.
    site_col, year_col, doy_col, pred_col, obs_col : str
        Column names in df.

    Returns
    -------
    dict
        {site_name: r2_value} for all sites.
    """

    # ── Compute per-site R², MAE, MSE ──
    site_metrics = {}
    for site, g in df.groupby(site_col):
        p = g[pred_col].values
        o = g[obs_col].values
        mask = np.isfinite(p) & np.isfinite(o)
        if mask.sum() < 10:
            continue
        p, o = p[mask], o[mask]
        ss_res = np.sum((o - p) ** 2)
        ss_tot = np.sum((o - o.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(o - p)))
        mse = float(np.mean((o - p) ** 2))
        site_metrics[site] = (r2, mae, mse)

    # Sort by R² descending (best first)
    sorted_sites = sorted(
        site_metrics.items(),
        key=lambda x: -x[1][0] if np.isfinite(x[1][0]) else float("inf"),
    )
    n_sites = len(sorted_sites)

    if n_sites == 0:
        raise ValueError("No sites with enough data to plot")

    rows = math.ceil(n_sites / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 3.4 * rows),   # taller rows: room for multi-line titles
        squeeze=False,
        constrained_layout=True,          # auto-space titles/suptitle (no overlap)
    )

    fig.suptitle(f"LAI predicted vs observed — {n_sites} sites (sorted by R²)")

    legend_handles = []
    legend_labels = []

    for idx, (site_name, metrics) in enumerate(sorted_sites):
        r2_val, mae_val, mse_val = metrics
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        site_df = df[df[site_col] == site_name]
        years = sorted(site_df[year_col].unique())

        linestyles = mpltex.linestyle_generator(markers=[])
        for i, year in enumerate(years):
            ydf = site_df[site_df[year_col] == year].sort_values(doy_col)

            doys = ydf[doy_col].values
            if len(doys) > 0 and doys.max() > 400:
                doys = doys - doys.min() + 1

            line_obs = ax.plot(doys, ydf[obs_col].values, **next(linestyles))
            line_pred = ax.plot(doys, ydf[pred_col].values, **next(linestyles))
            line_extra = (ax.plot(doys, ydf[extra_col].values, **next(linestyles))
                          if extra_col is not None else None)

            # Collect legend handles and labels from the first site only
            if idx == 0:
                legend_handles.append(line_obs[0])
                legend_labels.append(f"{year} obs")
                legend_handles.append(line_pred[0])
                legend_labels.append(f"{year} pred")
                if line_extra is not None:
                    legend_handles.append(line_extra[0])
                    legend_labels.append(f"{year} {extra_label}")

        # Title with site name, R², MAE, MSE
        r2_str = f"{r2_val:.2f}" if np.isfinite(r2_val) else "N/A"
        mae_str = f"{mae_val:.2f}" if np.isfinite(mae_val) else "N/A"
        mse_str = f"{mse_val:.2f}" if np.isfinite(mse_val) else "N/A"
        title = f"{site_name}\nR²={r2_str}  MAE={mae_str}  MSE={mse_str}"
        if pft_by_site is not None and site_name in pft_by_site:
            frac = pft_by_site[site_name]
            items = sorted(
                ((pft_names[i] if pft_names and i < len(pft_names) else f"PFT{i+1}",
                  float(frac[i]))
                 for i in range(len(frac))
                 if np.isfinite(frac[i]) and frac[i] >= pft_min_frac),
                key=lambda t: -t[1],
            )
            if items:
                title += "\n" + "  ".join(f"{nm}:{100 * f:.0f}%" for nm, f in items)
        ax.set_title(title, fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n_sites, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # Shared legend (one for the whole figure)
    # Build legend entries from the first site's years
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="center right",
            bbox_to_anchor=(1.1, 0.5),
            ncol=1,
        )

    # Save the figure
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    # Log or print the save location
    if logger:
        logger.info(f"Saved GCC curves plot to {filename}")
    else:
        print(f"Saved GCC curves plot to {filename}")

    return dict(sorted_sites)


# ── 6. Feature and target distributions ──────────────────────────────────────


def plot_feature_distributions(
    site_files: List[str],
    filename: str = "feature_distributions.png",
    logger=None,
    cols: int = 4,
    n_bins: int = 60,
    max_sites: int = 20,
) -> None:
    """
    Plot histograms of all input features and the target variable.

    Shows the raw distribution and, for log-transformed features, the
    distribution after log1p. Useful for checking skewness, spotting
    outliers, and verifying that normalization choices are appropriate.

    Parameters
    ----------
    site_files : list of str
        Paths to site CSVs. A random subset of max_sites is used to
        keep computation fast.
    filename : str
        Output path for the figure.
    logger : phenonn.utils.logger.Logger, optional
    cols : int
        Number of columns in the grid.
    n_bins : int
        Number of histogram bins.
    max_sites : int
        Maximum number of sites to sample (for speed).

    Notes
    -----
    For each feature, the histogram shows:
    - Blue: raw values (pooled across all sampled sites)
    - Orange (if applicable): values after log1p transform
    - Vertical red dashed lines: mean ± 1 std
    - Title includes skewness and % of near-zero values

    The target (LAI) is shown in green in the last panel.
    """

    # Sample sites for speed
    rng = np.random.RandomState(42)
    if len(site_files) > max_sites:
        indices = rng.choice(len(site_files), max_sites, replace=False)
        sampled = [site_files[i] for i in indices]
    else:
        sampled = site_files

    # Load and concatenate data
    all_dfs = []
    for f in sampled:
        df = pd.read_csv(f)
        df = df.sort_values(["year", "doy"]).reset_index(drop=True)
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
        # `add_derived_features` is the legacy pandas helper from the old
        # LaiNN CSV pipeline. It lived in `data_creation/`, which is NOT part
        # of this repository (dataset building is kept elsewhere), so this
        # legacy plotting helper is unavailable here.
        try:
            from phenonn.data_creation.feature_engineering_pd_legacy import (  # noqa
                add_derived_features,
            )
        except ModuleNotFoundError as exc:                      # pragma: no cover
            raise ModuleNotFoundError(
                "plot_feature_distributions() relies on the legacy CSV pipeline "
                "helper `data_creation.feature_engineering_pd_legacy`, which is "
                "not shipped with this repository. It is unused by the current "
                "training/prediction pipeline."
            ) from exc
        df = add_derived_features(df)
        all_dfs.append(df)
    data = pd.concat(all_dfs, ignore_index=True)

    # Features to plot
    features = list(DYNAMIC_FEATURES) + list(CYCLIC_FEATURES) + list(PFT_COLS)
    targets = ["LAI"]
    all_vars = features + targets

    n_vars = len(all_vars)
    rows = math.ceil(n_vars / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3 * rows))
    plt.subplots_adjust(hspace=0.3, left=0.3, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle(
        f"Feature & target distributions ({len(sampled)} sites, "
        f"{len(data):,} days)",
        fontsize=14,
    )

    for idx, var_name in enumerate(all_vars):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]

        if var_name not in data.columns:
            ax.set_title(f"{var_name}\n(not found)", fontsize=9)
            ax.set_visible(False)
            continue

        vals = data[var_name].dropna().values

        if len(vals) == 0:
            ax.set_title(f"{var_name}\n(all NaN)", fontsize=9)
            continue

        # Compute stats
        skewness = (
            float(((vals - vals.mean()) ** 3).mean() / (vals.std() ** 3))
            if vals.std() > 0
            else 0
        )
        pct_zero = float(100 * (np.abs(vals) < 1e-6).mean())
        mean_val = vals.mean()
        std_val = vals.std()

        # Choose color based on type
        if var_name in targets:
            color = "#2ca02c"  # green for target
            label = "target"
        else:
            color = "#1f77b4"  # blue for features
            label = "raw"

        # Plot raw histogram
        ax.hist(
            vals,
            bins=n_bins,
            color=color,
            alpha=0.7,
            density=True,
            label=label,
            edgecolor="none",
        )

        # If log-transformed, overlay log1p distribution
        if var_name in LOG_TRANSFORM_FEATURES:
            vals_log = np.log1p(np.clip(vals, 0, None))
            ax.hist(
                vals_log,
                bins=n_bins,
                color="#ff7f0e",
                alpha=0.5,
                density=True,
                label="log1p",
                edgecolor="none",
            )

        # Mean ± std lines
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(
            mean_val - std_val, color="red", linestyle=":", linewidth=0.8, alpha=0.5
        )
        ax.axvline(
            mean_val + std_val, color="red", linestyle=":", linewidth=0.8, alpha=0.5
        )

        # Title with stats
        ax.set_title(
            f"{var_name}\n"
            f"skew={skewness:+.2f}  zero={pct_zero:.0f}%  "
            f"μ={mean_val:.2g}  σ={std_val:.2g}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for idx in range(n_vars, rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.set_visible(False)

    # Save the figure
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"Saved feature distributions plot to {filename}")
    else:
        print(f"Saved feature distributions plot to {filename}")


# ── 7. Feature distributions per site ────────────────────────────────────────
def plot_feature_distributions_per_site(
    site_files: List[str],
    output_dir: str = "./feature_distributions",
    logger=None,
    cols: int = 4,
    n_bins: int = 60,
) -> None:
    """
    Generate one feature distribution plot per site.

    Produces one PNG file per site, named {sitename}_feature_distribution.png,
    in the output directory. Each file shows the same histogram layout as
    plot_feature_distributions but for a single site only.

    Parameters
    ----------
    site_files : list of str
        Paths to site CSVs.
    output_dir : str
        Directory where individual PNGs will be saved.
    logger : phenonn.utils.logger.Logger, optional
    cols : int
        Number of columns in the histogram grid.
    n_bins : int
        Number of histogram bins.

    Examples
    --------
    >>> site_files = sorted(glob.glob("./data/DB/*.csv"))
    >>> plot_feature_distributions_per_site(site_files, output_dir="./runs/dist_per_site")
    # Creates: ./runs/dist_per_site/DB_asuhighlands_feature_distribution.png
    #          ./runs/dist_per_site/DB_bartlett_feature_distribution.png
    #          ... (one per site)
    """

    os.makedirs(output_dir, exist_ok=True)

    n_sites = len(site_files)
    failed_sites = []
    for i, filepath in enumerate(site_files):
        site_key = os.path.splitext(os.path.basename(filepath))[0]
        out_filename = os.path.join(output_dir, f"{site_key}_feature_distribution.png")

        progress_msg = f"  [{i+1}/{n_sites}] {site_key}"
        if logger:
            logger.info(progress_msg)
        else:
            print(progress_msg)

        try:
            plot_feature_distributions(
                site_files=[filepath],
                filename=out_filename,
                logger=logger,
                cols=cols,
                n_bins=n_bins,
                max_sites=1,
            )
        except Exception as e:
            error_msg = f"    Failed to plot {site_key}: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            failed_sites.append(site_key)
            continue

    if failed_sites:
        summary = f"Completed {n_sites - len(failed_sites)}/{n_sites} sites. Failed: {', '.join(failed_sites)}"
    else:
        summary = f"Successfully saved {n_sites} distribution plots to {output_dir}/"

    if logger:
        if hasattr(logger, "success"):
            logger.success(summary)
        else:
            logger.info(summary)
    else:
        print(summary)


# ── Convenience: build histories from a training loop ────────────────────────


def make_history_dicts() -> tuple:
    """
    Shortcut for initializing matched train/valid history dicts.

    Returns
    -------
    (train_hist, valid_hist) : tuple of dict
        Each with empty lists keyed by 'loss', 'rmse', 'r2'.

    Example
    -------
    >>> train_hist, valid_hist = make_history_dicts()
    >>> # inside training loop:
    >>> train_hist['loss'].append(train_loss)
    >>> valid_hist['loss'].append(val_loss)
    >>> valid_hist['rmse'].append(val_rmse)
    >>> valid_hist['r2'].append(val_r2)
    """
    keys = ["loss", "rmse", "r2"]
    return {k: [] for k in keys}, {k: [] for k in keys}
