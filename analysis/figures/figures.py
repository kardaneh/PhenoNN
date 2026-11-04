"""Paper figures F1–F13.

Each function takes tidy artefacts (DataFrames / metric dicts) and writes a
PDF + 300 dpi PNG through :func:`analysis.figures.style.savefig`.  The functions
never read the filesystem except when a convenience loader is provided; this
keeps them testable with synthetic bundles.
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.figures.style import (
    MODEL_COLORS,
    PALETTE,
    apply_style,
    savefig,
)
from analysis.figures.ipsl_aid import (
    add_log_count_colorbar,
    collect_handles,
    figure_legend,
    figure_note,
    hexbin_density,
    identity_line,
    integer_epoch_axis,
    linestyle_cycle,
    metrics_line,
    model_label,
    next_style,
)

__all__ = [
    "plot_paper_pred_vs_obs",
    "plot_paper_site_year_curves",
    "plot_paper_anomaly_scatter",
    "plot_paper_rmse_by_doy",
    "plot_paper_ablation",
    "plot_paper_perturbation_curves",
    "plot_paper_warming_co2",
    "plot_paper_pft_curves",
    "plot_paper_loss_curves",
    "plot_paper_memory_kernel",
    # Deprecated F-number aliases (kept for external scripts).
    "f1_pred_vs_obs",
    "f2_site_year_curves",
    "f3_anomaly_scatter",
    "f4_rmse_by_doy",
    "f8_ablation",
    "f9_perturbation_curves",
    "f10_warming_co2",
    "f11_pft_curves",
    "f12_loss_curves",
    "f13_memory_kernel",
]


SCENARIO_ORDER = ("drought", "heatwave", "winter_frost", "warming_co2")
SCENARIO_COLORS = {
    "drought": PALETTE["orange"],
    "heatwave": PALETTE["blue"],
    "winter_frost": PALETTE["teal"],
    "warming_co2": PALETTE["purple"],
}


def _clean(ax):
    ax.grid(alpha=0.25)


def plot_paper_pred_vs_obs(
    pred: pd.DataFrame,
    metrics: Dict,
    out: str,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    title: str = "Predicted vs observed LAI",
) -> None:
    """F1: hexbin density of predictions against observations."""
    apply_style()
    d = pred[[obs_col, pred_col]].dropna()
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    if not d.empty:
        hb = hexbin_density(ax, d[obs_col], d[pred_col])
        add_log_count_colorbar(fig, hb, ax)
        lo = float(min(d[obs_col].min(), d[pred_col].min()))
        hi = float(max(d[obs_col].max(), d[pred_col].max()))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        identity_line(ax, lo - pad, hi + pad)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Observed LAI")
    ax.set_ylabel("Predicted LAI")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    # Two short lines keep the note inside the figure width, below the x-label.
    figure_note(
        fig,
        "\n".join(
            [
                metrics_line(metrics, ("overall_r2", "rmse", "mae")),
                metrics_line(metrics, ("bias", "slope", "n_points")),
            ]
        ),
    )
    savefig(fig, out)


def _pick_site_years(pred: pd.DataFrame, n: int = 2) -> List[Tuple[str, int]]:
    """Pick two contrasting, well-predicted site-years.

    The default is deliberately not "best vs worst": a catastrophic site-year
    is not a useful illustration of the model.  We keep site-years with at
    least 24 valid dekads, non-flat observations and an RMSE in the better half
    of the distribution, then choose the lowest and highest observed mean LAI.
    """
    d = pred[["site_id", "year", "lai_obs", "lai_pred"]].dropna()
    if d.empty:
        return []
    d = d.assign(err2=(d["lai_pred"] - d["lai_obs"]) ** 2)
    g = (
        d.groupby(["site_id", "year"], sort=True)
        .agg(
            obs_mean=("lai_obs", "mean"),
            obs_std=("lai_obs", "std"),
            mse=("err2", "mean"),
            size=("lai_obs", "size"),
        )
        .reset_index()
    )
    g = g[(g["size"] >= 24) & (g["obs_std"] > 0.05) & g["mse"].notna()].copy()
    if g.empty:
        return []
    g["rmse"] = np.sqrt(g["mse"])
    good = g[g["rmse"] <= g["rmse"].quantile(0.5)]
    if len(good) < 2:
        good = g
    good = good.sort_values("obs_mean")
    picks = [good.iloc[0], good.iloc[-1]]
    if (
        len(good) > 2
        and picks[0][["site_id", "year"]].tolist()
        == picks[1][["site_id", "year"]].tolist()
    ):
        picks[1] = good.iloc[-2]
    return [(str(r.site_id), int(r.year)) for r in picks[:n]]


def _site_year_scores(frame: pd.DataFrame) -> Tuple[float, float]:
    """Return (R², RMSE) for one site-year frame."""
    obs = frame["lai_obs"].to_numpy(float)
    pred = frame["lai_pred"].to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if obs.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = float(1.0 - np.sum((pred - obs) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return r2, rmse


def _composition_text(
    pft_df: Optional[pd.DataFrame],
    site,
    year,
    min_frac: float = 0.05,
    top: int = 3,
) -> str:
    """Top PFT fractions for one site-year, or an empty string."""
    if pft_df is None or pft_df.empty:
        return ""
    g = pft_df[(pft_df["site_id"].astype(str) == str(site)) & (pft_df["year"] == year)]
    if g.empty:
        return ""
    g = (
        g.groupby(["pft", "pft_name"], as_index=False)["frac"]
        .first()
        .sort_values("frac", ascending=False)
    )
    g = g[g["frac"] >= min_frac].head(top)
    if g.empty:
        return ""
    return "\n".join(
        f"{row.pft_name}: {100.0 * row.frac:.0f}%" for row in g.itertuples()
    )


def _pft_composition_from_files(pft_dir: str, site, year, top: int = 3) -> str:
    """Top PFT fractions read directly from ``PFTmap_{year}.nc``."""
    if not pft_dir:
        return ""
    try:
        import xarray as xr

        from phenonn.utils.config import PFT_COLS, PFT_FNAME, PFT_NAMES
    except Exception:
        return ""
    path = os.path.join(pft_dir, PFT_FNAME.format(year=int(year)))
    if not os.path.exists(path):
        return ""
    try:
        with xr.open_dataset(path) as ds:
            ids = np.asarray(ds["site_id"].values).astype(str)
            hit = np.flatnonzero(ids == str(site))
            if hit.size == 0:
                return ""
            row = int(hit[0])
            if "pft_frac" in ds:
                values = np.asarray(ds["pft_frac"].isel(site=row).values, dtype=float)
                fractions = {
                    name: float(values[i])
                    for i, name in enumerate(PFT_NAMES)
                    if i < values.size
                }
            else:
                fractions = {
                    name: float(ds[col].isel(site=row).values)
                    for name, col in zip(PFT_NAMES, PFT_COLS)
                    if col in ds
                }
    except Exception:
        return ""
    ranked = sorted(fractions.items(), key=lambda item: item[1], reverse=True)
    ranked = [(name, frac) for name, frac in ranked if frac >= 0.01][:top]
    return "\n".join(f"{name}: {100.0 * frac:.0f}%" for name, frac in ranked)


def plot_paper_site_year_curves(
    pred: pd.DataFrame,
    out: str,
    choices: Optional[Sequence[Tuple[str, int]]] = None,
    title: str = "",
    pft_df: Optional[pd.DataFrame] = None,
    pft_dir: str = "",
) -> None:
    """F2: two site-year curves with observations, climatology and scores.

    Parameters
    ----------
    pred : pandas.DataFrame
        Prediction bundle with ``site_id``, ``year``, ``doy``, ``lai_obs``,
        ``lai_pred`` and optionally ``lai_clim``.
    out : str
        Output path stem; PDF and PNG are written.
    choices : sequence of (site_id, year), optional
        Explicit site-years.  Defaults to two quality-filtered, contrasting
        site-years (low and high observed LAI).
    title : str
        Optional figure title; normally left empty for the LaTeX caption.
    pft_df : pandas.DataFrame, optional
        Long per-PFT table (``site_id``, ``year``, ``pft``, ``pft_name``,
        ``frac``).  When given, the top three PFT fractions are shown.
    pft_dir : str, optional
        Folder of ``PFTmap_{year}.nc``; used when ``pft_df`` does not contain
        the chosen site-year.
    """
    apply_style()
    if choices is None:
        choices = _pick_site_years(pred, 2)
    if not choices:
        raise ValueError("no site-year with valid observations")
    fig, axes = plt.subplots(
        1,
        len(choices),
        figsize=(5.8 * len(choices), 4.6),
        squeeze=False,
        layout="constrained",
    )
    # Stable semantics: observations are markers, predictions the first mpltex
    # style (solid), climatology the second (dashed).
    styles = linestyle_cycle()
    pred_ls = next(styles)["linestyle"]
    clim_ls = next(styles)["linestyle"]
    for index, (ax, (site, year)) in enumerate(zip(axes[0], choices)):
        g = pred[(pred["site_id"].astype(str) == str(site)) & (pred["year"] == year)]
        g = g.sort_values("doy")
        if g.empty:
            ax.set_visible(False)
            continue
        ax.plot(
            g["doy"],
            g["lai_obs"],
            "o",
            ms=4,
            color=PALETTE["blue"],
            label="Observed",
            zorder=3,
        )
        ax.plot(
            g["doy"],
            g["lai_pred"],
            linestyle=pred_ls,
            lw=2,
            color=PALETTE["red"],
            label="Predicted",
            zorder=2,
        )
        if "lai_clim" in g.columns and g["lai_clim"].notna().any():
            ax.plot(
                g["doy"],
                g["lai_clim"],
                linestyle=clim_ls,
                lw=1.5,
                color=PALETTE["green"],
                label="Climatology",
                zorder=1,
            )
        ax.set_xlabel("Day of year")
        if index == 0:
            ax.set_ylabel("LAI")
        ax.set_xlim(0, 365)
        _clean(ax)

        # Composition and scores go in the panel title: constrained layout
        # reserves that space, so nothing is drawn on top of the curves.
        r2, rmse = _site_year_scores(g)
        composition = _composition_text(pft_df, site, year)
        if not composition:
            composition = _pft_composition_from_files(pft_dir, site, year)
        if not composition and "pft_dom_name" in g.columns:
            purity = (
                f" (purity {g['purity'].iloc[0]:.2f})"
                if "purity" in g.columns and np.isfinite(g["purity"].iloc[0])
                else ""
            )
            composition = f"{g['pft_dom_name'].iloc[0]}{purity}"

        panel_lines = [f"{site} — {year}"]
        if composition:
            panel_lines.append(composition)
        if np.isfinite(rmse):
            panel_lines.append(
                f"R² = {r2:.3f} · RMSE = {rmse:.3f}"
                if np.isfinite(r2)
                else f"RMSE = {rmse:.3f}"
            )
        ax.set_title("\n".join(panel_lines), fontsize=9)

    handles, labels = collect_handles(axes[0])
    figure_legend(fig, handles, labels, ncol=3)
    if title:
        fig.suptitle(title)
    savefig(fig, out)


def plot_paper_anomaly_scatter(
    pred: pd.DataFrame,
    metrics: Dict,
    out: str,
    obs_col: str = "obs_anom",
    pred_col: str = "pred_anom",
) -> None:
    """F3: anomaly scatter (no PFT mixing, training-year climatology)."""
    apply_style()
    d = pred[[obs_col, pred_col]].dropna()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if not d.empty:
        hb = hexbin_density(ax, d[obs_col], d[pred_col])
        add_log_count_colorbar(fig, hb, ax)
        lim = float(np.nanpercentile(np.abs(d.values), 99.5))
        lim = max(lim, 0.05)
        identity_line(ax, -lim, lim)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Observed ΔLAI")
    ax.set_ylabel("Predicted ΔLAI")
    ax.set_title("Anomaly skill")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    an = metrics.get("anomaly") if isinstance(metrics, dict) else None
    figure_note(
        fig,
        metrics_line(an or metrics, keys=("overall_r2", "rmse", "mae", "bias")),
    )
    savefig(fig, out)


def plot_paper_rmse_by_doy(
    rmse: pd.DataFrame,
    out: str,
    title: str = "RMSE by day of year",
) -> None:
    """F4: RMSE per DOY against climatology and the derived skill score."""
    apply_style()
    if rmse.empty:
        raise ValueError("empty rmse_by_doy table")

    has_clim = "rmse_clim" in rmse.columns and rmse["rmse_clim"].notna().any()
    if has_clim:
        fig, (ax, ax_skill) = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
            layout="constrained",
        )
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax_skill = None

    styles = linestyle_cycle()
    model_ls = next(styles)["linestyle"]
    clim_ls = next(styles)["linestyle"]

    ax.plot(
        rmse["doy"],
        rmse["rmse_model"],
        linestyle=model_ls,
        color=PALETTE["red"],
        label="Model",
    )
    has_better = False
    if has_clim:
        ax.plot(
            rmse["doy"],
            rmse["rmse_clim"],
            linestyle=clim_ls,
            color=PALETTE["grey"],
            label="Climatology",
        )
        better = rmse["rmse_model"] < rmse["rmse_clim"]
        has_better = bool(better.any())
        if has_better:
            ax.fill_between(
                rmse["doy"],
                rmse["rmse_model"],
                rmse["rmse_clim"],
                where=better,
                color=PALETTE["green"],
                alpha=0.22,
                lw=0,
                label="Model beats climatology",
            )
    ax.set_ylabel("RMSE (LAI)")
    ax.set_title(title)
    ax.set_xlim(0, 365)
    ax.set_ylim(bottom=0)
    _clean(ax)

    if ax_skill is not None:
        denom = rmse["rmse_clim"].replace(0.0, np.nan)
        skill = 1.0 - rmse["rmse_model"] / denom
        ax_skill.axhline(0.0, color="grey", lw=0.8)
        ax_skill.plot(rmse["doy"], skill, color=PALETTE["blue"], lw=1.8)
        ax_skill.fill_between(
            rmse["doy"],
            0,
            skill,
            where=skill > 0,
            color=PALETTE["green"],
            alpha=0.18,
            lw=0,
        )
        ax_skill.set_xlabel("Day of year")
        ax_skill.set_ylabel("Skill score")
        ax_skill.set_xlim(0, 365)
        _clean(ax_skill)
    else:
        ax.set_xlabel("Day of year")

    handles, labels = collect_handles([ax])
    figure_legend(fig, handles, labels, ncol=3)
    if has_clim and not has_better:
        figure_note(fig, "Model does not beat climatology")
    savefig(fig, out)


def plot_paper_ablation(
    df: pd.DataFrame,
    out: str,
    group_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> None:
    """F8: ΔR² per ablated feature group, with bootstrap CIs when present.

    The aggregate script writes ``label``/``delta_r2``; older hand-made tables
    may use ``group``/``dR2``.  Both spellings are accepted.
    """
    apply_style()
    if value_col is None:
        value_col = "delta_r2" if "delta_r2" in df.columns else "dR2"
    if group_col is None:
        group_col = "group" if "group" in df.columns else "label"
    missing = [name for name in (group_col, value_col) if name not in df.columns]
    if missing:
        raise ValueError(
            f"ablation plot needs columns {missing}; got {list(df.columns)}"
        )

    d = df.dropna(subset=[value_col]).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * max(len(d), 3) + 1.6))
    y = np.arange(len(d))
    colors = [PALETTE["green"] if v >= 0 else PALETTE["orange"] for v in d[value_col]]
    ax.barh(y, d[value_col], color=colors, alpha=0.85)
    if "ci_low" in d.columns and "ci_high" in d.columns:
        ax.errorbar(
            d[value_col],
            y,
            xerr=[d[value_col] - d["ci_low"], d["ci_high"] - d[value_col]],
            fmt="none",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d[group_col])
    ax.set_xlabel("ΔR² (full − ablated)")
    ax.set_title("Feature-group ablation")
    _clean(ax)
    fig.tight_layout()
    savefig(fig, out)


def plot_paper_perturbation_curves(
    frames: Dict[str, pd.DataFrame],
    out: str,
    value_col: str = "delta",
) -> None:
    """F9: seasonal ΔLAI curves for one or more perturbation scenarios."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    ax.axhline(0, color="grey", lw=0.8)
    labels = [s for s in SCENARIO_ORDER if s in frames]
    labels += [s for s in frames if s not in SCENARIO_ORDER]
    styles = linestyle_cycle()
    for label in labels:
        df = frames[label]
        if df.empty or "doy" not in df.columns or value_col not in df.columns:
            continue
        color = SCENARIO_COLORS.get(
            label,
            plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        )
        g = df.groupby("doy")[value_col].agg(["mean", "std", "count"]).sort_index()
        sem = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.plot(g.index, g["mean"], label=label, **next_style(styles, color=color))
        ax.fill_between(
            g.index,
            g["mean"] - 1.96 * sem,
            g["mean"] + 1.96 * sem,
            color=color,
            alpha=0.18,
            lw=0,
        )
    ax.set_xlabel("Day of year")
    ax.set_ylabel("ΔLAI (perturbed − baseline)")
    ax.set_title("Perturbation response")
    ax.set_xlim(0, 365)
    _clean(ax)
    handles, labels = collect_handles([ax])
    figure_legend(fig, handles, labels, ncol=min(3, max(2, len(labels))))
    savefig(fig, out)


def plot_paper_warming_co2(df: pd.DataFrame, out: str) -> None:
    """F10: ΔLAI and phenology shifts versus warming, one line per CO2 mode."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    d = df.copy()
    if "co2_mode" not in d.columns:
        d["co2_mode"] = "absolute"
    styles = linestyle_cycle(markers=True)
    for mode, g in d.groupby("co2_mode"):
        g = g.sort_values("delta_t")
        style = next_style(styles)
        axes[0].plot(g["delta_t"], g["mean_delta_lai"], label=mode, **style)
        if "median_dsos_10" in g.columns:
            axes[1].plot(g["delta_t"], g["median_dsos_10"], label=mode, **style)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("ΔT (°C)")
    axes[0].set_ylabel("Mean ΔLAI")
    axes[0].set_title("LAI response")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("ΔT (°C)")
    axes[1].set_ylabel("ΔSOS (days)")
    axes[1].set_title("Phenology response")
    handles, labels = collect_handles(axes)
    for ax in axes:
        _clean(ax)
    figure_legend(
        fig,
        handles,
        labels,
        ncol=min(3, max(2, len(labels))),
        loc="outside upper center",
    )
    savefig(fig, out)


def plot_paper_pft_curves(
    df: pd.DataFrame,
    out: str,
    site_id: Optional[str] = None,
    year: Optional[int] = None,
    min_frac: float = 0.05,
) -> None:
    """F11: per-PFT pure-LAI curves for one deterministic mixed site-year."""
    apply_style()
    if df.empty:
        raise ValueError("empty per-PFT table")
    if site_id is None or year is None:
        counts = (
            df[df["frac"] >= min_frac]
            .groupby(["site_id", "year"])["pft"]
            .nunique()
            .sort_values(ascending=False)
        )
        site_id, year = counts.index[0]
    g = df[(df["site_id"].astype(str) == str(site_id)) & (df["year"] == year)]
    if g.empty:
        raise ValueError(f"no rows for {site_id} / {year}")

    fig, ax = plt.subplots(figsize=(7.4, 5.2), layout="constrained")
    colors = plt.cm.tab20(np.linspace(0, 1, 15))
    styles = linestyle_cycle()
    grey_label = "PFT < 5%"
    grey_labelled = False
    for pft, h in g.groupby("pft"):
        h = h.sort_values("doy")
        frac = float(h["frac"].iloc[0])
        if frac < min_frac:
            ax.plot(
                h["doy"],
                h["lai_pure"],
                color="lightgrey",
                lw=0.7,
                zorder=1,
                label=None if grey_labelled else grey_label,
            )
            grey_labelled = True
        else:
            ax.plot(
                h["doy"],
                h["lai_pure"],
                lw=1.8,
                label=f"{h['pft_name'].iloc[0]} ({frac * 100:.0f}%)",
                **next_style(styles, color=colors[int(pft)]),
            )
    ax.plot(
        g.sort_values("doy")["doy"],
        g.sort_values("doy")["lai_pred"],
        color="k",
        lw=2.4,
        label="Weighted sum",
    )
    ax.plot(
        g.sort_values("doy")["doy"],
        g.sort_values("doy")["lai_obs"],
        "o",
        color=PALETTE["red"],
        ms=4,
        label="Observed",
    )
    ax.set_xlabel("Day of year")
    ax.set_ylabel("LAI")
    ax.set_xlim(0, 365)
    _clean(ax)
    handles, labels = collect_handles([ax])
    figure_legend(fig, handles, labels, ncol=4)
    savefig(fig, out)


def _history_groups(histories) -> Dict[str, List[pd.DataFrame]]:
    """Normalise a single history or ``{label: history(ies)}`` to lists."""
    if isinstance(histories, pd.DataFrame):
        return {"run": [histories]}
    if hasattr(histories, "items"):
        groups = {}
        for label, value in histories.items():
            frames = [value] if isinstance(value, pd.DataFrame) else list(value)
            frames = [frame for frame in frames if isinstance(frame, pd.DataFrame)]
            if frames:
                groups[str(label)] = frames
        return groups
    raise TypeError("histories must be a DataFrame or a mapping of histories")


def _mean_curve(frames, column: str) -> pd.DataFrame:
    """Mean and std of one loss column, aligned on epoch."""
    if column not in frames[0].columns:
        return pd.DataFrame()
    parts = [
        frame[["epoch", column]].dropna()
        for frame in frames
        if {"epoch", column}.issubset(frame.columns)
    ]
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby("epoch")[column].agg(["mean", "std"]).sort_index()


def plot_paper_loss_curves(
    histories,
    out: str,
    early_stop_epochs: Optional[Mapping[str, int]] = None,
    log_scale: bool = False,
) -> None:
    """F12: training/validation loss for one or more architectures.

    Parameters
    ----------
    histories : pandas.DataFrame or mapping
        A single ``history.csv`` frame, or ``{architecture: history}``.  A
        mapping value may be a list of frames to average several seeds.
    out : str
        Output path stem; PDF and PNG are written.
    early_stop_epochs : mapping, optional
        ``{architecture: epoch}`` for an early-stop marker.  The marker is
        omitted when the value is missing or ``None``.
    log_scale : bool
        Use a logarithmic loss axis.  ``make_all`` writes both variants.
    """
    apply_style()
    groups = _history_groups(histories)
    if not groups:
        raise ValueError("no training history to plot")

    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.6), sharex=True, layout="constrained"
    )
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = linestyle_cycle()
    drew_best = False
    drew_stop = False

    for index, (label, frames) in enumerate(groups.items()):
        color = MODEL_COLORS.get(label, default_colors[index % len(default_colors)])
        line = next(styles)
        line_style = line.get("linestyle", "-") if isinstance(line, dict) else "-"
        display = model_label(label)
        train = _mean_curve(frames, "train_loss")
        valid = _mean_curve(frames, "val_loss")

        if not train.empty:
            axes[0].plot(
                train.index,
                train["mean"],
                color=color,
                linestyle=line_style,
                label=display,
            )
            if len(frames) > 1:
                axes[0].fill_between(
                    train.index,
                    train["mean"] - train["std"].fillna(0.0),
                    train["mean"] + train["std"].fillna(0.0),
                    color=color,
                    alpha=0.15,
                    lw=0,
                )
        if not valid.empty:
            axes[1].plot(
                valid.index,
                valid["mean"],
                color=color,
                linestyle=line_style,
                label=display,
            )
            if len(frames) > 1:
                axes[1].fill_between(
                    valid.index,
                    valid["mean"] - valid["std"].fillna(0.0),
                    valid["mean"] + valid["std"].fillna(0.0),
                    color=color,
                    alpha=0.15,
                    lw=0,
                )

            best_epoch = None
            for frame in frames:
                if "is_best" in frame.columns and frame["is_best"].any():
                    best_epoch = int(
                        frame.loc[frame["is_best"].astype(bool), "epoch"].iloc[-1]
                    )
                    break
            if best_epoch is None:
                best_epoch = int(valid["mean"].idxmin())
            if best_epoch in valid.index:
                axes[1].plot(
                    best_epoch,
                    valid.loc[best_epoch, "mean"],
                    "o",
                    color=color,
                    ms=6,
                )
                drew_best = True

        if early_stop_epochs:
            stop = early_stop_epochs.get(label)
            if stop is not None:
                for axis in axes:
                    axis.axvline(stop, color=color, ls=":", lw=1.2)
                drew_stop = True

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss (dot = best epoch)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_yscale("log" if log_scale else "linear")
        integer_epoch_axis(axis)
        _clean(axis)

    handles, labels = collect_handles([axes[0]])
    if drew_best:
        handles.append(Line2D([], [], marker="o", color="k", linestyle="None", ms=6))
        labels.append("best epoch")
    if drew_stop:
        handles.append(Line2D([], [], color="grey", linestyle=":", lw=1.2))
        labels.append("early stop")
    figure_legend(
        fig,
        handles,
        labels,
        ncol=min(3, max(2, len(labels))),
        loc="outside upper center",
    )
    savefig(fig, out)


def plot_paper_memory_kernel(
    kernel: pd.DataFrame, controls: Optional[pd.DataFrame], out: str
) -> None:
    """F13: memory kernel with the random-day control band."""
    apply_style()
    if kernel.empty:
        raise ValueError("empty memory kernel")
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    styles = linestyle_cycle()
    ax.plot(
        kernel["lag_days"],
        kernel["mean_abs_delta"],
        label="Impulse response",
        **next_style(styles, color=PALETTE["blue"], marker="o"),
    )
    if {"ci_low", "ci_high"}.issubset(kernel.columns):
        ax.fill_between(
            kernel["lag_days"],
            kernel["ci_low"],
            kernel["ci_high"],
            color=PALETTE["blue"],
            alpha=0.18,
            lw=0,
        )
    note = ""
    if controls is not None and not controls.empty:
        values = controls.drop_duplicates("control").set_index("control")["value"]
        if "random_day" in values.index:
            ax.axhline(
                float(values["random_day"]),
                color=PALETTE["grey"],
                ls="--",
                lw=1.2,
                label="Random-day control",
            )
        if "random_day_p95" in values.index:
            ax.axhline(
                float(values["random_day_p95"]),
                color=PALETTE["grey"],
                ls=":",
                lw=1.0,
                alpha=0.8,
                label="Random-day p95",
            )
        if "noop" in values.index:
            ax.axhline(
                float(values["noop"]),
                color=PALETTE["lightgrey"],
                ls="-.",
                lw=1.0,
                label="No-op control",
            )
        if "linearity" in values.index:
            note = f"linearity ratio = {float(values['linearity']):.2f}"
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Mean |ΔLAI|")
    ax.set_title("Memory kernel")
    _clean(ax)
    handles, labels = collect_handles([ax])
    figure_legend(fig, handles, labels, ncol=2)
    figure_note(fig, note)
    savefig(fig, out)


# ── Deprecated F-number aliases ─────────────────────────────────────────────
# Kept so existing scripts that call ``F.f1_pred_vs_obs`` keep working.
f1_pred_vs_obs = plot_paper_pred_vs_obs
f2_site_year_curves = plot_paper_site_year_curves
f3_anomaly_scatter = plot_paper_anomaly_scatter
f4_rmse_by_doy = plot_paper_rmse_by_doy
f8_ablation = plot_paper_ablation
f9_perturbation_curves = plot_paper_perturbation_curves
f10_warming_co2 = plot_paper_warming_co2
f11_pft_curves = plot_paper_pft_curves
f12_loss_curves = plot_paper_loss_curves
f13_memory_kernel = plot_paper_memory_kernel
