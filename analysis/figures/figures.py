"""Paper figures F1–F13.

Each function takes tidy artefacts (DataFrames / metric dicts) and writes a
PDF + 300 dpi PNG through :func:`analysis.figures.style.savefig`.  The functions
never read the filesystem except when a convenience loader is provided; this
keeps them testable with synthetic bundles.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analysis.figures.style import (
    MODEL_COLORS,
    PALETTE,
    annotate_metrics,
    apply_style,
    savefig,
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
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if not d.empty:
        hb = ax.hexbin(
            d[obs_col],
            d[pred_col],
            gridsize=70,
            bins="log",
            cmap="viridis",
            mincnt=1,
            linewidths=0,
        )
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("log₁₀(count)")
        lo = float(min(d[obs_col].min(), d[pred_col].min()))
        hi = float(max(d[obs_col].max(), d[pred_col].max()))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Observed LAI")
    ax.set_ylabel("Predicted LAI")
    ax.set_title(title)
    _clean(ax)
    annotate_metrics(ax, metrics)
    savefig(fig, out)


def _pick_site_years(pred: pd.DataFrame, n: int = 2) -> List[Tuple[str, int]]:
    """Deterministically pick the best and worst site-years by RMSE."""
    d = pred[["site_id", "year", "lai_obs", "lai_pred"]].dropna()
    if d.empty:
        return []
    d = d.assign(err2=(d["lai_pred"] - d["lai_obs"]) ** 2)
    g = d.groupby(["site_id", "year"], sort=True)["err2"].agg(["mean", "size"])
    g = g[g["size"] >= 12]
    if g.empty:
        g = d.groupby(["site_id", "year"], sort=True)["err2"].mean().to_frame("mean")
        g["size"] = 1
    g = g.sort_values("mean")
    picks = [tuple(g.index[0])]
    if len(g) > 1 and n > 1:
        picks.append(tuple(g.index[-1]))
    return [(str(s), int(y)) for s, y in picks[:n]]


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


def plot_paper_site_year_curves(
    pred: pd.DataFrame,
    out: str,
    choices: Optional[Sequence[Tuple[str, int]]] = None,
    title: str = "Contrasted site-years",
    pft_df: Optional[pd.DataFrame] = None,
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
        Explicit site-years.  Defaults to the best/worst RMSE site-years.
    title : str
        Figure title.
    pft_df : pandas.DataFrame, optional
        Long per-PFT table (``site_id``, ``year``, ``pft``, ``pft_name``,
        ``frac``).  When given, the top three PFT fractions are shown.
    """
    apply_style()
    if choices is None:
        choices = _pick_site_years(pred, 2)
    if not choices:
        raise ValueError("no site-year with valid observations")
    fig, axes = plt.subplots(
        1, len(choices), figsize=(5.6 * len(choices), 4.0), squeeze=False
    )
    for ax, (site, year) in zip(axes[0], choices):
        g = pred[(pred["site_id"].astype(str) == str(site)) & (pred["year"] == year)]
        g = g.sort_values("doy")
        if g.empty:
            ax.set_visible(False)
            continue
        ax.plot(g["doy"], g["lai_obs"], "o", ms=4, color="k", label="Observed")
        ax.plot(
            g["doy"],
            g["lai_pred"],
            "-",
            lw=2,
            color=PALETTE["orange"],
            label="Predicted",
        )
        if "lai_clim" in g.columns and g["lai_clim"].notna().any():
            ax.plot(
                g["doy"],
                g["lai_clim"],
                "--",
                lw=1.5,
                color=PALETTE["grey"],
                label="Climatology",
            )
        ax.set_title(f"{site} — {year}")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("LAI")
        _clean(ax)
        ax.legend(loc="best")

        r2, rmse = _site_year_scores(g)
        if np.isfinite(rmse):
            score = (
                f"R² = {r2:.3f}\nRMSE = {rmse:.3f}"
                if np.isfinite(r2)
                else f"RMSE = {rmse:.3f}"
            )
            ax.text(
                0.98,
                0.02,
                score,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85),
            )

        composition = _composition_text(pft_df, site, year)
        if composition:
            ax.text(
                0.02,
                0.02,
                composition,
                transform=ax.transAxes,
                fontsize=7,
                va="bottom",
            )
        elif "pft_dom_name" in g.columns:
            ax.text(
                0.02,
                0.02,
                f"dominant PFT: {g['pft_dom_name'].iloc[0]}"
                + (
                    f" (purity {g['purity'].iloc[0]:.2f})"
                    if "purity" in g.columns and np.isfinite(g["purity"].iloc[0])
                    else ""
                ),
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )
    fig.suptitle(title)
    fig.tight_layout()
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
        ax.hexbin(
            d[obs_col],
            d[pred_col],
            gridsize=70,
            bins="log",
            cmap="magma",
            mincnt=1,
            linewidths=0,
        )
        lim = float(np.nanpercentile(np.abs(d.values), 99.5))
        lim = max(lim, 0.05)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Observed ΔLAI")
    ax.set_ylabel("Predicted ΔLAI")
    ax.set_title("Anomaly skill")
    _clean(ax)
    an = metrics.get("anomaly") if isinstance(metrics, dict) else None
    annotate_metrics(ax, an or metrics)
    savefig(fig, out)


def plot_paper_rmse_by_doy(
    rmse: pd.DataFrame,
    out: str,
    title: str = "Error by day of year",
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
        )
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax_skill = None

    ax.plot(rmse["doy"], rmse["rmse_model"], color=PALETTE["orange"], label="Model")
    if has_clim:
        ax.plot(
            rmse["doy"],
            rmse["rmse_clim"],
            "--",
            color=PALETTE["grey"],
            label="Climatology",
        )
        better = rmse["rmse_model"] < rmse["rmse_clim"]
        ax.fill_between(
            rmse["doy"],
            0,
            rmse["rmse_model"].max() * 1.05,
            where=better,
            color=PALETTE["green"],
            alpha=0.12,
            lw=0,
            label="Model beats climatology",
        )
    ax.set_ylabel("RMSE (LAI)")
    ax.set_title(title)
    ax.set_xlim(0, 365)
    _clean(ax)
    ax.legend(loc="upper right")

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

    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="grey", lw=0.8)
    labels = [s for s in SCENARIO_ORDER if s in frames]
    labels += [s for s in frames if s not in SCENARIO_ORDER]
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
        ax.plot(g.index, g["mean"], color=color, label=label)
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
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, out)


def plot_paper_warming_co2(df: pd.DataFrame, out: str) -> None:
    """F10: ΔLAI and phenology shifts versus warming, one line per CO2 mode."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    d = df.copy()
    if "co2_mode" not in d.columns:
        d["co2_mode"] = "absolute"
    for mode, g in d.groupby("co2_mode"):
        g = g.sort_values("delta_t")
        axes[0].plot(g["delta_t"], g["mean_delta_lai"], "o-", label=mode)
        if "median_dsos_10" in g.columns:
            axes[1].plot(g["delta_t"], g["median_dsos_10"], "o-", label=mode)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("ΔT (°C)")
    axes[0].set_ylabel("Mean ΔLAI")
    axes[0].set_title("LAI response")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("ΔT (°C)")
    axes[1].set_ylabel("ΔSOS (days)")
    axes[1].set_title("Phenology response")
    for ax in axes:
        _clean(ax)
        ax.legend(loc="best")
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = plt.cm.tab20(np.linspace(0, 1, 15))
    for pft, h in g.groupby("pft"):
        h = h.sort_values("doy")
        frac = float(h["frac"].iloc[0])
        if frac < min_frac:
            ax.plot(h["doy"], h["lai_pure"], color="lightgrey", lw=0.7, zorder=1)
        else:
            ax.plot(
                h["doy"],
                h["lai_pure"],
                color=colors[int(pft)],
                lw=1.8,
                label=f"{h['pft_name'].iloc[0]} ({frac * 100:.0f}%)",
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
        color=PALETTE["orange"],
        ms=4,
        label="Observed",
    )
    ax.set_xlabel("Day of year")
    ax.set_ylabel("LAI")
    ax.set_title(f"Per-PFT decomposition — {site_id} / {year}")
    ax.set_xlim(0, 365)
    _clean(ax)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
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
    """
    apply_style()
    groups = _history_groups(histories)
    if not groups:
        raise ValueError("no training history to plot")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (label, frames) in enumerate(groups.items()):
        color = MODEL_COLORS.get(label, default_colors[index % len(default_colors)])
        train = _mean_curve(frames, "train_loss")
        valid = _mean_curve(frames, "val_loss")

        if not train.empty:
            axes[0].plot(train.index, train["mean"], color=color, label=label)
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
            axes[1].plot(valid.index, valid["mean"], color=color, label=label)
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

        if early_stop_epochs:
            stop = early_stop_epochs.get(label)
            if stop is not None:
                for axis in axes:
                    axis.axvline(stop, color=color, ls=":", lw=1.2)

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss (dot = best epoch)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        _clean(axis)
        axis.legend(loc="best")
    fig.tight_layout()
    savefig(fig, out)


def plot_paper_memory_kernel(
    kernel: pd.DataFrame, controls: Optional[pd.DataFrame], out: str
) -> None:
    """F13: memory kernel with the random-day control band."""
    apply_style()
    if kernel.empty:
        raise ValueError("empty memory kernel")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        kernel["lag_days"],
        kernel["mean_abs_delta"],
        "o-",
        color=PALETTE["blue"],
        label="Impulse response",
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
            ax.text(
                0.98,
                0.02,
                f"linearity ratio = {float(values['linearity']):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Mean |ΔLAI|")
    ax.set_title("Memory kernel")
    _clean(ax)
    ax.legend(loc="best")
    fig.tight_layout()
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
