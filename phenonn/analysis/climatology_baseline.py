#!/usr/bin/env python3
"""
Climatology baseline comparison.

Tests whether the model has learned inter-annual information by comparing
it against a trivial climatology baseline:

    clim(site, doy) = mean over years of  lai_obs(site, year, doy)

The climatology knows nothing about inter-annual variability — it predicts
the same value for every year at a given (site, DOY). So:
  - if the model's RMSE ≤ climatology's RMSE → the model has learned
    nothing beyond the average annual cycle.
  - if the model's RMSE <  climatology's RMSE → the model captures some
    real inter-annual signal (skill score > 0).

Skill Score
-----------
    SS = 1 − MSE_model / MSE_clim

      SS = 1   : perfect prediction
      SS > 0   : model beats climatology
      SS = 0   : equivalent — model just learned the climatology
      SS < 0   : model worse than climatology  (bad sign)

Leave-one-year-out (default)
----------------------------
By default, the climatology used to predict year y is computed from all
years EXCEPT y at that (site, DOY). This prevents the baseline from
"seeing" the answer it is trying to predict.

The --pooled flag uses every year (including the target year) — faster but
slightly optimistic for the baseline.

Output
------
  *_clim_per_site.csv         — per-site metrics (model vs clim)
  *_clim_scatter.png          — side-by-side scatter:  model | climatology
  *_clim_skill_hist.png       — distribution of per-site skill scores
  *_clim_skill_per_doy.png    — skill score by DOY: where in the year does the
                                model actually add value?
  *_clim_lai_curves.png       — (if --n_curves ≥ 3) climatology vs obs annual
                                curves for 3 representative sites (low/mid/high
                                R² within the sampled subset)
  *_clim_lai_curves_all.png   — (if --n_curves > 0) grid of climatology vs obs
                                annual curves for the n_curves sampled sites

Usage
-----
    cd LAI/
    python -m prediction.climatology_baseline \\
        --predictions runs/exp_flat/predictions_flat.csv \\
        --output      runs/exp_flat/clim_baseline

    # use pooled climatology (faster, slightly favours the baseline):
    python -m prediction.climatology_baseline \\
        --predictions runs/exp_flat/predictions_flat.csv \\
        --pooled
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from phenonn.utils.diagnostics import (
    _compute_metrics,
    _save,
    plot_gcc_curves,
    plot_gcc_curves_all,
)

_CMAP_YELLOW_RED = LinearSegmentedColormap.from_list(
    "yellow_red", ["#FFD700", "#FF4500", "#CC0000"]
)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Climatology baseline comparison for LAI predictions"
    )
    p.add_argument("--predictions", required=True,
                   help="Path to predictions_flat.csv")
    p.add_argument("--output", default="",
                   help="Output base path "
                        "(default: same dir as predictions, stem 'clim')")
    p.add_argument("--pooled", action="store_true",
                   help="Use pooled climatology (all years incl. target). "
                        "Default: leave-one-year-out.")
    p.add_argument("--min_years", type=int, default=2,
                   help="Minimum years per site to evaluate the baseline "
                        "(default: 2 — needs ≥ 2 to leave one out)")
    p.add_argument("--clim_years", default="",
                   help="Years used to compute the climatology mean per (site, "
                        "DOY). E.g. '1992-2010' or '2000,2001,2005'. When given, "
                        "the climatology is a fixed reference period applied "
                        "uniformly to every evaluated row (LOO is disabled). "
                        "Default: use all years in the predictions CSV with "
                        "--pooled or LOO behavior.")
    p.add_argument("--eval_years", default="",
                   help="Years used for evaluation (metrics + plots). E.g. "
                        "'2011-2018'. Default: all years in the predictions CSV.")
    p.add_argument("--clim_target_dir", default="",
                   help="Folder containing target_{year}.csv files for the "
                        "climatology reference period. Use when --clim_years "
                        "are outside the predictions CSV (typical: historical "
                        "clim vs. future evaluation). The script loads "
                        "observations from these files filtered to the sites "
                        "present in the predictions CSV.")
    p.add_argument("--n_curves", type=int, default=0,
                   help="If > 0, plot climatology-vs-obs annual LAI curves for "
                        "n_curves sites sampled to cover the per-site R² "
                        "distribution. Produces *_clim_lai_curves_all.png "
                        "(grid). When n_curves ≥ 3, also produces "
                        "*_clim_lai_curves.png with 3 representative sites "
                        "(low/medium/high R² within the sampled subset). "
                        "Default 0 = skip curve plots.")
    p.add_argument("--n_curves_seed", type=int, default=42,
                   help="Seed for the n_curves site sampling and the "
                        "tercile picks. Default 42.")
    p.add_argument("--pft_dir", default="",
                   help="PFT pixelset dir (PFTmap_{year}.nc). When set, each "
                        "*_clim_lai_curves_all.png subplot title lists the PFTs "
                        "with fraction ≥ --pft_min_frac.")
    p.add_argument("--pft_min_frac", type=float, default=0.05,
                   help="Show PFTs with fraction ≥ this threshold above each "
                        "curve subplot (default: 0.05 = 5%%).")
    return p.parse_args()


def _parse_year_spec(s: str):
    """Parse '1992-2010' or '2000,2001' into a set of ints. Empty → None."""
    if not s:
        return None
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    return set(int(y) for y in s.split(","))


# ── Climatology ───────────────────────────────────────────────────────────────


def _load_clim_source_from_targets(
    target_dir: str,
    clim_years,
    wanted_sites,
) -> pd.DataFrame:
    """
    Load LAI observations for `clim_years` from PhenoNN's NetCDF target
    cubes (`LAI_dekadal_{year}.nc`), filtered to `wanted_sites`.

    Returns a DataFrame with columns (site_id, doy, lai_obs).
    """
    import datetime
    import xarray as xr
    from phenonn.utils.config import TARGETS_FNAME, DEKAD_DAYS

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"--clim_target_dir {target_dir!r} not found")

    wanted = sorted(set(wanted_sites))
    if not wanted:
        return pd.DataFrame(columns=["site_id", "doy", "lai_obs"])

    # 36 (month, day, doy) tuples of a non-leap year, matching the dekad
    # index produced by preprocess_targets.py.
    obs_dates = [
        (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
        for m in range(1, 13) for d in DEKAD_DAYS
    ]
    doys = np.array([t[2] for t in obs_dates], dtype=np.int16)   # (36,)

    parts = []
    for year in sorted(clim_years):
        path = os.path.join(target_dir, TARGETS_FNAME.format(year=year))
        if not os.path.exists(path):
            print(f"  [warn] missing {os.path.basename(path)} — skipping")
            continue
        ds = xr.open_dataset(path)
        # Pixelset LAI(dekad, site): align rows to `wanted` by site_id
        # (missing sites stay NaN).
        all_sites = np.asarray(ds["site_id"].values).astype(str)
        lai = ds["LAI"].transpose("site", "dekad").values.astype(np.float32)
        ds.close()
        idx_of = {s: i for i, s in enumerate(all_sites)}
        rows = np.array([idx_of.get(s, -1) for s in wanted], dtype=np.int64)
        vals = np.full((len(wanted), lai.shape[1]), np.nan, dtype=np.float32)
        ok = rows >= 0
        vals[ok] = lai[rows[ok]]
        n_sites = len(wanted)
        df = pd.DataFrame({
            "site_id": np.repeat(wanted, len(doys)),
            "doy":     np.tile(doys, n_sites),
            "lai_obs": vals.reshape(-1),
        })
        parts.append(df)

    if not parts:
        raise RuntimeError(
            f"No target data found in {target_dir!r} for years "
            f"{sorted(clim_years)}. Check the folder and clim_years range."
        )

    return pd.concat(parts, ignore_index=True)


def add_climatology(
    df: pd.DataFrame,
    pooled: bool,
    clim_years=None,
    clim_target_dir: str = "",
) -> pd.DataFrame:
    """
    Add a `lai_clim` column to df.

    Three modes:
      - clim_years given  : `lai_clim` = mean over `clim_years` of LAI obs at
                            (site, doy). Same value applied to every row of
                            that (site, doy), regardless of the row's year.
                            LOO is disabled in this mode.

        The source of observations for the mean is:
          * `df` itself, if all clim_years appear in df["year"]; or
          * external `target_{year}.csv` files in `clim_target_dir` (handy
            when the predictions CSV only covers the eval years).

      - pooled            : clim = mean over ALL years in df.
      - leave-one-year-out: clim for row at year y = mean over years ≠ y.

    Rows where the climatology cannot be computed (no obs at that (site, doy)
    in the source, or only 1 year in LOO mode) get NaN.
    """
    # ── Fixed reference period (clim_years) ──
    if clim_years is not None:
        years_in_df = set(int(y) for y in df["year"].unique())
        if not clim_years.issubset(years_in_df):
            # Need an external source for the missing years.
            if not clim_target_dir:
                missing = sorted(clim_years - years_in_df)
                raise RuntimeError(
                    f"--clim_years {sorted(clim_years)} are not all in the "
                    f"predictions CSV (missing: {missing}, available: "
                    f"{sorted(years_in_df)}). Either rerun prediction_big.py "
                    f"with --predict_years covering the clim period, or pass "
                    f"--clim_target_dir to load LAI observations directly "
                    f"from your target_{{year}}.csv archive."
                )
            print(f"Loading climatology source from {clim_target_dir!r} for "
                  f"{len(clim_years)} year(s)...")
            source = _load_clim_source_from_targets(
                clim_target_dir, clim_years, df["site_id"].unique(),
            )
        else:
            # Predictions CSV covers the clim years — use it as source.
            source = df.loc[df["year"].isin(clim_years), ["site_id", "doy", "lai_obs"]]

        if source.empty:
            raise RuntimeError(
                f"Climatology source is empty for --clim_years "
                f"{sorted(clim_years)}."
            )

        clim_lookup = (
            source.groupby(["site_id", "doy"])["lai_obs"]
                  .mean()  # NaN-aware
                  .reset_index(name="lai_clim")
        )
        df = df.merge(clim_lookup, on=["site_id", "doy"], how="left")
        return df

    g = df.groupby(["site_id", "doy"])["lai_obs"]

    if pooled:
        df = df.copy()
        df["lai_clim"] = g.transform("mean")
        return df

    total = g.transform("sum")
    count = g.transform("count")
    denom = count - 1
    loo = np.where(denom > 0,
                   (total - df["lai_obs"]) / denom.replace(0, np.nan),
                   np.nan)
    df = df.copy()
    df["lai_clim"] = loo
    return df


# ── Per-site metrics ──────────────────────────────────────────────────────────


def per_site_metrics(df: pd.DataFrame, min_years: int) -> pd.DataFrame:
    """
    For each site, compute model vs obs and climatology vs obs metrics
    plus the skill score.
    """
    rows = []
    for site, g in df.groupby("site_id"):
        n_years = g["year"].nunique()
        if n_years < min_years:
            continue

        sub = g[["lai_obs", "lai_pred", "lai_clim"]].dropna()
        if len(sub) < 2:
            continue

        obs   = sub["lai_obs"].values
        pred  = sub["lai_pred"].values
        clim  = sub["lai_clim"].values

        m_pred = _compute_metrics(pred, obs)
        m_clim = _compute_metrics(clim, obs)

        mse_pred = float(np.mean((pred - obs) ** 2))
        mse_clim = float(np.mean((clim - obs) ** 2))
        ss = 1.0 - mse_pred / mse_clim if mse_clim > 1e-12 else float("nan")

        rows.append({
            "site_id"        : site,
            "n_years"        : n_years,
            "n_points"       : len(sub),
            "rmse_model"     : m_pred["rmse"],
            "rmse_clim"      : m_clim["rmse"],
            "mae_model"      : m_pred["mae"],
            "mae_clim"       : m_clim["mae"],
            "r2_model"       : m_pred["r2"],
            "r2_clim"        : m_clim["r2"],
            "bias_model"     : m_pred["bias"],
            "bias_clim"      : m_clim["bias"],
            "skill_score"    : ss,
            "model_beats_clim": bool(mse_pred < mse_clim),
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_scatter_comparison(df: pd.DataFrame, filename: str) -> None:
    """
    Pred vs obs hexbin: model on the left, climatology on the right.

    Visual style matches `phenonn.utils.diagnostics.plot_pred_vs_obs` (also used by
    prediction_big.py): viridis density with log binning, upper-left text
    box containing R² / RMSE / MAE / Bias / N, 1:1 diagonal, equal aspect.
    """
    sub = df[["lai_obs", "lai_pred", "lai_clim"]].dropna()
    obs   = sub["lai_obs"].values
    pred  = sub["lai_pred"].values
    clim  = sub["lai_clim"].values

    lo = float(min(obs.min(), pred.min(), clim.min()))
    hi = float(max(obs.max(), pred.max(), clim.max()))
    pad = (hi - lo) * 0.05 or 0.1
    axmin, axmax = lo - pad, hi + pad

    m_pred = _compute_metrics(pred, obs)
    m_clim = _compute_metrics(clim, obs)

    mse_pred = float(np.mean((pred - obs) ** 2))
    mse_clim = float(np.mean((clim - obs) ** 2))
    ss = 1.0 - mse_pred / mse_clim if mse_clim > 1e-12 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    for ax, vals, m, label in [
        (axes[0], pred, m_pred, "Model"),
        (axes[1], clim, m_clim, "Climatology"),
    ]:
        hb = ax.hexbin(obs, vals, gridsize=70, cmap="viridis",
                       mincnt=1, bins="log")
        cb = fig.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label("log₁₀(count)")

        ax.plot([axmin, axmax], [axmin, axmax], "k--", lw=1.2,
                alpha=0.7, label="1:1")
        ax.set_xlim(axmin, axmax)
        ax.set_ylim(axmin, axmax)
        ax.set_aspect("equal")

        txt = (
            f"$R^2$ = {m['r2']:.4f}\n"
            f"RMSE = {m['rmse']:.4f}\n"
            f"MAE  = {m['mae']:.4f}\n"
            f"Bias = {m['bias']:+.4f}\n"
            f"N    = {m['n']:,}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                va="top", ha="left", family="monospace", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9,
                          boxstyle="round,pad=0.4"))

        ax.set_xlabel("Observed LAI")
        ax.set_ylabel(f"{label} LAI")
        ax.set_title(f"{label} vs observed")
        ax.legend(loc="lower right")

    fig.suptitle(
        f"Model vs Climatology baseline    "
        f"Skill Score = {ss:+.3f}    "
        f"({'model beats climatology' if ss > 0 else 'model does NOT beat climatology'})",
        fontsize=12,
    )
    _save(fig, filename)


def plot_skill_histogram(site_df: pd.DataFrame, filename: str) -> None:
    """Distribution of per-site skill scores."""
    ss = site_df["skill_score"].dropna().values
    n_sites = len(ss)
    n_win   = int((ss > 0).sum())
    median  = float(np.median(ss))

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    # clip very negative scores for plotting; show real count in title
    plot_ss = np.clip(ss, -1.0, 1.0)
    ax.hist(plot_ss, bins=60, color="steelblue", edgecolor="white",
            linewidth=0.4)

    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(median, color="red", linewidth=1.2, linestyle="--",
               label=f"median = {median:+.3f}")

    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Skill Score   (1 − MSE_model / MSE_clim)")
    ax.set_ylabel("Number of sites")
    ax.set_title(
        f"Per-site skill against climatology   "
        f"{n_win} / {n_sites} sites beat the baseline "
        f"({100 * n_win / max(n_sites, 1):.1f}%)"
    )
    ax.legend()
    _save(fig, filename)


def _load_pft_fracs(pft_dir: str, year: int, site_ids) -> dict:
    """{site_id: (N_PFT,) fraction vector} from PFTmap_{year}.nc, aligned by
    site_id. Empty dict if pft_dir unset or the file is missing (torch-free)."""
    if not pft_dir:
        return {}
    import xarray as xr
    from phenonn.utils.config import PFT_FNAME
    path = os.path.join(pft_dir, PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        print(f"  [n_curves] PFT file not found ({path}) — titles without PFT.")
        return {}
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
    all_sites = np.asarray(ds["site_id"].values).astype(str)
    arr = da.transpose("pft", "site").values.astype(np.float32)   # (N_PFT, n_all)
    ds.close()
    idx_of = {s: i for i, s in enumerate(all_sites)}
    out = {}
    for s in map(str, site_ids):
        j = idx_of.get(s, -1)
        if j >= 0:
            out[s] = arr[:, j]
    return out


def plot_clim_lai_curves(
    df: pd.DataFrame,
    n_curves: int,
    base: str,
    seed: int = 42,
    pft_dir: str = "",
    pft_year: int = None,
    pft_min_frac: float = 0.05,
) -> None:
    """
    Plot annual LAI curves for `n_curves` sites, overlaying three series per
    year: observation, model prediction (lai_pred) and climatology (lai_clim).

    Reuses `phenonn.utils.diagnostics.plot_gcc_curves` / `plot_gcc_curves_all`, passing
    the climatology via the `extra_col` argument (dotted line) so it appears
    alongside obs (solid) and the model prediction (dashed). Same style as the
    `*_lai_curves.png` figures produced by predict.py.

    Site selection
    --------------
    Per-site R² is computed for the climatology. Sites are sorted by R²,
    then `n_curves` of them are picked at evenly spaced ranks across the
    sorted list, so the sample covers the whole R² distribution rather
    than only the best or worst sites.

    Produced files
    --------------
      base + "_clim_lai_curves_all.png" : grid of n_curves sites.
      base + "_clim_lai_curves.png"     : 3 representative sites
                                          (low/medium/high R² *within*
                                          the n_curves subset).
                                          Only when n_curves >= 3.
    """
    sub = df[["site_id", "year", "doy", "lai_obs", "lai_pred", "lai_clim"]].dropna()
    if sub.empty:
        print("  [n_curves] No rows with both obs and climatology — skipping.")
        return

    # ── Per-site R² of climatology ──
    r2_by_site = {}
    for site, g in sub.groupby("site_id"):
        o = g["lai_obs"].values
        c = g["lai_clim"].values
        if len(o) < 10:
            continue
        ss_res = np.sum((o - c) ** 2)
        ss_tot = np.sum((o - o.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        r2_by_site[site] = r2

    if not r2_by_site:
        print("  [n_curves] No site has ≥10 valid (obs, clim) pairs — skipping.")
        return

    # Sort by R² ascending; fall NaN to the very bottom so they don't get picked
    # unless we ask for everything.
    sorted_sites = sorted(
        r2_by_site.items(),
        key=lambda x: x[1] if np.isfinite(x[1]) else float("-inf"),
    )
    total = len(sorted_sites)

    n_take = min(n_curves, total)
    if n_take <= 0:
        return

    # Evenly-spaced ranks across [0, total-1]
    if n_take == 1:
        picks_idx = [total // 2]
    else:
        picks_idx = np.linspace(0, total - 1, n_take).round().astype(int).tolist()
        picks_idx = sorted(set(picks_idx))   # de-dup if total < n_curves

    picked = [sorted_sites[i][0] for i in picks_idx]
    sub_sites = sub[sub["site_id"].isin(picked)].copy()

    print(f"  [n_curves] Plotting {len(picked)} site(s) "
          f"(R² range: {sorted_sites[picks_idx[0]][1]:+.3f} → "
          f"{sorted_sites[picks_idx[-1]][1]:+.3f})")

    # ── PFT fractions per picked site (for the subplot titles) ──
    from phenonn.utils.config import PFT_NAMES
    pft_by_site = (_load_pft_fracs(pft_dir, pft_year, picked)
                   if pft_year is not None else {})

    # ── Grid figure ──
    plot_gcc_curves_all(
        sub_sites,
        filename=base + "_clim_lai_curves_all.png",
        site_col="site_id",
        year_col="year",
        doy_col="doy",
        pred_col="lai_pred",
        obs_col="lai_obs",
        extra_col="lai_clim",
        extra_label="climatology",
        pft_by_site=pft_by_site,
        pft_names=PFT_NAMES,
        pft_min_frac=pft_min_frac,
    )

    # ── 3-site low/medium/high figure (only if we have at least 3 sites) ──
    if len(picked) >= 3:
        plot_gcc_curves(
            sub_sites,
            filename=base + "_clim_lai_curves.png",
            seed=seed,
            site_col="site_id",
            year_col="year",
            doy_col="doy",
            pred_col="lai_pred",
            obs_col="lai_obs",
            extra_col="lai_clim",
            extra_label="climatology",
        )


def plot_skill_per_doy(df: pd.DataFrame, filename: str) -> None:
    """
    Skill score computed per DOY, pooled across all sites and years.
    Shows where in the annual cycle the model adds value (or doesn't).
    """
    sub = df[["doy", "lai_obs", "lai_pred", "lai_clim"]].dropna()

    rows = []
    for doy, g in sub.groupby("doy"):
        obs  = g["lai_obs"].values
        pred = g["lai_pred"].values
        clim = g["lai_clim"].values
        mse_pred = float(np.mean((pred - obs) ** 2))
        mse_clim = float(np.mean((clim - obs) ** 2))
        ss = 1.0 - mse_pred / mse_clim if mse_clim > 1e-12 else np.nan
        rows.append({
            "doy"     : int(doy),
            "ss"      : ss,
            "rmse_model": float(np.sqrt(mse_pred)),
            "rmse_clim" : float(np.sqrt(mse_clim)),
        })
    per_doy = pd.DataFrame(rows).sort_values("doy")

    fig, (ax_rmse, ax_ss) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.1},
        constrained_layout=True,
    )

    ax_rmse.plot(per_doy["doy"], per_doy["rmse_model"],
                 color="darkorange", linewidth=1.6, label="Model")
    ax_rmse.plot(per_doy["doy"], per_doy["rmse_clim"],
                 color="steelblue", linewidth=1.6, linestyle="--",
                 label="Climatology")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("RMSE per observation DOY — pooled across sites")
    ax_rmse.legend()

    ax_ss.axhline(0, color="black", linewidth=0.8)
    ax_ss.fill_between(per_doy["doy"], 0, per_doy["ss"],
                       where=per_doy["ss"] > 0,
                       color="green", alpha=0.25, label="model wins")
    ax_ss.fill_between(per_doy["doy"], 0, per_doy["ss"],
                       where=per_doy["ss"] <= 0,
                       color="red",   alpha=0.25, label="clim wins")
    ax_ss.plot(per_doy["doy"], per_doy["ss"],
               color="black", linewidth=1.2)
    ax_ss.set_xlabel("Day of year")
    ax_ss.set_ylabel("Skill Score")
    ax_ss.legend(loc="lower left")

    _save(fig, filename)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    base = (args.output.rstrip("/\\") if args.output
            else os.path.join(os.path.dirname(os.path.abspath(args.predictions)),
                              "clim"))

    clim_years = _parse_year_spec(args.clim_years)
    eval_years = _parse_year_spec(args.eval_years)

    if clim_years is not None:
        mode = f"fixed reference period ({min(clim_years)}–{max(clim_years)})"
    else:
        mode = "pooled" if args.pooled else "leave-one-year-out"

    # ── Load ──
    df = pd.read_csv(args.predictions)
    n_sites = df["site_id"].nunique()
    n_years = df["year"].nunique()
    print(f"Loaded {len(df):,} rows — {n_sites} sites, {n_years} years "
          f"({sorted(df['year'].unique())})")
    print(f"Climatology mode: {mode}")
    if eval_years is not None:
        print(f"Eval years      : {sorted(eval_years)}")

    # ── Compute climatology BEFORE filtering eval_years ──
    # We need the full df so the lookup can include observations from years
    # outside the eval window (typical use case: clim from past, eval on future).
    df = add_climatology(
        df,
        pooled=args.pooled,
        clim_years=clim_years,
        clim_target_dir=args.clim_target_dir,
    )
    n_with_clim = df["lai_clim"].notna().sum()
    print(f"Rows with climatology: {n_with_clim:,} / {len(df):,}")

    # ── Filter to eval_years AFTER climatology is computed ──
    if eval_years is not None:
        before = len(df)
        df = df[df["year"].isin(eval_years)].copy()
        print(f"Filtered to eval years: {len(df):,} / {before:,} rows kept")
        if df.empty:
            raise RuntimeError(
                f"--eval_years {sorted(eval_years)} do not match any row "
                f"in the CSV. Available: {sorted(df['year'].unique().tolist())}"
            )

    # ── Global metrics ──
    sub = df[["lai_obs", "lai_pred", "lai_clim"]].dropna()
    obs   = sub["lai_obs"].values
    pred  = sub["lai_pred"].values
    clim  = sub["lai_clim"].values

    m_pred = _compute_metrics(pred, obs)
    m_clim = _compute_metrics(clim, obs)
    mse_pred = float(np.mean((pred - obs) ** 2))
    mse_clim = float(np.mean((clim - obs) ** 2))
    ss_global = 1.0 - mse_pred / mse_clim if mse_clim > 1e-12 else float("nan")

    print("\n── Global metrics ────────────────────────────────────────")
    print(f"  Model         RMSE={m_pred['rmse']:.4f}  MAE={m_pred['mae']:.4f}  "
          f"R²={m_pred['r2']:+.4f}  bias={m_pred['bias']:+.4f}")
    print(f"  Climatology   RMSE={m_clim['rmse']:.4f}  MAE={m_clim['mae']:.4f}  "
          f"R²={m_clim['r2']:+.4f}  bias={m_clim['bias']:+.4f}")
    print(f"  Skill Score   {ss_global:+.4f}   "
          f"({'✓ model beats climatology' if ss_global > 0 else '✗ model does NOT beat climatology'})")

    # ── Per-site metrics ──
    site_df = per_site_metrics(df, min_years=args.min_years)
    if len(site_df) > 0:
        n_win = int(site_df["model_beats_clim"].sum())
        n_tot = len(site_df)
        med_ss = float(site_df["skill_score"].median())
        print(f"\n── Per-site summary (sites with ≥ {args.min_years} years) ──")
        print(f"  Sites evaluated         : {n_tot}")
        print(f"  Sites where model wins  : {n_win} ({100 * n_win / n_tot:.1f}%)")
        print(f"  Median skill score      : {med_ss:+.4f}")
        print(f"  Mean   skill score      : {float(site_df['skill_score'].mean()):+.4f}")
        print(f"  Median MAE  (model)     : {float(site_df['mae_model'].median()):.4f}")
        print(f"  Median MAE  (clim)      : {float(site_df['mae_clim'].median()):.4f}")

        csv_path = base + "_clim_per_site.csv"
        site_df.to_csv(csv_path, index=False)
        print(f"  Per-site metrics → {csv_path}")

    # ── Plots ──
    plot_scatter_comparison(df, base + "_clim_scatter.png")
    if len(site_df) > 0:
        plot_skill_histogram(site_df, base + "_clim_skill_hist.png")
    plot_skill_per_doy(df, base + "_clim_skill_per_doy.png")

    if args.n_curves > 0:
        plot_clim_lai_curves(
            df, n_curves=args.n_curves, base=base, seed=args.n_curves_seed,
            pft_dir=args.pft_dir, pft_year=int(df["year"].min()),
            pft_min_frac=args.pft_min_frac,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
