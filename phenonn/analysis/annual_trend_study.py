#!/usr/bin/env python3
"""
Annual phenological trend study from LAI predictions and/or observations.

For each (site, year) curve, computes 5 metrics:

  LAI_tot   sum of the 36 dekadal LAI values
  SOS_10    Start Of Season — first DOY where LAI >= min + 0.10 · (max − min),
            considering only the rising phase (DOY <= argmax)
  EOS_10    End   Of Season — last  DOY where LAI >= min + 0.10 · (max − min),
            considering only the falling phase (DOY >= argmax)
  SOS_25    same as SOS_10 with 25 % of amplitude
  EOS_25    same as EOS_10 with 25 % of amplitude

For each metric, one PNG file is produced. The x-axis is year and each curve
shows the median across the selected sites with a shaded inter-quartile (IQR)
envelope. When predictions are provided alongside observations, both curves
are drawn so visual comparison of inter-annual variability is direct.

Input
-----
Two mutually-compatible modes — pick whichever is most convenient:

  --predictions runs/exp/predictions_big.csv
        Reads a predictions CSV (output of prediction_big.py or
        xgb_predict.py) which has columns:
            site_id, year, doy, lai_obs, lai_pred
        Both obs and pred curves appear in the plots.

  --target_dir LAI/data/target_years
        Reads raw target_{YYYY}.csv files. Obs-only mode (no pred curve).

Site selection
--------------
The script does NOT couple to a checkpoint. To restrict to validation or
test sites, run prediction_big.py with `--predict_sites val|test` first;
its output CSV already contains exactly the sites you want. Then feed that
CSV to this script.

  --sites_mode all       (default)  use every site present in the input
  --sites_mode random    random subsample of --n  --sites_mode explicit  use the comma-separated --sites list
_sites sites

Usage
-----
    # Validation set, obs + pred
    python -m prediction.predict --predict_sites val \\
        --output_csv runs/exp/preds_val.csv
    python -m util.annual_trend_study \\
        --predictions runs/exp/preds_val.csv \\
        --output_dir  runs/exp/trends_val

    # Test set (overlap mode)
    python -m prediction.predict --predict_sites test \\
        --output_csv runs/exp/preds_test.csv
    python -m util.annual_trend_study \\
        --predictions runs/exp/preds_test.csv \\
        --output_dir  runs/exp/trends_test

    # Obs-only baseline on a random subsample of the raw archive
    python -m util.annual_trend_study \\
        --target_dir LAI/data/target_years \\
        --sites_mode random --n_sites 500 \\
        --output_dir  runs/baseline_trends
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Annual phenological trends — LAI_tot, SOS, EOS",
    )
    p.add_argument("--predictions", default="",
                   help="predictions CSV with columns site_id, year, doy, "
                        "lai_obs, lai_pred. Output of prediction_big.py or "
                        "xgb_predict.py.")
    p.add_argument("--target_dir", default="",
                   help="Folder of target_{Y}.csv (obs-only mode if "
                        "--predictions is empty).")
    p.add_argument("--sites_mode", default="all",
                   choices=["all", "random", "explicit"],
                   help="all = every site in the input; random = subsample of "
                        "--n_sites; explicit = the --sites list.")
    p.add_argument("--n_sites", type=int, default=200,
                   help="Number of sites to sample when sites_mode=random.")
    p.add_argument("--sites", default="",
                   help="Comma-separated site IDs (used when "
                        "sites_mode=explicit).")
    p.add_argument("--years", default="",
                   help="Year filter, e.g. '2001-2018' or '2010,2015'. "
                        "Empty = use every year present in the input.")
    p.add_argument("--clim_years", default="",
                   help="Years used to build the OBSERVATION climatology line "
                        "(e.g. '2001-2005'). For each site the across-year mean "
                        "LAI curve is built, the metric is computed on it, and "
                        "the median over sites is drawn as one horizontal "
                        "reference line (same for every year) on each plot. "
                        "Empty = no climatology line.")
    p.add_argument("--min_valid", type=int, default=34,
                   help="Skip (site, year) curves with fewer than this many "
                        "valid dekads (default: 34).")
    p.add_argument("--min_amplitude", type=float, default=0.3,
                   help="Skip curves whose amplitude (max-min LAI) is below "
                        "this — protects SOS/EOS from noise-dominated flat "
                        "curves.")
    p.add_argument("--output_dir", default="annual_trends",
                   help="Output directory for the 5 PNG files and the raw "
                        "metric tables.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_year_spec(s: str):
    if not s:
        return None
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return set(range(int(a), int(b) + 1))
    return {int(y) for y in s.split(",")}


# ── Loading ──────────────────────────────────────────────────────────────────


def _load_from_predictions(path: str) -> pd.DataFrame:
    """
    Read a predictions CSV produced by prediction_big.py / xgb_predict.py.
    Returns a DataFrame with columns site_id, year, doy, lai_obs, lai_pred.
    """
    df = pd.read_csv(path)
    needed = {"site_id", "year", "doy", "lai_obs", "lai_pred"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns {sorted(missing)}. "
            f"Got {list(df.columns)}."
        )
    return df[["site_id", "year", "doy", "lai_obs", "lai_pred"]].copy()


def _load_from_target_dir(path: str) -> pd.DataFrame:
    """
    Read PhenoNN LAI_dekadal_{YYYY}.nc files into a long DataFrame compatible
    with the rest of this script: columns site_id, year, doy, lai_obs.

    Obs-only mode (lai_pred is filled with NaN downstream by the main()).
    Falls back to the legacy target_{YYYY}.csv loader when the folder
    contains those files instead.
    """
    import datetime
    import xarray as xr

    nc_files = sorted(glob.glob(os.path.join(path, "LAI_dekadal_*.nc")))
    csv_files = sorted(glob.glob(os.path.join(path, "target_*.csv")))

    if nc_files:
        # The 36 dekads of a non-leap year are stored under integer dekad
        # indices 0..35; we precompute their (month, day, doy) tuple.
        obs = [
            (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
            for m in range(1, 13) for d in [5, 15, 25]
        ]
        doys_v = np.array([t[2] for t in obs], dtype=np.int16)
        frames = []
        for f in nc_files:
            ds = xr.open_dataset(f)
            try:
                year = int(os.path.basename(f).split("_")[-1].split(".")[0])
            except Exception:
                year = -1
            # Pixelset LAI(dekad, site): site_id comes straight from the coord.
            lai = ds["LAI"].transpose("site", "dekad").values.astype(np.float32)
            site_ids = np.asarray(ds["site_id"].values).astype(str)  # (n_site,)
            ds.close()
            for k, doy in enumerate(doys_v):
                col = lai[:, k]
                finite = np.isfinite(col)
                if not finite.any():
                    continue
                frames.append(pd.DataFrame({
                    "site_id": site_ids[finite],
                    "year":    int(year),
                    "doy":     int(doy),
                    "lai_obs": col[finite].astype(np.float32),
                }))
        if not frames:
            raise RuntimeError("No valid LAI observations in --target_dir.")
        return pd.concat(frames, ignore_index=True)

    if not csv_files:
        raise FileNotFoundError(
            f"No LAI_dekadal_*.nc or target_*.csv files in {path!r}"
        )
    # Legacy CSV branch (LaiNN format)
    frames = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=["site_id", "date", "year", "LAI"])
        dt = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["doy"] = dt.dt.day_of_year
        frames.append(
            df[["site_id", "year", "doy", "LAI"]]
              .rename(columns={"LAI": "lai_obs"})
        )
    return pd.concat(frames, ignore_index=True)


def _select_sites(df: pd.DataFrame, args) -> pd.DataFrame:
    sites_all = df["site_id"].unique().tolist()
    if args.sites_mode == "all":
        keep = sites_all
    elif args.sites_mode == "random":
        rng = np.random.default_rng(args.seed)
        n = min(args.n_sites, len(sites_all))
        keep = rng.choice(sites_all, size=n, replace=False).tolist()
    elif args.sites_mode == "explicit":
        wanted = {s.strip() for s in args.sites.split(",") if s.strip()}
        if not wanted:
            raise ValueError("sites_mode=explicit but --sites is empty.")
        keep = [s for s in sites_all if s in wanted]
        if not keep:
            raise RuntimeError(
                "None of the --sites IDs matched the input data."
            )
    else:
        raise ValueError(args.sites_mode)
    print(f"Selected {len(keep):,} sites / {len(sites_all):,} in input.")
    return df[df["site_id"].isin(set(keep))].copy()


# ── Metrics ──────────────────────────────────────────────────────────────────


def _sos_eos_for_curve(
    doy: np.ndarray, lai: np.ndarray, p: float,
) -> Tuple[float, float]:
    """
    SOS_p : first DOY in the rising phase where LAI >= min + p·(max − min).
    EOS_p : last  DOY in the falling phase where LAI >= min + p·(max − min).

    Linear interpolation is used between bracketing dekads to give sub-dekadal
    resolution. Inputs must be sorted by DOY and free of NaN.

    Returns (np.nan, np.nan) when the curve is degenerate (amp == 0,
    too few points, …).
    """
    if lai.size < 3:
        return np.nan, np.nan
    lai_min = float(np.min(lai))
    lai_max = float(np.max(lai))
    amp = lai_max - lai_min
    if amp <= 0:
        return np.nan, np.nan
    target = lai_min + p * amp
    imax = int(np.argmax(lai))

    # ── SOS: first index i in [0..imax] where lai[i] >= target ──
    sos = np.nan
    rising = lai[: imax + 1]
    rising_doy = doy[: imax + 1]
    above = rising >= target
    if above.any():
        i = int(np.argmax(above))
        if i == 0:
            sos = float(rising_doy[0])
        else:
            y0, y1 = rising[i - 1], rising[i]
            x0, x1 = rising_doy[i - 1], rising_doy[i]
            if y1 != y0:
                frac = (target - y0) / (y1 - y0)
                sos = float(x0 + frac * (x1 - x0))
            else:
                sos = float(x1)

    # ── EOS: last index i in [imax..end] where lai[i] >= target ──
    # The crossing happens between i (above) and i+1 (below), if any.
    eos = np.nan
    falling = lai[imax:]
    falling_doy = doy[imax:]
    above = falling >= target
    if above.any():
        # last True position
        i = int(len(above) - 1 - np.argmax(above[::-1]))
        if i == len(falling) - 1:
            # never falls below threshold within the curve → end of curve
            eos = float(falling_doy[i])
        else:
            y0, y1 = falling[i], falling[i + 1]
            x0, x1 = falling_doy[i], falling_doy[i + 1]
            if y1 != y0:
                frac = (target - y0) / (y1 - y0)
                eos = float(x0 + frac * (x1 - x0))
            else:
                eos = float(x0)
    return sos, eos


def _metrics_per_curve(
    g: pd.DataFrame, value_col: str,
    min_valid: int, min_amplitude: float,
) -> Dict[str, float]:
    """
    Compute the per-curve metrics (LAI_tot, SOS/EOS/LOS at 10 % and 25 %) for
    one (site, year) curve.
    Returns NaN for any metric that can't be computed under the constraints.
    """
    g = g.sort_values("doy")
    doy = g["doy"].to_numpy()
    lai = g[value_col].to_numpy(dtype=float)
    valid = np.isfinite(lai)
    n_valid = int(valid.sum())

    res = {
        "lai_tot": np.nan,
        "sos_10":  np.nan,
        "eos_10":  np.nan,
        "sos_25":  np.nan,
        "eos_25":  np.nan,
        "los_10":  np.nan,
        "los_25":  np.nan,
        "n_valid": n_valid,
        "amp":     np.nan,
    }
    if n_valid < min_valid:
        return res

    # LAI_tot — strict: only when ALL dekads of the curve are valid, otherwise
    # the sum has missing terms and is not comparable across (site, year).
    if n_valid == lai.size:
        res["lai_tot"] = float(np.sum(lai))

    doy_v = doy[valid]
    lai_v = lai[valid]
    amp = float(np.max(lai_v) - np.min(lai_v))
    res["amp"] = amp
    if amp < min_amplitude:
        return res

    sos10, eos10 = _sos_eos_for_curve(doy_v, lai_v, 0.10)
    sos25, eos25 = _sos_eos_for_curve(doy_v, lai_v, 0.25)
    res["sos_10"] = sos10
    res["eos_10"] = eos10
    res["sos_25"] = sos25
    res["eos_25"] = eos25
    # Length of season — EOS − SOS (NaN-propagating: NaN if either end missing).
    res["los_10"] = eos10 - sos10
    res["los_25"] = eos25 - sos25
    return res


def compute_metrics(
    df: pd.DataFrame, value_col: str,
    min_valid: int, min_amplitude: float,
) -> pd.DataFrame:
    """One row per (site, year) with the 5 metrics + diagnostics."""
    records = []
    for (site, year), g in df.groupby(["site_id", "year"], sort=False):
        m = _metrics_per_curve(g, value_col, min_valid, min_amplitude)
        m["site_id"] = site
        m["year"] = int(year)
        records.append(m)
    return pd.DataFrame.from_records(records)


def compute_climatology_metrics(
    df: pd.DataFrame, value_col: str,
    min_valid: int, min_amplitude: float,
) -> pd.DataFrame:
    """
    One row per site with the 5 metrics computed on the *climatology* curve:
    the across-year mean of `value_col` at each dekad (doy). `df` must already
    be restricted to the climatology years and the selected sites. Because the
    inter-annual variability is averaged out, the resulting median over sites is
    a single value — drawn as a horizontal reference line on every plot.
    """
    clim = df.groupby(["site_id", "doy"], as_index=False)[value_col].mean()
    records = []
    for site, g in clim.groupby("site_id", sort=False):
        m = _metrics_per_curve(g, value_col, min_valid, min_amplitude)
        m["site_id"] = site
        records.append(m)
    return pd.DataFrame.from_records(records)


def aggregate_yearly(
    df_metrics: pd.DataFrame, metric_col: str,
) -> pd.DataFrame:
    """
    Per-year aggregation across sites: count, median, Q1, Q3.
    """
    df = df_metrics.dropna(subset=[metric_col])
    if df.empty:
        return pd.DataFrame(columns=["year", "n", "median", "q1", "q3"])
    g = df.groupby("year")[metric_col]
    out = pd.DataFrame({
        "year":   g.median().index,
        "n":      g.size().values,
        "median": g.median().values,
        "q1":     g.quantile(0.25).values,
        "q3":     g.quantile(0.75).values,
    })
    return out.sort_values("year").reset_index(drop=True)


# ── Plotting ─────────────────────────────────────────────────────────────────


METRICS = [
    ("lai_tot", "Annual sum of dekadal LAI", "LAI_tot  (Σ over 36 dekads)"),
    ("sos_10",  "Start of season — 10 % of amplitude (rising)",  "DOY"),
    ("sos_25",  "Start of season — 25 % of amplitude (rising)",  "DOY"),
    ("eos_10",  "End of season — 10 % of amplitude (falling)",   "DOY"),
    ("eos_25",  "End of season — 25 % of amplitude (falling)",   "DOY"),
    ("los_10",  "Length of season — 10 % of amplitude (EOS − SOS)", "Days"),
    ("los_25",  "Length of season — 25 % of amplitude (EOS − SOS)", "Days"),
]


def plot_one_metric(
    obs_yearly: pd.DataFrame,
    pred_yearly: Optional[pd.DataFrame],
    title: str,
    ylabel: str,
    filename: str,
    n_sites: int,
    clim_value: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    if clim_value is not None and np.isfinite(clim_value):
        ax.axhline(
            clim_value, color="#264653", lw=1.8, ls=":",
            label=f"Climatology (obs) — median = {clim_value:.3g}",
        )

    if not obs_yearly.empty:
        ax.fill_between(
            obs_yearly["year"], obs_yearly["q1"], obs_yearly["q3"],
            color="#2a9d8f", alpha=0.25,
        )
        ax.plot(
            obs_yearly["year"], obs_yearly["median"],
            "o-", color="#1b6e63", lw=2.0, markersize=5,
            label="Observation — median, IQR shaded",
        )

    if pred_yearly is not None and not pred_yearly.empty:
        ax.fill_between(
            pred_yearly["year"], pred_yearly["q1"], pred_yearly["q3"],
            color="#e76f51", alpha=0.25,
        )
        ax.plot(
            pred_yearly["year"], pred_yearly["median"],
            "s--", color="#c14b30", lw=2.0, markersize=5,
            label="Prediction — median, IQR shaded",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(up to {n_sites:,} sites in the pool)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)
    print(f"  → {filename}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    if not args.predictions and not args.target_dir:
        raise ValueError("Provide either --predictions or --target_dir.")
    if args.predictions and args.target_dir:
        print("[info] --predictions takes precedence; --target_dir is ignored.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──
    if args.predictions:
        df_full = _load_from_predictions(args.predictions)
        has_pred = True
        print(f"Loaded predictions CSV : {len(df_full):,} rows "
              f"from {args.predictions}")
    else:
        df_full = _load_from_target_dir(args.target_dir)
        df_full["lai_pred"] = np.nan
        has_pred = False
        print(f"Loaded target dir      : {len(df_full):,} rows "
              f"from {args.target_dir}")

    # Keep the un-year-filtered frame so the climatology can be built from its
    # own (possibly different) set of years, independent of --years.
    df_all_years = df_full

    # ── Year filter ──
    year_filter = _parse_year_spec(args.years)
    if year_filter is not None:
        df_full = df_full[df_full["year"].isin(year_filter)].copy()
    years_in_scope = sorted(df_full["year"].unique().tolist())
    print(f"Years in scope         : {years_in_scope}")

    # ── Site selection ──
    df_full = _select_sites(df_full, args)
    n_sites_total = df_full["site_id"].nunique()

    # ── Observation climatology (one horizontal reference value per metric) ──
    clim_values = None
    if args.clim_years:
        clim_year_set = _parse_year_spec(args.clim_years)
        selected_sites = set(df_full["site_id"].unique())
        clim_src = df_all_years[
            df_all_years["site_id"].isin(selected_sites)
            & df_all_years["year"].isin(clim_year_set)
        ]
        if clim_src.empty:
            raise RuntimeError(
                f"No observations in --clim_years {sorted(clim_year_set)} "
                f"for the selected sites."
            )
        clim_metrics = compute_climatology_metrics(
            clim_src, "lai_obs", args.min_valid, args.min_amplitude,
        )
        clim_metrics.to_csv(
            os.path.join(args.output_dir, "metrics_climatology.csv"),
            index=False,
        )
        clim_values = {}
        for col, _, _ in METRICS:
            v = clim_metrics[col].dropna()
            clim_values[col] = float(v.median()) if not v.empty else np.nan
        print(
            f"Climatology years      : {sorted(clim_year_set)}  "
            f"({len(clim_metrics):,} site curves)"
        )

    # ── Compute metrics ──
    print("\nComputing OBS metrics …")
    obs_metrics = compute_metrics(
        df_full, "lai_obs", args.min_valid, args.min_amplitude,
    )
    print(
        f"  {len(obs_metrics):,} (site, year) curves   "
        f"LAI_tot ok: {obs_metrics['lai_tot'].notna().sum():,}   "
        f"SOS/EOS ok: {obs_metrics['sos_10'].notna().sum():,}"
    )

    pred_metrics = None
    if has_pred:
        print("Computing PRED metrics …")
        pred_metrics = compute_metrics(
            df_full, "lai_pred", args.min_valid, args.min_amplitude,
        )
        print(
            f"  {len(pred_metrics):,} (site, year) curves   "
            f"LAI_tot ok: {pred_metrics['lai_tot'].notna().sum():,}   "
            f"SOS/EOS ok: {pred_metrics['sos_10'].notna().sum():,}"
        )

    # ── Save raw tables for offline use ──
    obs_metrics.to_csv(
        os.path.join(args.output_dir, "metrics_obs.csv"), index=False,
    )
    if pred_metrics is not None:
        pred_metrics.to_csv(
            os.path.join(args.output_dir, "metrics_pred.csv"), index=False,
        )

    # ── Plots ──
    print("\nPlotting trend figures …")
    for col, title, ylabel in METRICS:
        obs_y = aggregate_yearly(obs_metrics, col)
        pred_y = (aggregate_yearly(pred_metrics, col)
                  if pred_metrics is not None else None)
        plot_one_metric(
            obs_y, pred_y,
            title=title, ylabel=ylabel,
            filename=os.path.join(args.output_dir, f"{col}_trend.png"),
            n_sites=n_sites_total,
            clim_value=(clim_values.get(col) if clim_values else None),
        )

    # ── Console summary ──
    print("\nMedian over years (median across sites, then median across years):")
    for col, title, _ in METRICS:
        obs_y = aggregate_yearly(obs_metrics, col)
        if obs_y.empty:
            print(f"  {col:8s} : no valid data")
            continue
        line = (f"  {col:8s} : obs={obs_y['median'].median():8.3f}  "
                f"(n_curves={int(obs_y['n'].sum()):,})")
        if pred_metrics is not None:
            pred_y = aggregate_yearly(pred_metrics, col)
            if not pred_y.empty:
                line += (f"   pred={pred_y['median'].median():8.3f}  "
                         f"(n_curves={int(pred_y['n'].sum()):,})")
        if clim_values is not None and np.isfinite(clim_values.get(col, np.nan)):
            line += f"   clim={clim_values[col]:8.3f}"
        print(line)


if __name__ == "__main__":
    main()
