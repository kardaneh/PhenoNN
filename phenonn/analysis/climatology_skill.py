#!/usr/bin/env python3
"""
climatology_skill.py — skill of the CLIMATOLOGY baseline on the dataset itself.

No model, **no torch**: runs in the h5-only preprocessing venv (numpy / pandas /
xarray only). Builds the per-(site, dekad) climatology from `--clim_years` and
scores it against the observations of `--eval_years`:

    clim(site, dekad) = mean over clim_years of LAI_obs(site, year, dekad)
    prediction for ANY eval year = clim(site, dekad)

This is THE reference every model must beat: the climatology knows the average
seasonal cycle of each site but nothing about a particular year. Its R² tells you
how much of the signal is pure seasonality; its residual is the interannual
variability a model has to capture to add value.

Metrics (same definitions as phenonn.prediction.predict):
  • Overall R² (NSE over all points), RMSE, MAE, bias
  • Site-year R²   : NSE within each site-year (its 36 dekads) → mean ± std
  • Interannual R² : NSE across site-years at a FIXED day-of-year → mean ± std
  • Per-site R² distribution (mean ± std, median, percentiles, % > 0)

Run it as a PLAIN SCRIPT (not `python -m phenonn…`): importing the package would
execute phenonn/__init__.py, which imports torch.

    python phenonn/analysis/climatology_skill.py \\
        --target_dir     $DATA/LAI_pixelset \\
        --selected_pixels $DATA/selected_pixels005_balanced.nc \\
        --clim_years 1992-2009 --eval_years 2010-2019 \\
        --n_sites 20000 --output_dir runs_clim_skill
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import xarray as xr

# Inlined from phenonn/utils/config.py ON PURPOSE: importing anything under
# `phenonn.` executes phenonn/__init__.py, which imports torch. Keeping these two
# constants local is what makes this script runnable in the h5-only venv.
N_DEKAD_YEAR = 36
TARGETS_FNAME = "LAI_dekadal_{year}.nc"

_OBS_DOY = [datetime.date(2001, m, d).timetuple().tm_yday
            for m in range(1, 13) for d in (5, 15, 25)]


# ── Local pixelset readers (torch-free copies of the lai_dataset helpers) ─────


def load_selected_pixels(path):
    """Site IDs stored in a selected_pixels*.nc."""
    with xr.open_dataset(path, engine="netcdf4") as ds:
        return [str(s) for s in ds["site_id"].values]


def _site_row_index(da, site_ids):
    """Row of each requested site in `da` (-1 when absent). Built once, reused:
    the pixelset site_id coord is identical across years."""
    file_ids = pd.Index(np.asarray(da["site_id"].values).astype(str))
    return file_ids.get_indexer(pd.Index([str(s) for s in site_ids]))


def _read_site_vectors(da, site_ids, value_dim, rows):
    """(n_site, size(value_dim)) float32, NaN rows for unknown sites.

    The LAI pixelset uses very large site-chunks, so a scattered per-site isel
    would decompress them repeatedly — read the whole variable ONCE, then gather.
    """
    out = np.full((len(site_ids), da.sizes[value_dim]), np.nan, dtype=np.float32)
    ok = rows >= 0
    if ok.any():
        full = da.transpose("site", value_dim).values
        out[ok] = full[rows[ok]].astype(np.float32)
    return out


def _climatology(target_dir, clim_years, sites, rows_holder):
    """(n_sites, 36) per-(site, dekad) mean over clim_years; NaN where no obs."""
    sum_ = np.zeros((len(sites), N_DEKAD_YEAR), dtype=np.float64)
    cnt = np.zeros((len(sites), N_DEKAD_YEAR), dtype=np.int32)
    for y in sorted(set(clim_years)):
        path = os.path.join(target_dir, TARGETS_FNAME.format(year=y))
        if not os.path.exists(path):
            print(f"  [clim] {os.path.basename(path)} missing — skipped")
            continue
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if rows_holder[0] is None:
                rows_holder[0] = _site_row_index(ds["LAI"], sites)
            vals = _read_site_vectors(ds["LAI"], sites, "dekad", rows_holder[0])
        finite = np.isfinite(vals)
        sum_ += np.where(finite, vals, 0.0)
        cnt += finite.astype(np.int32)
        print(f"  [clim] {y} ok")
    return np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan).astype(np.float32)


# ── Metrics ──────────────────────────────────────────────────────────────────


def _nse(o, p, nmin=5):
    """Nash–Sutcliffe R² of p against o, or None when ill-defined."""
    o = np.asarray(o, float); p = np.asarray(p, float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    sstot = float(np.sum((o - o.mean()) ** 2))
    if o.size < nmin or sstot <= 0:
        return None
    return 1.0 - float(np.sum((p - o) ** 2)) / sstot


def _parse_years(spec):
    if not spec:
        return []
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_dir", required=True,
                   help="Folder of LAI_dekadal_{Y}.nc.")
    p.add_argument("--selected_pixels", required=True,
                   help="selected_pixels*.nc giving the site pool.")
    p.add_argument("--clim_years", required=True,
                   help="Years building the climatology, e.g. 1992-2009.")
    p.add_argument("--eval_years", required=True,
                   help="Years it is scored on, e.g. 2010-2019.")
    p.add_argument("--n_sites", type=int, default=20000,
                   help="Random subsample of sites (0 = all). Default 20000.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="runs_clim_skill")
    args = p.parse_args()

    clim_years = _parse_years(args.clim_years)
    eval_years = _parse_years(args.eval_years)
    overlap = sorted(set(clim_years) & set(eval_years))
    if overlap:
        print(f"[warn] clim_years and eval_years overlap on {overlap} → the "
              f"baseline partly 'sees' the answer (optimistic).")

    sites = list(dict.fromkeys(load_selected_pixels(args.selected_pixels)))
    if 0 < args.n_sites < len(sites):
        rng = np.random.RandomState(args.seed)
        sites = sorted(rng.choice(sites, size=args.n_sites, replace=False).tolist())
    print(f"Sites        : {len(sites):,}")
    print(f"Climatology  : {clim_years}")
    print(f"Evaluated on : {eval_years}")

    rows_holder = [None]                       # site→row index, built once
    print("Building climatology …")
    clim = _climatology(args.target_dir, clim_years, sites, rows_holder)
    print(f"  {int(np.isfinite(clim).any(axis=1).sum()):,} sites with a climatology")

    records = []
    for y in eval_years:
        path = os.path.join(args.target_dir, TARGETS_FNAME.format(year=y))
        if not os.path.exists(path):
            print(f"  ✗ {y} missing — skipped")
            continue
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if rows_holder[0] is None:
                rows_holder[0] = _site_row_index(ds["LAI"], sites)
            vals = _read_site_vectors(ds["LAI"], sites, "dekad", rows_holder[0])
        # Vectorised: keep only (site, dekad) pairs finite in BOTH obs and clim.
        ok = np.isfinite(vals) & np.isfinite(clim)
        si, di = np.nonzero(ok)
        if si.size:
            records.append(pd.DataFrame({
                "site_id": np.asarray(sites, dtype=object)[si],
                "year": np.int32(y),
                "doy": np.asarray(_OBS_DOY, dtype=np.int32)[di],
                "lai_obs": vals[si, di],
                "lai_clim": clim[si, di],
            }))
        print(f"  ✓ {y}  ({len(np.unique(si)):,} sites, {si.size:,} points)")

    if not records:
        raise SystemExit("No valid (obs, climatology) pairs.")
    df = pd.concat(records, ignore_index=True)
    df["error"] = df["lai_clim"] - df["lai_obs"]

    os.makedirs(args.output_dir, exist_ok=True)
    csv = os.path.join(args.output_dir, "clim_predictions.csv")
    df.to_csv(csv, index=False)

    _summary(df, args.output_dir, clim_years, eval_years)
    _plot(df, args.output_dir)
    print(f"\nCSV → {csv}")


def _summary(df, out_dir, clim_years, eval_years):
    o = df["lai_obs"].values; c = df["lai_clim"].values
    err = c - o
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    r2_overall = _nse(o, c)
    r2_overall = float("nan") if r2_overall is None else r2_overall

    sy = [r for _, g in df.groupby(["site_id", "year"], sort=False)
          if (r := _nse(g["lai_obs"].values, g["lai_clim"].values)) is not None]
    ia = [r for _, g in df.groupby("doy", sort=False)
          if (r := _nse(g["lai_obs"].values, g["lai_clim"].values)) is not None]
    ps = [r for _, g in df.groupby("site_id", sort=False)
          if (r := _nse(g["lai_obs"].values, g["lai_clim"].values)) is not None]
    sy, ia, ps = np.array(sy), np.array(ia), np.array(ps)

    L = ["── Climatology baseline ──",
         f"  Climatology from : {clim_years}",
         f"  Evaluated on     : {eval_years}",
         f"  Points           : {len(df):,}  "
         f"({df['site_id'].nunique():,} sites × {df['year'].nunique()} years)",
         "",
         "── Pooled ──",
         f"  RMSE             : {rmse:.4f}",
         f"  MAE              : {mae:.4f}",
         f"  Bias (clim−obs)  : {bias:+.4f}",
         "",
         "── R² decomposition ──",
         f"  Overall      R² : {r2_overall:+.4f}",
         (f"  Site-year    R² : {sy.mean():+.4f} ± {sy.std():.4f}  (n={sy.size})"
          if sy.size else "  Site-year    R² : n/a"),
         (f"  Interannual  R² : {ia.mean():+.4f} ± {ia.std():.4f}  (n={ia.size})"
          if ia.size else "  Interannual  R² : n/a"),
         ""]
    if ps.size:
        L += [f"── Per-site R² ({ps.size:,} sites) ──",
              f"  Mean ± std      : {ps.mean():+.4f} ± {ps.std():.4f}",
              f"  Median          : {np.median(ps):+.4f}",
              f"  5th / 95th pct  : {np.percentile(ps, 5):+.4f} / "
              f"{np.percentile(ps, 95):+.4f}",
              f"  Sites with R²>0 : {int((ps > 0).sum()):,} / {ps.size:,} "
              f"({100.0 * (ps > 0).mean():.1f}%)"]
    L += ["",
          "Interpretation: 'Interannual R²' is the key number — the climatology",
          "cannot distinguish years, so it is ~0 by construction. A model only",
          "adds value if it beats these figures, especially interannually.",
          f"Model skill score: SS = 1 − MSE_model / MSE_clim  "
          f"(MSE_clim = {rmse ** 2:.5f})."]

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"Summary → {path}")


def _plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sub = df.sample(min(len(df), 50000), random_state=0)
    ax[0].scatter(sub["lai_obs"], sub["lai_clim"], s=2, alpha=0.15)
    lo = float(min(sub["lai_obs"].min(), sub["lai_clim"].min()))
    hi = float(max(sub["lai_obs"].max(), sub["lai_clim"].max()))
    ax[0].plot([lo, hi], [lo, hi], "r-", lw=1)
    ax[0].set_xlabel("observed LAI"); ax[0].set_ylabel("climatology LAI")
    ax[0].set_title("Climatology vs observations")
    for col, lab in [("lai_obs", "obs"), ("lai_clim", "climatology")]:
        ax[1].plot(df.groupby("doy")[col].mean().sort_index(),
                   marker="o", ms=3, label=lab)
    ax[1].set_xlabel("day of year"); ax[1].set_ylabel("mean LAI")
    ax[1].set_title("Mean seasonal cycle"); ax[1].legend()
    fig.tight_layout()
    png = os.path.join(out_dir, "clim_skill.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"Plot → {png}")


if __name__ == "__main__":
    main()
