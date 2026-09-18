#!/usr/bin/env python3
"""
Leaf-out date histograms from THEIA target files.

For each (site, year) curve, find the day-of-year (DOY) at which LAI first
crosses a fraction of its annual maximum (during the rising phase only).
Plot one histogram per threshold over a random sample of N sites.

Usage
-----
    python -m util.phenology_dates \
        --target_dir LAI/data/target_years \
        --n_sites 200 \
        --output phenology_hist.png

    # custom thresholds and year range
    python -m util.phenology_dates \
        --target_dir LAI/data/target_years \
        --n_sites 500 \
        --thresholds 10,25,50,75 \
        --years 2001-2003 \
        --min_max_lai 1.0
"""

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot leaf-out date histograms")
    p.add_argument("--target_dir", required=True,
                   help="Directory with target_{year}.csv files")
    p.add_argument("--target_files", nargs="*", default=None,
                   help="Explicit list of target CSVs (overrides --target_dir)")
    p.add_argument("--n_sites", type=int, default=200,
                   help="Number of sites to sample randomly")
    p.add_argument("--years", default=None,
                   help="Year filter, e.g. '2001-2003' or '2000,2005'")
    p.add_argument("--thresholds", default="10,25,50,75",
                   help="Comma-separated thresholds in %% of annual max")
    p.add_argument("--min_max_lai", type=float, default=0.5,
                   help="Skip curves whose annual max LAI is below this")
    p.add_argument("--min_valid", type=int, default=12,
                   help="Skip curves with fewer than this many valid dekads")
    p.add_argument("--output", default="phenology_dates.png",
                   help="Output figure path")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for site sampling")
    return p.parse_args()


def _parse_years(s: str) -> set:
    s = s.strip()
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return set(range(int(lo), int(hi) + 1))
    return {int(y) for y in s.split(",")}


def _list_target_files(args) -> List[str]:
    """
    Resolve the input files. Two sources are supported:
      - PhenoNN NetCDF cubes : LAI_dekadal_{YYYY}.nc  (new pipeline)
      - LaiNN legacy CSVs    : target_{YYYY}.csv      (old pipeline)

    NetCDF takes precedence when both are present.
    """
    if args.target_files:
        files = list(args.target_files)
    else:
        nc  = sorted(glob.glob(os.path.join(args.target_dir,
                                            "LAI_dekadal_*.nc")))
        csv = sorted(glob.glob(os.path.join(args.target_dir,
                                            "target_*.csv")))
        files = nc if nc else csv
    if not files:
        raise FileNotFoundError(
            f"No LAI_dekadal_*.nc or target_*.csv files in {args.target_dir}"
        )
    if args.years:
        keep = _parse_years(args.years)

        def _year_of(fp: str) -> int:
            name = os.path.basename(fp)
            stem = name.split(".")[0]
            # LAI_dekadal_2010 → "2010"; target_2010 → "2010"
            return int(stem.split("_")[-1])

        files = [f for f in files if _year_of(f) in keep]
    return files


def _crossing_doy(doy: np.ndarray, lai: np.ndarray, thresh_frac: float):
    """
    First DOY at which LAI reaches thresh_frac * max, considering only the
    rising phase (DOY <= argmax). Returns np.nan if no such crossing.

    Linear interpolation between the last sub-threshold sample and the first
    above-threshold sample to get sub-dekad resolution.
    """
    if lai.size == 0 or not np.isfinite(lai).any():
        return np.nan
    imax = int(np.nanargmax(lai))
    if imax == 0:
        return np.nan
    target = thresh_frac * lai[imax]
    rising_doy = doy[: imax + 1]
    rising_lai = lai[: imax + 1]
    above = rising_lai >= target
    if not above.any():
        return np.nan
    i = int(np.argmax(above))  # first True
    if i == 0:
        return float(rising_doy[0])
    y0, y1 = rising_lai[i - 1], rising_lai[i]
    x0, x1 = rising_doy[i - 1], rising_doy[i]
    if y1 == y0:
        return float(x1)
    frac = (target - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def compute_phenology_dates(df: pd.DataFrame, thresholds: List[float],
                            min_max_lai: float, min_valid: int) -> pd.DataFrame:
    """
    Return one row per (site_id, year) with crossing DOYs for each threshold.
    """
    # date is int YYYYMMDD → DOY
    dt = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.assign(doy=dt.dt.dayofyear)
    df = df.sort_values(["site_id", "year", "doy"])

    records = []
    for (site, year), g in df.groupby(["site_id", "year"], sort=False):
        valid = g["LAI"].notna()
        if valid.sum() < min_valid:
            continue
        doy = g["doy"].to_numpy()
        lai = g["LAI"].to_numpy(dtype=float)
        max_lai = np.nanmax(lai)
        if not np.isfinite(max_lai) or max_lai < min_max_lai:
            continue
        rec = {"site_id": site, "year": int(year), "max_lai": float(max_lai)}
        for t in thresholds:
            rec[f"doy_{int(round(t*100))}"] = _crossing_doy(doy, lai, t)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def plot_histograms(dates: pd.DataFrame, thresholds: List[float],
                    output: str):
    n = len(thresholds)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    # shared x-range from all thresholds for visual comparison
    cols = [f"doy_{int(round(t*100))}" for t in thresholds]
    all_vals = pd.concat([dates[c].dropna() for c in cols])
    if all_vals.empty:
        raise RuntimeError("No valid leaf-out dates computed — check inputs.")
    xmin = max(0, int(np.floor(all_vals.min() / 10) * 10))
    xmax = min(366, int(np.ceil(all_vals.max() / 10) * 10))
    bins = np.arange(xmin, xmax + 5, 5)

    for ax, t, col in zip(axes, thresholds, cols):
        vals = dates[col].dropna().to_numpy()
        ax.hist(vals, bins=bins, color="#3a86ff", edgecolor="white",
                alpha=0.85)
        mean = float(np.mean(vals)) if vals.size else float("nan")
        median = float(np.median(vals)) if vals.size else float("nan")
        ax.axvline(median, color="#d62828", lw=1.8, label=f"median = {median:.0f}")
        ax.axvline(mean,   color="#073b4c", lw=1.4, ls="--",
                   label=f"mean = {mean:.1f}")
        ax.set_title(f"DOY at {int(round(t*100))}% of annual max  "
                     f"(n={vals.size})")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Count of (site, year) curves")
        ax.set_xlim(xmin, xmax)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)

    # hide unused panels
    for ax in axes[len(thresholds):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Leaf-out dates over {dates['site_id'].nunique()} sites "
        f"× {dates['year'].nunique()} years "
        f"({len(dates)} curves)",
        fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=150)
    print(f"→ {output}")


def _load_year_long(path: str) -> pd.DataFrame:
    """
    Return a (site_id, date, year, LAI) long DataFrame for one input file,
    auto-detecting `.nc` (PhenoNN) or `.csv` (LaiNN) format.

    For NetCDF, the spatial grid is converted to `pix_LLLL_OOOOO` site IDs
    so the rest of the script processes both sources identically.
    """
    import datetime
    if path.endswith(".nc"):
        import xarray as xr
        year = int(os.path.basename(path).split(".")[0].split("_")[-1])
        ds = xr.open_dataset(path)
        # Pixelset LAI(dekad, site): site_id straight from the coord.
        lai = ds["LAI"].transpose("site", "dekad").values.astype(np.float32)
        site_ids = np.asarray(ds["site_id"].values).astype(str)  # (n_site,)
        ds.close()
        # 36 (month, day, doy) tuples for a non-leap year
        obs = [(m, d) for m in range(1, 13) for d in (5, 15, 25)]
        dates = [int(f"{year:04d}{m:02d}{d:02d}") for (m, d) in obs]
        frames = []
        for k, date_int in enumerate(dates):
            frames.append(pd.DataFrame({
                "site_id": site_ids,
                "date":    date_int,
                "year":    year,
                "LAI":     lai[:, k],
            }))
        return pd.concat(frames, ignore_index=True)
    # Legacy CSV
    return pd.read_csv(path, usecols=["site_id", "date", "year", "LAI"])


def main():
    args = parse_args()
    files = _list_target_files(args)
    print(f"Reading {len(files)} target file(s): "
          f"{os.path.basename(files[0])} … {os.path.basename(files[-1])}")

    thresholds = [float(t) / 100.0 for t in args.thresholds.split(",")]
    print(f"Thresholds (% of annual max): {[int(t*100) for t in thresholds]}")

    # pick sites once from the first file (all years share the site grid)
    first = _load_year_long(files[0])
    all_sites = first["site_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(args.seed)
    n_pick = min(args.n_sites, len(all_sites))
    sample = set(rng.choice(all_sites, size=n_pick, replace=False).tolist())
    print(f"Sampled {n_pick} sites out of {len(all_sites)}")

    # stream per-year, keep only sampled sites
    frames = []
    for f in files:
        df = _load_year_long(f)
        df = df[df["site_id"].isin(sample)]
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No rows survived site sampling — check --n_sites.")
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows across {df['year'].nunique()} year(s)")

    dates = compute_phenology_dates(
        df, thresholds,
        min_max_lai=args.min_max_lai,
        min_valid=args.min_valid,
    )
    print(f"Computed leaf-out dates for {len(dates)} (site, year) curves "
          f"({dates['site_id'].nunique()} unique sites)")

    if dates.empty:
        raise RuntimeError(
            "No valid curves passed the filters "
            f"(min_max_lai={args.min_max_lai}, min_valid={args.min_valid})."
        )

    # summary table
    print("\nSummary by threshold:")
    for t in thresholds:
        col = f"doy_{int(round(t*100))}"
        v = dates[col].dropna()
        print(f"  {int(t*100):>3d}%:  mean={v.mean():6.1f}  "
              f"median={v.median():5.0f}  std={v.std():5.1f}  n={len(v)}")

    plot_histograms(dates, thresholds, args.output)


if __name__ == "__main__":
    main()
