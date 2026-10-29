# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Per-PFT diagnostics from a single-PFT pixel subset.

Input is a ``selected_pixels_PFT{k}.nc`` subset (the cells dominated by one
PFT). Reads the LAI and ERA5 pixelset datasets and produces four figures. LAI is
keyed by the native 0.05 degree ``pix_...`` site IDs. ERA5 may be either the
0.05 degree identity route or the deduplicated 0.1 degree route; for the
latter, pass ``--parent_map`` so the subset IDs are translated to ``E...``
cell IDs before the feature read.

1. ``lai_seasonal.png``
   Mean dekadal LAI curve across the selected cells, with a ±1 std band and
   the full min/max envelope. LAI is DEKADAL (36 obs/year on days 5/15/25), so
   the x-axis has 36 points placed at their day-of-year; this is the native
   resolution of the data, not a daily series. All selected (site, year)
   curves are pooled per dekad.

2. ``features_hist.png``
   One histogram per meteo feature, pooling every daily value of every
   selected cell over every year. The x-range is clipped to the 0.5–99.5
   percentiles so heavy-tailed variables (precip, radiation) stay readable.

3. ``features_dist_by_year.png``
   Same features, but one density curve per year (shared bins, coloured by
   year with a colorbar) so inter-annual shifts in the feature distribution
   are visible.

4. ``features_seasonal.png``
   One subplot per feature: mean daily value across the selected cells, with
   a ±1 std band and the full min/max envelope, pooling every selected cell
   over every year. Leap (366) and non-leap (365) years are aligned on a fixed
   366-day day-of-year grid (non-leap years skip the Feb-29 slot), so ALL
   years contribute — none are dropped.

Input
-----
``--selected_pixels``
    ``selected_pixels_PFT{k}.nc`` — the PFT-dominated subset.
``--era5_dir``
    Folder of ``ERA5_daily_pixelset_{Y}.nc``.
``--target_dir``
    Folder of ``LAI_dekadal_{Y}.nc``.
``--year_start`` / ``--year_end``
    Inclusive year range.
``--parent_map``
    Optional ``selected_pixels01_1.nc`` mapping each 0.05 degree ``pix_...``
    site to its 0.1 degree ERA5 ``E...`` parent cell. Only the ERA5 side is
    translated; LAI remains keyed by the subset IDs.

Notes
-----
Torch-free and scipy-free; read-only (no NetCDF write, so no subprocess
needed).

Examples
--------
::

    python -m phenonn.datagen.pft_diagnostics \\
        --selected_pixels /path/to/data/pixelset1/selected_pixels_10%_PFT/selected_pixels_10%_PFT_0.95/selected_pixels_10%_PFT1 \\
        --era5_dir        /path/to/data/pixelset1/era5_pixelset \\
        --target_dir      /path/to/data/pixelset1/LAI_pixelset \\
        --year_start 1992 --year_end 2018 \\
        --output_dir /path/to/data/pixelset1/diag_plots/PFT1/
"""

import argparse
import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phenonn.datagen.config import (
    DYNAMIC_FEATURES,
    FEATURES_FNAME,
    PFT_NAMES,
    TARGETS_FNAME,
)


HIST_BINS = 60
CLIP_PCT = (0.5, 99.5)  # percentile x-range for readability of heavy tails

# Seasonal stats are pooled on a fixed 366-day day-of-year grid so leap (366)
# and non-leap (365) years align by calendar date. Feb 29 is grid slot 59
# (0-based); non-leap years skip it, so March onward stays aligned.
SEASON_LEN = 366
LEAP_DAY_IDX = 59


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``selected_pixels``, ``era5_dir``,
        ``target_dir``, ``year_start``, ``year_end``, ``features``,
        ``parent_map``, ``output_dir``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels_PFT{k}.nc — the PFT-dominated subset.",
    )
    p.add_argument(
        "--era5_dir", required=True, help="Folder of ERA5_daily_pixelset_{Y}.nc."
    )
    p.add_argument("--target_dir", required=True, help="Folder of LAI_dekadal_{Y}.nc.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument(
        "--features",
        default="",
        help="Comma-separated feature names to plot "
        "(default: config.DYNAMIC_FEATURES). Names absent from a "
        "file are skipped with a warning.",
    )
    p.add_argument(
        "--parent_map",
        default="",
        help="selected_pixels01_1.nc mapping the subset's 0.05 degree "
        "(pix_...) ids to the ERA5 cubes' 0.1 degree (E...) ids. Only the "
        "ERA5 side is translated; LAI stays keyed by the subset ids.",
    )
    p.add_argument("--output_dir", default="pft_diagnostics")
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _dekad_doys() -> np.ndarray:
    """
    Day-of-year of the 36 dekads (days 5/15/25 of each month, non-leap).

    Returns
    -------
    doys : ndarray of shape (36,)
        Day-of-year of each dekad anchor, dtype int16.
    """
    return np.array(
        [
            datetime.date(2001, m, d).timetuple().tm_yday
            for m in range(1, 13)
            for d in (5, 15, 25)
        ],
        dtype=np.int16,
    )


def _load_selected(path: str) -> Tuple[np.ndarray, Optional[int]]:
    """
    Read the site IDs and the ORCHIDEE PFT number from a selection file.

    Parameters
    ----------
    path : str
        Path to ``selected_pixels_PFT{k}.nc``.

    Returns
    -------
    site_ids : ndarray of shape (n_site,)
        Site identifiers (object array of str).
    pft : int or None
        Value of the ``filter_pft_orchidee`` attribute, or ``None`` if absent.
    """
    with xr.open_dataset(path) as sel:
        site_ids = np.asarray(sel["site_id"].values).astype(str)
        pft = sel.attrs.get("filter_pft_orchidee")
    pft = int(pft) if pft is not None else None
    return site_ids, pft


def _load_parent01_map(selected_pixels_01: str) -> Dict[str, str]:
    """
    Return ``{site_id05: site_id01}`` from a ``selected_pixels01_1.nc``.

    Kept local so this diagnostic remains torch-free.

    Parameters
    ----------
    selected_pixels_01 : str
        Parent-map file produced by ``make_selected_pixels_01.py``.

    Returns
    -------
    dict
        Mapping from 0.05 degree site IDs to 0.1 degree ERA5 cell IDs.
    """
    with xr.open_dataset(selected_pixels_01, engine="netcdf4") as ds:
        site_id05 = np.asarray(ds["site_id05"].values).astype(str)
        parent_site = np.asarray(ds["parent_site"].values).astype(np.int64)
        site_id01 = np.asarray(ds["site_id"].values).astype(str)
    return {s05: site_id01[p] for s05, p in zip(site_id05, parent_site)}


def _site_mask(file_site_ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """
    Boolean mask keeping the file's sites that are in ``wanted``.

    Parameters
    ----------
    file_site_ids : ndarray of shape (n_file,)
        Site IDs read from a pixelset.
    wanted : ndarray of shape (n_wanted,)
        Site IDs to keep.

    Returns
    -------
    mask : ndarray of shape (n_file,) of bool
        True for sites that are in ``wanted``.
    """
    return np.isin(file_site_ids.astype(str), wanted)


def _load_lai(target_dir: Path, years: range, wanted: np.ndarray) -> np.ndarray:
    """
    Pool dekadal LAI of the selected cells over all years.

    Parameters
    ----------
    target_dir : pathlib.Path
        Folder of ``LAI_dekadal_{Y}.nc`` pixelsets.
    years : range
        Years to read.
    wanted : ndarray of shape (n_wanted,)
        Site IDs to keep.

    Returns
    -------
    lai : ndarray of shape (36, n_curves)
        Each column is one (site, year) curve; missing dekads are NaN.

    Raises
    ------
    RuntimeError
        If no LAI data is found for the selected sites.
    """
    cols: List[np.ndarray] = []
    for year in years:
        fpath = target_dir / TARGETS_FNAME.format(year=year)
        if not fpath.exists():
            print(f"  ✗ LAI {year} skipped — missing {fpath.name}")
            continue
        with xr.open_dataset(fpath, engine="netcdf4", decode_times=False) as ds:
            file_ids = np.asarray(ds["site_id"].values)
            mask = _site_mask(file_ids, wanted)
            if not mask.any():
                continue
            lai = (
                ds["LAI"].transpose("dekad", "site").values[:, mask].astype(np.float32)
            )  # (36, n_sel)
        cols.append(lai)
        print(f"  ✓ LAI {year}: {int(mask.sum()):,} sites")
    if not cols:
        raise RuntimeError("No LAI data found for the selected sites.")
    return np.concatenate(cols, axis=1)


def _load_features(
    era5_dir: Path,
    years: range,
    wanted: np.ndarray,
    feats: List[str],
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Read each ERA5 year once and return two views of the selected cells.

    Parameters
    ----------
    era5_dir : pathlib.Path
        Folder of ``ERA5_daily_pixelset_{Y}.nc`` files.
    years : range
        Years to read.
    wanted : ndarray of shape (n_wanted,)
        Site IDs to keep.
    feats : list of str
        Features to extract.

    Returns
    -------
    data : dict
        ``{feature: {year: 1-D array of finite daily values}}`` for the
        histograms and by-year density plots.
    seasonal : dict
        ``{feature: {"mean", "std", "min", "max"}}`` — per-day-of-year stats
        pooled over all years.
    """
    data: Dict[str, Dict[int, np.ndarray]] = {f: {} for f in feats}
    acc: Dict[str, Optional[Dict[str, np.ndarray]]] = {f: None for f in feats}
    for year in years:
        fpath = era5_dir / FEATURES_FNAME.format(year=year)
        if not fpath.exists():
            print(f"  ✗ ERA5 {year} skipped — missing {fpath.name}")
            continue
        with xr.open_dataset(fpath, engine="netcdf4", decode_times=False) as ds:
            file_ids = np.asarray(ds["site_id"].values)
            mask = _site_mask(file_ids, wanted)
            if not mask.any():
                continue
            for f in feats:
                if f not in ds:
                    continue
                arr = (
                    ds[f].transpose("time", "site").values[:, mask].astype(np.float32)
                )  # (n_time, n_sel)
                flat = arr.ravel()
                data[f][year] = flat[np.isfinite(flat)]
                _accumulate_seasonal(acc, f, arr, year)
        print(f"  ✓ ERA5 {year}: {int(mask.sum()):,} sites")

    seasonal = {f: _finalize_seasonal(acc[f]) for f in feats if acc[f] is not None}
    return data, seasonal


def _doy_grid_index(n_time: int):
    """
    Map a year's time index onto the fixed 366-day DOY grid.

    Parameters
    ----------
    n_time : int
        Number of time steps in the year.

    Returns
    -------
    gidx : ndarray of shape (n_time,) or None
        Index into the 366-day grid, or ``None`` if ``n_time`` is neither 365
        nor 366.
    """
    if n_time == SEASON_LEN:
        return np.arange(SEASON_LEN)
    if n_time == SEASON_LEN - 1:
        g = np.arange(SEASON_LEN - 1)
        g[LEAP_DAY_IDX:] += 1
        return g
    return None


def _accumulate_seasonal(acc, f: str, arr: np.ndarray, year: int) -> None:
    """
    Fold one ``(n_time, n_sel)`` block into the running per-DOY accumulators.

    Parameters
    ----------
    acc : dict
        Mapping ``feature -> accumulator dict`` (created on first use).
    f : str
        Feature name.
    arr : ndarray of shape (n_time, n_sel)
        Feature values for the year.
    year : int
        Year label (used only for warnings).
    """
    gidx = _doy_grid_index(arr.shape[0])
    if gidx is None:
        print(
            f"    [warn] {f} {year}: unexpected time length {arr.shape[0]} "
            f"(not 365/366) — skipped in seasonal plot"
        )
        return
    a = acc[f]
    if a is None:
        a = acc[f] = {
            "ssum": np.zeros(SEASON_LEN, np.float64),
            "ssq": np.zeros(SEASON_LEN, np.float64),
            "cnt": np.zeros(SEASON_LEN, np.int64),
            "smin": np.full(SEASON_LEN, np.inf, np.float64),
            "smax": np.full(SEASON_LEN, -np.inf, np.float64),
        }
    valid = np.isfinite(arr)
    # gidx is strictly increasing and unique, so scatter-add has no collisions.
    a["ssum"][gidx] += np.nansum(arr, axis=1)
    a["ssq"][gidx] += np.nansum(arr.astype(np.float64) ** 2, axis=1)
    a["cnt"][gidx] += valid.sum(axis=1)
    a["smin"][gidx] = np.minimum(
        a["smin"][gidx], np.where(valid, arr, np.inf).min(axis=1)
    )
    a["smax"][gidx] = np.maximum(
        a["smax"][gidx], np.where(valid, arr, -np.inf).max(axis=1)
    )


def _finalize_seasonal(a: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Turn running accumulators into mean/std/min/max, NaN where no data.

    Parameters
    ----------
    a : dict
        Accumulator with keys ``"ssum"``, ``"ssq"``, ``"cnt"``, ``"smin"``,
        ``"smax"``.

    Returns
    -------
    out : dict
        Dictionary with keys ``"mean"``, ``"std"``, ``"min"``, ``"max"``, each
        an ndarray of shape (366,).
    """
    cnt = a["cnt"]
    empty = cnt == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = a["ssum"] / cnt
        var = a["ssq"] / cnt - mean**2
    std = np.sqrt(np.clip(var, 0.0, None))
    mean[empty] = np.nan
    std[empty] = np.nan
    return {
        "mean": mean,
        "std": std,
        "min": np.where(np.isfinite(a["smin"]), a["smin"], np.nan),
        "max": np.where(np.isfinite(a["smax"]), a["smax"], np.nan),
    }


# ── Plots ────────────────────────────────────────────────────────────────────


def _plot_lai_seasonal(lai: np.ndarray, out_path: str, title: str) -> None:
    """
    Plot the seasonal LAI curve with ±1 std and min/max envelopes.

    Parameters
    ----------
    lai : ndarray of shape (36, n_curves)
        Dekadal LAI curves.
    out_path : str
        Destination PNG path.
    title : str
        Plot title.
    """
    doys = _dekad_doys()
    mean = np.nanmean(lai, axis=1)
    std = np.nanstd(lai, axis=1)
    lo = np.nanmin(lai, axis=1)
    hi = np.nanmax(lai, axis=1)
    n = np.isfinite(lai).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.fill_between(doys, lo, hi, color="#a8dadc", alpha=0.35, label="min–max envelope")
    ax.fill_between(
        doys, mean - std, mean + std, color="#457b9d", alpha=0.35, label="±1 std"
    )
    ax.plot(doys, mean, "o-", color="#1d3557", lw=2.0, markersize=4, label="mean")
    ax.set_xlabel("Day of year (dekadal, days 5/15/25)")
    ax.set_ylabel("LAI")
    ax.set_title(
        f"{title}\nseasonal LAI — {int(n.max()):,} (site, year) " f"curves pooled",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def _grid(n: int) -> Tuple[int, int]:
    """
    Choose a compact subplot grid for ``n`` panels.

    Parameters
    ----------
    n : int
        Number of panels.

    Returns
    -------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns (at most 3).
    """
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _feature_bins(all_vals: np.ndarray) -> np.ndarray:
    """
    Build histogram bins clipped to the 0.5–99.5 percentile range.

    Parameters
    ----------
    all_vals : ndarray
        Pooled values used to compute the percentile range.

    Returns
    -------
    bins : ndarray of shape (HIST_BINS + 1,)
        Bin edges.
    """
    lo, hi = np.percentile(all_vals, CLIP_PCT)
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, HIST_BINS + 1)


def _plot_features_hist(
    data: Dict[str, Dict[int, np.ndarray]], feats: List[str], out_path: str, title: str
) -> None:
    """
    Plot one histogram per feature.

    Parameters
    ----------
    data : dict
        ``{feature: {year: 1-D array}}``.
    feats : list of str
        Features to plot.
    out_path : str
        Destination PNG path.
    title : str
        Plot title.
    """
    nrows, ncols = _grid(len(feats))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        allv = np.concatenate(list(data[f].values())) if data[f] else np.array([])
        if allv.size == 0:
            ax.set_visible(False)
            continue
        bins = _feature_bins(allv)
        ax.hist(allv, bins=bins, color="#457b9d", alpha=0.8)
        ax.set_title(f, fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes[len(feats) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_features_by_year(
    data: Dict[str, Dict[int, np.ndarray]], feats: List[str], out_path: str, title: str
) -> None:
    """
    Plot one density curve per year, per feature.

    Parameters
    ----------
    data : dict
        ``{feature: {year: 1-D array}}``.
    feats : list of str
        Features to plot.
    out_path : str
        Destination PNG path.
    title : str
        Plot title.
    """
    years_all = sorted({y for f in feats for y in data[f]})
    if not years_all:
        print("  ✗ no feature data — skipping by-year plot")
        return
    norm = matplotlib.colors.Normalize(vmin=years_all[0], vmax=years_all[-1])
    cmap = matplotlib.cm.viridis

    nrows, ncols = _grid(len(feats))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        if not data[f]:
            ax.set_visible(False)
            continue
        allv = np.concatenate(list(data[f].values()))
        bins = _feature_bins(allv)
        centers = 0.5 * (bins[:-1] + bins[1:])
        for year in sorted(data[f]):
            dens, _ = np.histogram(data[f][year], bins=bins, density=True)
            ax.plot(centers, dens, color=cmap(norm(year)), lw=1.3)
        ax.set_title(f, fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes[len(feats) :]:
        ax.set_visible(False)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), label="year", shrink=0.85)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_features_seasonal(
    seasonal: Dict[str, Dict[str, np.ndarray]],
    feats: List[str],
    out_path: str,
    title: str,
) -> None:
    """
    Plot the per-day-of-year seasonal profile of each feature.

    Parameters
    ----------
    seasonal : dict
        ``{feature: {"mean", "std", "min", "max"}}`` from
        :func:`_load_features`.
    feats : list of str
        Features to plot.
    out_path : str
        Destination PNG path.
    title : str
        Plot title.
    """
    feats = [f for f in feats if f in seasonal]
    if not feats:
        print("  ✗ no seasonal feature data — skipping")
        return
    nrows, ncols = _grid(len(feats))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        s = seasonal[f]
        doy = np.arange(1, s["mean"].shape[0] + 1)
        ax.fill_between(
            doy, s["min"], s["max"], color="#a8dadc", alpha=0.35, label="min–max"
        )
        ax.fill_between(
            doy,
            s["mean"] - s["std"],
            s["mean"] + s["std"],
            color="#457b9d",
            alpha=0.35,
            label="±1 std",
        )
        ax.plot(doy, s["mean"], color="#1d3557", lw=1.5, label="mean")
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("Day of year")
        ax.grid(alpha=0.25)
    for ax in axes[len(feats) :]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    """
    Run the per-PFT diagnostics.

    Loads the selection, pools LAI over the years, pools the requested
    features, and writes the four PNGs to ``args.output_dir``.

    Raises
    ------
    RuntimeError
        If no LAI data is found for the selected sites.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    site_ids, pft = _load_selected(args.selected_pixels)
    if args.features:
        feats = [s.strip() for s in args.features.split(",") if s.strip()]
    else:
        feats = list(DYNAMIC_FEATURES)

    pft_lbl = (
        f"PFT {pft} ({PFT_NAMES[pft - 1]})"
        if pft is not None and 1 <= pft <= len(PFT_NAMES)
        else Path(args.selected_pixels).stem
    )
    title = f"{pft_lbl} — {len(site_ids):,} cells"
    print(f"{title}\nFeatures: {feats}\n")

    years = range(args.year_start, args.year_end + 1)

    print("Loading LAI ...")
    lai = _load_lai(Path(args.target_dir).resolve(), years, site_ids)

    era5_ids = site_ids
    if args.parent_map:
        pm = _load_parent01_map(args.parent_map)
        missing = [s for s in site_ids if s not in pm]
        if len(missing) == len(site_ids):
            raise ValueError(
                "parent_map matched none of the selected 0.05 degree sites; "
                "check that it was built from the same selection file."
            )
        if missing:
            print(
                f"[warn] parent_map is missing {len(missing):,} of "
                f"{len(site_ids):,} selected sites; they will be skipped."
            )
        era5_ids = np.array(sorted({pm[s] for s in site_ids if s in pm}))
        print(
            f"Parent map  : {args.parent_map} ({len(pm):,} links; "
            f"{era5_ids.size:,} ERA5 cells for {len(site_ids):,} subset sites)"
        )

    print("\nLoading features ...")
    data, seasonal = _load_features(
        Path(args.era5_dir).resolve(), years, era5_ids, feats
    )

    # Drop features with no data (e.g. a name absent from every file).
    feats = [f for f in feats if data[f]]
    if not feats:
        print("[warn] none of the requested features were found in the files.")

    print("\nPlotting ...")
    _plot_lai_seasonal(lai, os.path.join(args.output_dir, "lai_seasonal.png"), title)
    if feats:
        _plot_features_hist(
            data, feats, os.path.join(args.output_dir, "features_hist.png"), title
        )
        _plot_features_by_year(
            data,
            feats,
            os.path.join(args.output_dir, "features_dist_by_year.png"),
            title,
        )
        _plot_features_seasonal(
            seasonal,
            feats,
            os.path.join(args.output_dir, "features_seasonal.png"),
            title,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
