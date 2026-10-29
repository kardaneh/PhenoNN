# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Scan the preprocessed NetCDF cubes for a year range and emit normalisation stats.

Produces one ``norm_stats.json`` = ``{feat: {"mean", "std"}}`` for every channel
the training pipeline z-scores.

Coverage
--------
- ``DYNAMIC_FEATURES`` (METEO [+ DERIVED] in :mod:`phenonn.datagen.config`) —
  from the ERA5 feature cubes.
- ``PFT_COLS`` (``pft1_frac..pft15_frac``, pooled across years) — from PFT.
- ``LAI`` — target z-scoring stats.
- ``co2`` — when ``--co2_path`` is provided.

Heavy-tailed dynamic features are ``log1p``-transformed BEFORE mean/std,
matching what ``phenonn.data.lai_dataset`` does at training time. The output
format is byte-identical to the old ``phenon/data_old/create_stats_file.py``,
so ``--stats_path`` consumers (train / train_full_RAM / predict) work unchanged.

New-paradigm note
-----------------
This port is grid-agnostic: it computes each channel's stats over whatever
``site`` population each file carries. It therefore works when the ERA5
features live on the deduplicated 0.1° grid (``ERA5_daily_pixelset`` with
``site_id`` ``"E{lat}_{lon}"``) while LAI/PFT stay on the native 0.05° sites —
no 0.05°-indexed feature file is required.

The dynamic-feature stats are then pooled over the *unique 0.1° cells* rather
than over the expanded 0.05° sites. For z-scoring this is the standard, correct
approach: the only difference is a per-cell weighting (each 0.1° cell would
otherwise be counted once per child 0.05° site), which shifts mean/std
negligibly and does not affect the normalisation the model relies on.

Notes
-----
Torch-free. All reads use ``engine="netcdf4"`` (mixing h5netcdf here would
load a second bundled libhdf5 → ``NetCDF: HDF error``).

Examples
--------
::

    python -m phenonn.datagen.create_stats_file \\
        --features_dir /path/to/data/era5_pixelset \\
        --target_dir   /path/to/data/LAI_pixelset \\
        --pft_dir      /path/to/data/PFT_pixelset \\
        --co2_path     /path/to/data/CO2_1700_2023_TRENDYv2024.txt \\
        --year_start 1992 --year_end 2019 \\
        --output       /path/to/data/norm_stats_1992_2019.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from phenonn.datagen.config import (
    CO2_FEATURES,
    DYNAMIC_FEATURES,
    FEATURES_FNAME,
    LOG_TRANSFORM_FEATURES,
    N_PFT,
    PFT_COLS,
    PFT_FNAME,
    TARGETS_FNAME,
)


# ── Torch-free local helpers (importing lai_dataset would pull torch) ────────


def load_co2_lut(path: str) -> Dict[int, float]:
    """
    Parse a LaiNN-style CO2 file.

    Parameters
    ----------
    path : str
        Path to a text file with one ``Annee_YYYY=VALUE`` entry per line.

    Returns
    -------
    lut : dict
        Mapping ``year -> concentration`` (float).

    Raises
    ------
    ValueError
        If no ``Annee_YYYY=VALUE`` entry could be parsed.
    """
    lut: Dict[int, float] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.lower().startswith("annee_"):
                continue
            try:
                year = int(key.split("_", 1)[1])
                lut[year] = float(value.strip())
            except ValueError:
                continue
    if not lut:
        raise ValueError(f"No `Annee_YYYY=VALUE` entries parsed from {path!r}.")
    return lut


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Extract a single DataArray with a ``pft`` dim from a PFT file.

    Supports ORCHIDEE ``maxvegetfrac``, 15 separate ``pft{k}_frac`` variables,
    or a single variable with a ``pft`` dim (pixelset ``pft_frac(pft, site)``,
    or grid).

    Parameters
    ----------
    ds : xarray.Dataset
        An open PFT file.

    Returns
    -------
    pft : xarray.DataArray
        A DataArray with dim ``pft``.

    Raises
    ------
    ValueError
        If no recognised PFT layout is found.
    """
    rename = {}
    for src, dst in [("lat", "latitude"), ("lon", "longitude"), ("veget", "pft")]:
        if src in ds.coords or src in ds.dims:
            rename[src] = dst
    if rename:
        ds = ds.rename(rename)

    if "maxvegetfrac" in ds.data_vars:
        da = ds["maxvegetfrac"]
        if "time_counter" in da.dims:
            da = da.isel(time_counter=0, drop=True)
        return da

    pft_named = [f"pft{k}_frac" for k in range(1, N_PFT + 1)]
    if all(v in ds.data_vars for v in pft_named):
        return xr.concat(
            [ds[v].expand_dims(pft=[k - 1]) for k, v in enumerate(pft_named, start=1)],
            dim="pft",
        )

    for cand in ("pft_fraction", "PFT", "pft", "pft_frac", "fraction"):
        if cand in ds.data_vars and "pft" in ds[cand].dims:
            return ds[cand]

    raise ValueError(
        f"Could not detect a PFT layout in {list(ds.data_vars)}. "
        f"Expected one of: 'maxvegetfrac(time_counter, veget, lat, lon)' "
        f"(ORCHIDEE), 15 'pftK_frac' vars, or a 3-D var with a 'pft' dim."
    )


# ── Running stats accumulator ───────────────────────────────────────────────


class RunningStats:
    """
    Online accumulator of mean and standard deviation.

    Merges sum, sum-of-squares and count over finite values only; both
    :meth:`update_from_array` and :meth:`mean_std` are O(1) in memory.

    Attributes
    ----------
    n : int
        Number of finite values seen so far.
    sum_x : float
        Sum of the finite values.
    sum_x2 : float
        Sum of the squared finite values.
    """

    __slots__ = ("n", "sum_x", "sum_x2")

    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update_from_array(self, arr: np.ndarray) -> None:
        finite = np.isfinite(arr)
        if not finite.any():
            return
        x = arr[finite].astype(np.float64)
        self.n += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_x2 += float(np.square(x).sum())

    def mean_std(self) -> Dict[str, float]:
        if self.n == 0:
            return {"mean": 0.0, "std": 1.0}
        mean = self.sum_x / self.n
        var = max(self.sum_x2 / self.n - mean * mean, 0.0)
        std = math.sqrt(var)
        return {"mean": mean, "std": max(std, 1e-8)}


# ── Per-year scans ───────────────────────────────────────────────────────────


def _site_sel(n_site: int, n_sites: int, seed: int) -> Optional[np.ndarray]:
    """
    Return a sorted random subset of site indices, or ``None``.

    Parameters
    ----------
    n_site : int
        Total number of sites in the file.
    n_sites : int
        Desired subset size. ``<= 0`` or ``>= n_site`` → use every site.
    seed : int
        RNG seed for the subset.

    Returns
    -------
    sel : ndarray of shape (n_sites,) or None
        Sorted indices into ``0..n_site-1``, or ``None`` (use every site).
    """
    if n_sites <= 0 or n_site <= 0 or n_sites >= n_site:
        return None
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_site, n_sites, replace=False))


def _scan_features_year(
    path: Path,
    accs: Dict[str, RunningStats],
    n_sites: int,
    seed: int,
) -> None:
    """
    Fold one ERA5 feature year into the dynamic-feature accumulators.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``ERA5_daily_pixelset_{Y}.nc``.
    accs : dict
        Mapping ``feature_name -> RunningStats``.
    n_sites : int
        Sites to subsample (0 = use all).
    seed : int
        RNG seed for the subsample.
    """
    # engine="netcdf4" (same libhdf5 as the targets/PFT reads below) — mixing
    # h5netcdf here would load a second bundled libhdf5 → "NetCDF: HDF error".
    ds = xr.open_dataset(path, engine="netcdf4")
    sel = _site_sel(ds.sizes.get("site", 0), n_sites, seed)
    for feat in DYNAMIC_FEATURES:
        if feat not in ds:
            continue
        da = ds[feat] if sel is None else ds[feat].isel(site=sel)
        arr = da.values.astype(np.float32)
        if feat in LOG_TRANSFORM_FEATURES:
            arr = np.log1p(np.clip(arr, 0.0, None))
        accs[feat].update_from_array(arr)
    ds.close()


def _scan_targets_year(
    path: Path,
    lai_acc: RunningStats,
    n_sites: int,
    seed: int,
) -> None:
    """
    Fold one LAI year into the target accumulator.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``LAI_dekadal_{Y}.nc``.
    lai_acc : RunningStats
        Accumulator for the LAI target.
    n_sites : int
        Sites to subsample (0 = use all).
    seed : int
        RNG seed for the subsample.
    """
    ds = xr.open_dataset(path, engine="netcdf4")
    sel = _site_sel(ds.sizes.get("site", 0), n_sites, seed)
    da = ds["LAI"] if sel is None else ds["LAI"].isel(site=sel)
    lai_acc.update_from_array(da.values.astype(np.float32))
    ds.close()


def _scan_pft_year(
    path: Path, accs: Dict[str, RunningStats], n_sites: int, seed: int
) -> None:
    """
    Fold one PFT year into the per-PFT accumulators.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``PFTmap_{Y}.nc``.
    accs : dict
        Mapping ``"pft{k}_frac" -> RunningStats``.
    n_sites : int
        Sites to subsample (0 = use all).
    seed : int
        RNG seed for the subsample.
    """
    # Pixelset PFT files (pft_frac(pft, site)) read like lai_dataset._open_pft_array:
    # engine="netcdf4", decode_times=False so the process stays on one libhdf5.
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    pft = _load_pft_vector_array(ds)
    sel = _site_sel(pft.sizes.get("site", 0), n_sites, seed)
    if sel is not None:
        pft = pft.isel(site=sel)
    arr = pft.values.astype(np.float32)  # (15, site) for pixelset
    for k in range(N_PFT):
        accs[PFT_COLS[k]].update_from_array(arr[k])
    ds.close()


def _co2_stats(co2_lut: Dict[int, float], years: List[int]) -> Dict[str, float]:
    """
    Compute mean and standard deviation of the annual CO2 concentration.

    Parameters
    ----------
    co2_lut : dict
        Mapping ``year -> concentration``.
    years : list of int
        Years to consider (only those present in ``co2_lut`` are used).

    Returns
    -------
    stats : dict
        Dictionary with keys ``"mean"`` and ``"std"``.
    """
    vals = np.array([co2_lut[y] for y in years if y in co2_lut], dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "std": 1.0}
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    return {"mean": mean, "std": max(std, 1e-8)}


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``features_dir``, ``target_dir``,
        ``pft_dir``, ``co2_path``, ``year_start``, ``year_end``, ``output``,
        ``n_sites``, ``seed``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--features_dir",
        required=True,
        help="Folder of ERA5_daily_pixelset_{Y}.nc (0.1° or 0.05°).",
    )
    p.add_argument(
        "--target_dir",
        required=True,
        help="Folder of LAI_dekadal_{Y}.nc (0.05° pixelset).",
    )
    p.add_argument(
        "--pft_dir", required=True, help="Folder of PFTmap_{Y}.nc (0.05° pixelset)."
    )
    p.add_argument(
        "--co2_path", default="", help="Optional CO2 LUT (Annee_YYYY=VALUE)."
    )
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument(
        "--output", required=True, help="Path of the norm_stats.json to write."
    )
    p.add_argument(
        "--n_sites",
        type=int,
        default=0,
        help="If >0, compute stats on a random subset of this many "
        "sites per file (reproducible via --seed) instead of "
        "every site. Speeds up the scan; 0 = all sites (default).",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the --n_sites subsampling."
    )
    return p.parse_args()


def main():
    """
    Run the normalisation-statistics computation.

    Scans every year in the range, folds ERA5 / LAI / PFT values into the
    accumulators, then writes the assembled stats dict to ``args.output`` as
    JSON with one ``{mean, std}`` entry per channel.

    Raises
    ------
    FileNotFoundError
        If ``args.co2_path`` is set but the file does not exist.
    """
    args = parse_args()
    features_dir = Path(args.features_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    pft_dir = Path(args.pft_dir).resolve()
    years = list(range(args.year_start, args.year_end + 1))

    print(f"Features dir : {features_dir}")
    print(f"Target dir   : {target_dir}")
    print(f"PFT dir      : {pft_dir}")
    print(f"Years        : {args.year_start} → {args.year_end}")
    print(
        f"Sites        : {args.n_sites if args.n_sites > 0 else 'all'}"
        f"{f' (seed={args.seed})' if args.n_sites > 0 else ''}"
    )
    print(f"Output       : {args.output}")

    dyn_accs = {feat: RunningStats() for feat in DYNAMIC_FEATURES}
    pft_accs = {feat: RunningStats() for feat in PFT_COLS}
    lai_acc = RunningStats()

    n_ok_feat = n_ok_tgt = n_ok_pft = 0
    t0 = time.time()
    for y in years:
        feat_path = features_dir / FEATURES_FNAME.format(year=y)
        tgt_path = target_dir / TARGETS_FNAME.format(year=y)
        pft_path = pft_dir / PFT_FNAME.format(year=y)

        if feat_path.exists():
            _scan_features_year(feat_path, dyn_accs, args.n_sites, args.seed)
            n_ok_feat += 1
        else:
            print(f"  [warn] missing features {feat_path.name}")

        if tgt_path.exists():
            _scan_targets_year(tgt_path, lai_acc, args.n_sites, args.seed)
            n_ok_tgt += 1
        else:
            print(f"  [warn] missing targets {tgt_path.name}")

        if pft_path.exists():
            _scan_pft_year(pft_path, pft_accs, args.n_sites, args.seed)
            n_ok_pft += 1
        else:
            print(f"  [warn] missing PFT {pft_path.name}")

        if y % 5 == 0:
            print(f"  scanned through {y}  ({time.time() - t0:.1f}s)")

    print(
        f"\nScanned: features={n_ok_feat}/{len(years)}, "
        f"targets={n_ok_tgt}/{len(years)}, "
        f"pft={n_ok_pft}/{len(years)}  ({time.time() - t0:.1f}s)"
    )

    # ── Assemble stats dict ──
    stats: Dict[str, Dict[str, float]] = {}
    for feat in DYNAMIC_FEATURES:
        stats[feat] = dyn_accs[feat].mean_std()
    for feat in PFT_COLS:
        stats[feat] = pft_accs[feat].mean_std()
    stats["LAI"] = lai_acc.mean_std()

    if args.co2_path:
        if not os.path.exists(args.co2_path):
            raise FileNotFoundError(args.co2_path)
        lut = load_co2_lut(args.co2_path)
        if CO2_FEATURES:
            stats["co2"] = _co2_stats(lut, years)

    print("\nResulting stats:")
    for feat, s in stats.items():
        print(f"  {feat:20s}  mean={s['mean']:+.4f}  std={s['std']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ saved to {args.output}")


if __name__ == "__main__":
    main()
