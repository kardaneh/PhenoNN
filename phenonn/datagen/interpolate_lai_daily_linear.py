# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Turn the 36 dekadal LAI observations of each year into a DAILY LAI curve.

The interpolation is linear between the dekads, placed at their real
day-of-year positions.

Behaviour
---------
- The curve passes EXACTLY through every finite dekad (no smoothing).
- Straight segments between anchors; NaN dekads are simply skipped and the
  line is drawn between the surrounding valid ones.
- Outside the valid range (before the first / after the last valid dekad) the
  value is held constant (``np.interp`` edge behaviour), NOT extrapolated.
- No model fit, so nothing can fail to converge; no scipy dependency.

A site needs at least 2 valid dekads to define a line; with fewer it is written
as NaN and counted.

Input / output
--------------
Input  (pixelset LAI, from ``build_pixelset_targets.py``)::

    {target_dir}/LAI_dekadal_{Y}.nc     LAI(dekad=36, site)

Output (same flat pixelset layout, new day axis)::

    {out_dir}/LAI_daily_{Y}.nc          LAI(day=365, site)

with ``coord day = DOY 1..365`` and the site coords copied from the input file.

Notes
-----
Torch-free. Reads use ``engine="netcdf4"``; the write is isolated in a
subprocess (``_ncwrite``) to dodge the netCDF4 / h5py dual-libhdf5 conflict.

Examples
--------
::

    python -m phenonn.datagen.interpolate_lai_daily_linear \\
        --target_dir /path/to/data/LAI_pixelset \\
        --out_dir    /path/to/data/LAI_daily_linear \\
        --year_start 1992 --year_end 2019
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.datagen.config import N_DEKAD_YEAR, TARGETS_FNAME
from phenonn.datagen._ncwrite import to_netcdf_subprocess


DAILY_FNAME = "LAI_daily_{year}.nc"
N_DAYS = 365
MIN_VALID = 2  # a line needs two points
_SITE_COORDS = ("site_id", "lat_idx", "lon_idx", "latitude", "longitude")


# ── Dekad calendar (leap-safe) ───────────────────────────────────────────────


def _dekad_doys(year: int) -> np.ndarray:
    """
    Day-of-year of the 36 dekad anchors for a given year.

    Parameters
    ----------
    year : int
        Year (leap-safe: uses ``datetime.date`` internally).

    Returns
    -------
    doys : ndarray of shape (36,)
        Day-of-year of days 5/15/25 of each month, 1-based, dtype float64.
    """
    return np.array(
        [
            datetime.date(year, m, d).timetuple().tm_yday
            for m in range(1, 13)
            for d in (5, 15, 25)
        ],
        dtype=np.float64,
    )


# ── Per-year build ───────────────────────────────────────────────────────────


def _process_year(year: int, target_dir: Path, out_dir: Path, complevel: int) -> None:
    """
    Interpolate the dekadal LAI to daily for one year.

    Parameters
    ----------
    year : int
        Year to process.
    target_dir : pathlib.Path
        Folder of ``LAI_dekadal_{Y}.nc`` pixelsets.
    out_dir : pathlib.Path
        Folder where ``LAI_daily_{Y}.nc`` is written.
    complevel : int
        zlib compression level.

    Notes
    -----
    A site with fewer than 2 finite dekads is written as NaN and counted in
    the printed summary.
    """
    tpath = target_dir / TARGETS_FNAME.format(year=year)
    if not tpath.exists():
        print(f"  ✗ {year} skipped — missing {tpath.name}")
        return

    with xr.open_dataset(tpath, engine="netcdf4") as ds:
        lai = ds["LAI"].transpose("dekad", "site").values.astype(np.float64)
        if lai.shape[0] != N_DEKAD_YEAR:
            print(f"  ✗ {year} skipped — {lai.shape[0]} dekads, expected 36")
            return
        site_coords = {
            c: (("site",), np.asarray(ds[c].values))
            for c in _SITE_COORDS
            if c in ds.coords
        }
        lai_attrs = dict(ds["LAI"].attrs)

    doy = _dekad_doys(year)  # (36,)
    days = np.arange(1, N_DAYS + 1, dtype=np.float64)
    n_site = lai.shape[1]
    out = np.full((N_DAYS, n_site), np.nan, dtype=np.float32)

    n_too_few = 0
    for j in range(n_site):
        col = lai[:, j]
        finite = np.isfinite(col)
        if int(finite.sum()) < MIN_VALID:
            n_too_few += 1
            continue
        # np.interp needs increasing x; the dekad DOYs already are.
        out[:, j] = np.interp(days, doy[finite], col[finite]).astype(np.float32)

    ds_out = xr.Dataset(
        {"LAI": (("day", "site"), out)},
        coords={"day": np.arange(1, N_DAYS + 1, dtype=np.int32), **site_coords},
    )
    ds_out["LAI"].attrs.update(lai_attrs)
    ds_out["LAI"].attrs["interpolation"] = "linear"
    ds_out.attrs["source_dekadal_file"] = tpath.name
    ds_out.attrs["n_sites"] = int(n_site)
    ds_out.attrs["n_sites_too_few_valid"] = int(n_too_few)

    enc = {
        "LAI": {"zlib": True, "complevel": complevel, "_FillValue": np.float32(np.nan)}
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(
        ds_out, out_dir / DAILY_FNAME.format(year=year), encoding=enc, engine="netcdf4"
    )
    n_ok = n_site - n_too_few
    print(f"  ✓ {year}  ({n_ok:,}/{n_site:,} sites, {n_too_few} <2 valid dekads)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``target_dir``, ``out_dir``,
        ``year_start``, ``year_end``, ``complevel``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--target_dir", required=True, help="Folder of pixelset LAI_dekadal_{Y}.nc."
    )
    p.add_argument("--out_dir", required=True, help="Where to write LAI_daily_{Y}.nc.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--complevel", type=int, default=1)
    return p.parse_args()


def main():
    """
    Run the daily-LAI interpolation over a year range.

    Iterates ``range(args.year_start, args.year_end + 1)`` calling
    :func:`_process_year`; per-year failures are caught, logged, and do not
    abort the run.
    """
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"Pixelset LAI in : {target_dir}")
    print(f"Daily LAI out   : {out_dir}")
    print(f"Years           : {args.year_start} → {args.year_end}")
    print(f"Model           : linear interp, {N_DAYS}-day output\n")

    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, target_dir, out_dir, args.complevel)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
