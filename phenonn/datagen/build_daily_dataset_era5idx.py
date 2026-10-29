# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Build a FLAT (time, site) ERA5-Land daily cube on deduplicated 0.1° cells.

This is the sibling of ``build_daily_dataset_pixelset.py``. The ONLY difference
is the site→grid mapping in :func:`main`:

- ``build_daily_dataset_pixelset.py`` : reads a 0.05° ``selected_pixels`` and
  maps each 0.05° cell to its containing 0.1° ERA5 cell (``(idx+1)//2`` +
  longitude roll);
- this script : reads the deduplicated 0.1° ``selected_pixels`` produced by
  ``make_selected_pixels_01.py``, whose ``lat_idx`` / ``lon_idx`` are ALREADY
  the raw ERA5-Land indices → used as-is in
  ``isel(latitude=…, longitude=…)``.

Using the deduplicated 0.1° set removes the up-to-4× redundancy of extracting
ERA5 once per 0.05° cell (four 0.05° cells share one 0.1° cell), so the
extraction and write are ~4× lighter than the 0.05° extractor.

Pipeline position
-----------------
::

    make_selected_pixels_01.py  →  selected_pixels01_1.nc      <- input here
    THIS SCRIPT                 →  ERA5_daily_pixelset_{Y}.nc    (site = 0.1° cells)

Re-expand later to the 0.05° sites with ``parent_site`` from
``selected_pixels01_1.nc``::

    feat_05 = feat_01.isel(site=parent_site)

Output schema
-------------
``dims: time=365, site=N01`` (the deduplicated 0.1° ERA5 cells)

Base variables: ``Tmin``, ``Tmax``, ``Tmean``, ``ssrd_sum``, ``strd_sum``,
``tp_sum``, ``VPD_max``, ``VPD_mean``, ``Rn_tot``, ``PET``. Append
``daylength`` and ``SMI`` afterwards with the two augmentation scripts to reach
the 12-channel report contract.

Coordinates: ``time``, ``latitude(site)``, ``longitude(site)``,
``site_id(site)``, ``lat_idx(site)``, ``lon_idx(site)`` — ``lat_idx`` /
``lon_idx`` are the raw ERA5-Land 0.1° indices used for ``isel``;
``site_id`` / ``latitude`` / ``longitude`` describe that 0.1° cell.

Notes
-----
Torch-free. The write uses ``engine="netcdf4"`` on the in-process Dataset; no
h5py is loaded.

Examples
--------
::

    python -m phenonn.datagen.build_daily_dataset_era5idx \\
        --input_dir  /bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF \\
        --output_dir /data/ecarlin/pixelset1/era5_pixelset \\
        --selected_pixels /data/ecarlin/pixelset1/selected_pixels01_1.nc \\
        --year_start 1992 --year_end 2019
"""

import argparse
import gc
import time as time_mod
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


HOURLY_VARS = ["t2m", "d2m", "sp", "ssrd", "strd", "tp"]


# ── Tetens saturation vapour pressure ───────────────────────────────────────


def _e_sat(T_celsius):
    """
    Tetens saturation vapour pressure.

    Parameters
    ----------
    T_celsius : array-like
        Air temperature in degrees Celsius.

    Returns
    -------
    e_sat : array-like
        Saturation vapour pressure in kPa, same shape as ``T_celsius``.
    """
    return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))


# ── Open + vectorised pixel extraction ──────────────────────────────────────


def _open_var_year(
    var: str,
    year: int,
    input_dir: Path,
    chunks: Dict[str, int],
    lat_idx_da: xr.DataArray,
    lon_idx_da: xr.DataArray,
) -> xr.DataArray:
    """
    Open the 12 monthly hourly files for one (var, year) pair and extract the
    selected pixels.

    Parameters
    ----------
    var : str
        ERA5-Land variable name (``"t2m"``, ``"d2m"``, ``"sp"``, ``"ssrd"``,
        ``"strd"`` or ``"tp"``).
    year : int
        Year to extract.
    input_dir : pathlib.Path
        Folder containing the per-variable monthly hourly NetCDF files.
    chunks : dict
        Dask chunking.
    lat_idx_da, lon_idx_da : xarray.DataArray
        RAW ERA5-Land indices with dim ``"site"``.

    Returns
    -------
    da : xarray.DataArray
        Extracted array with dims ``(time, site)``.

    Raises
    ------
    FileNotFoundError
        If no hourly files match any of the expected patterns for ``var`` and
        ``year`` under ``input_dir``.
    """
    patterns = [f"{var}.{year}*.nc", f"{var}_{year}*.nc"]
    files: List[str] = []
    for pat in patterns:
        files.extend(str(p) for p in input_dir.rglob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No files found for variable {var!r} year {year} under "
            f"{input_dir} (patterns: {patterns})"
        )

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False, chunks=chunks)
    da = ds[var].sel(time=str(year))
    da = da.isel(latitude=lat_idx_da, longitude=lon_idx_da)
    return da


def _resample(da: xr.DataArray, how: str) -> xr.DataArray:
    """
    Resample an hourly DataArray to daily values with a reduction.

    Parameters
    ----------
    da : xarray.DataArray
        Hourly array with a ``"time"`` dimension.
    how : str
        Reduction method: ``"min"``, ``"max"``, ``"mean"``, …

    Returns
    -------
    daily : xarray.DataArray
        Daily-resampled array, same non-time dims as ``da``.
    """
    return getattr(da.resample(time="1D"), how)()


def _resample_accum(da: xr.DataArray) -> xr.DataArray:
    """
    Daily total of an ERA5-Land accumulation (cumulative since 00 UTC).

    See :func:`build_daily_dataset_pixelset._resample_accum` for the full
    rationale; this is the identical implementation.

    Parameters
    ----------
    da : xarray.DataArray
        Hourly accumulation array with a ``"time"`` dimension.

    Returns
    -------
    daily_total : xarray.DataArray
        Daily accumulation total.
    """
    shifted = da.assign_coords(time=da["time"] - np.timedelta64(1, "h"))
    # skipna=False: take the literal last hourly step of each day (the cumulative
    # endpoint). dask does not implement `last` with skipna=True; skipna=False is
    # also what we want here (ocean NaN stays NaN).
    return shifted.resample(time="1D").last(skipna=False)


def _build_daily_aggregations(
    year: int,
    input_dir: Path,
    chunks: Dict[str, int],
    lat_idx_da: xr.DataArray,
    lon_idx_da: xr.DataArray,
) -> xr.Dataset:
    """
    Build the raw daily aggregations for one year (before derived variables).

    Parameters
    ----------
    year : int
        Year to aggregate.
    input_dir : pathlib.Path
        Folder of ERA5-Land hourly files.
    chunks : dict
        Dask chunking for the open.
    lat_idx_da, lon_idx_da : xarray.DataArray
        Site-index DataArrays on the ERA5 grid.

    Returns
    -------
    daily : xarray.Dataset
        Daily aggregations with variables ``Tmin``, ``Tmax``, ``Tmean``,
        ``Tdew_mean``, ``sp_mean``, ``ssrd_sum``, ``strd_sum``, ``tp_sum``.
    """
    daily = xr.Dataset()

    t2m = _open_var_year("t2m", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["Tmin"] = _resample(t2m, "min") - 273.15
    daily["Tmax"] = _resample(t2m, "max") - 273.15
    daily["Tmean"] = _resample(t2m, "mean") - 273.15

    d2m = _open_var_year("d2m", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["Tdew_mean"] = _resample(d2m, "mean") - 273.15

    sp = _open_var_year("sp", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["sp_mean"] = _resample(sp, "mean") / 100.0

    # ssrd / strd / tp are ERA5-Land accumulations reset at 00 UTC: the hourly
    # value already holds the running total since the start of the day, so the
    # daily total is the LAST step of the day (via _resample_accum, which also
    # shifts -1 h so the 24 h total lands on the right day), NOT the sum of the
    # 24 cumulative values (summing those over-counts ~5-12x). Units unchanged.
    ssrd = _open_var_year("ssrd", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["ssrd_sum"] = _resample_accum(ssrd) / 1e6

    strd = _open_var_year("strd", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["strd_sum"] = _resample_accum(strd) / 1e6

    tp = _open_var_year("tp", year, input_dir, chunks, lat_idx_da, lon_idx_da)
    daily["tp_sum"] = _resample_accum(tp) * 1000.0

    return daily


# ── Vectorised derived vars (identical to nosnow) ───────────────────────────


def _derive_vpd(daily: xr.Dataset) -> xr.Dataset:
    """
    Add ``VPD_max`` and ``VPD_mean`` to the daily dataset.

    Parameters
    ----------
    daily : xarray.Dataset
        Dataset with ``Tmax``, ``Tmean`` and ``Tdew_mean``.

    Returns
    -------
    daily : xarray.Dataset
        Same dataset with ``VPD_max`` and ``VPD_mean`` added (kPa, clipped to
        non-negative).
    """
    e_a = _e_sat(daily["Tdew_mean"])
    daily["VPD_max"] = (_e_sat(daily["Tmax"]) - e_a).clip(min=0.0)
    daily["VPD_mean"] = (_e_sat(daily["Tmean"]) - e_a).clip(min=0.0)
    return daily


def _derive_radiation(daily: xr.Dataset) -> xr.Dataset:
    """
    Add ``Rn_mean`` and ``Rn_tot`` to the daily dataset.

    Parameters
    ----------
    daily : xarray.Dataset
        Dataset with ``ssrd_sum``, ``strd_sum`` and ``Tmean``.

    Returns
    -------
    daily : xarray.Dataset
        Same dataset with ``Rn_mean`` and ``Rn_tot`` added.
    """
    ssrd_avg = daily["ssrd_sum"] * 1e6 / 86400.0
    strd_avg = daily["strd_sum"] * 1e6 / 86400.0
    tmean_K = daily["Tmean"] + 273.15
    rn_mean = (1 - 0.23) * ssrd_avg + strd_avg - 0.97 * 5.67e-8 * tmean_K**4
    daily["Rn_mean"] = rn_mean
    daily["Rn_tot"] = rn_mean * 0.0864
    return daily


def _derive_pet(daily: xr.Dataset) -> xr.Dataset:
    """
    Add ``PET`` (Priestley-Taylor) to the daily dataset.

    Parameters
    ----------
    daily : xarray.Dataset
        Dataset with ``Tmean``, ``sp_mean`` and ``Rn_tot``.

    Returns
    -------
    daily : xarray.Dataset
        Same dataset with ``PET`` added.
    """
    Tmean = daily["Tmean"]
    delta = 4098.0 * _e_sat(Tmean) / (Tmean + 237.3) ** 2
    P_kpa = daily["sp_mean"] / 10.0
    gamma = 0.000665 * P_kpa
    pet = 1.26 * (delta / (delta + gamma)) * (daily["Rn_tot"] / 2.45)
    daily["PET"] = xr.where(daily["Rn_tot"] < 0, 0.0, pet)
    return daily


# ── Attributes / encoding ───────────────────────────────────────────────────


_ATTRS = {
    "Tmin": ("Daily minimum 2 m air temperature", "degC"),
    "Tmax": ("Daily maximum 2 m air temperature", "degC"),
    "Tmean": ("Daily mean 2 m air temperature", "degC"),
    "Tdew_mean": ("Daily mean 2 m dew-point temperature", "degC"),
    "sp_mean": ("Daily mean surface pressure", "hPa"),
    "ssrd_sum": ("Daily downward short-wave radiation", "MJ m-2"),
    "strd_sum": ("Daily downward long-wave radiation", "MJ m-2"),
    "tp_sum": ("Daily total precipitation", "mm"),
    "VPD_max": ("Daily max vapour pressure deficit (Tmax)", "kPa"),
    "VPD_mean": ("Daily mean vapour pressure deficit (Tmean)", "kPa"),
    "Rn_mean": ("Daily mean net radiation", "W m-2"),
    "Rn_tot": ("Daily total net radiation", "MJ m-2 day-1"),
    "PET": ("Priestley–Taylor potential evapotranspiration", "mm day-1"),
}


def _set_attrs_and_encoding(
    daily: xr.Dataset,
    year: int,
    complevel: int = 4,
    site_chunk: int = 4096,
) -> dict:
    """
    Assign long names, units and NetCDF encoding to a daily dataset.

    Parameters
    ----------
    daily : xarray.Dataset
        Dataset to annotate; modified in place.
    year : int
        Year used to build the time encoding ``units`` string.
    complevel : int, optional
        zlib compression level. Default is 4.
    site_chunk : int, optional
        Target sites per chunk. Default is 4096.

    Returns
    -------
    encoding : dict
        Encoding dictionary suitable for ``Dataset.to_netcdf``.
    """
    for name, (long_name, units) in _ATTRS.items():
        if name in daily:
            daily[name].attrs["long_name"] = long_name
            daily[name].attrs["units"] = units

    daily["time"].encoding.update(
        {
            "units": f"days since {year}-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
            "dtype": "int32",
        }
    )

    n_time = daily.sizes["time"]
    n_site = daily.sizes["site"]
    encoding = {}
    for name in daily.data_vars:
        encoding[name] = {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (n_time, min(site_chunk, n_site)),
        }
    return encoding


# ── Per-year orchestration ──────────────────────────────────────────────────


def process_year(
    year: int,
    input_dir: Path,
    output_dir: Path,
    lat_idx_da: xr.DataArray,
    lon_idx_da: xr.DataArray,
    selected_meta: xr.Dataset,
    time_chunk: int,
    lat_chunk: int,
    lon_chunk: int,
    complevel: int,
) -> Path:
    """
    Build and write the ERA5 daily pixelset for one year (0.1° route).

    Parameters
    ----------
    year : int
        Year to process.
    input_dir : pathlib.Path
        Folder of ERA5-Land hourly files.
    output_dir : pathlib.Path
        Folder where ``ERA5_daily_pixelset_{Y}.nc`` is written.
    lat_idx_da, lon_idx_da : xarray.DataArray
        RAW ERA5-Land indices with dim ``"site"``.
    selected_meta : xarray.Dataset
        The open ``selected_pixels01_1`` file.
    time_chunk, lat_chunk, lon_chunk : int
        Dask chunk sizes.
    complevel : int
        zlib compression level.

    Returns
    -------
    out_path : pathlib.Path
        Path of the written file.
    """
    print(f"\n── Year {year} ─────────────────────────────")
    t0 = time_mod.time()

    chunks = {"time": time_chunk, "latitude": lat_chunk, "longitude": lon_chunk}

    daily = _build_daily_aggregations(
        year,
        input_dir,
        chunks,
        lat_idx_da,
        lon_idx_da,
    )
    print(f"  built daily aggregations (lazy)  +{time_mod.time()-t0:5.1f}s")

    daily = _derive_vpd(daily)
    daily = _derive_radiation(daily)
    daily = _derive_pet(daily)
    print(f"  derived VPD / Rn / PET (lazy)    +{time_mod.time()-t0:5.1f}s")

    # Drop intermediate vars not used as features downstream:
    #   - Tdew_mean : only needed to compute VPD
    #   - sp_mean   : only needed to compute PET
    #   - Rn_mean   : only needed to compute Rn_tot
    daily = daily.drop_vars(
        [v for v in ("Tdew_mean", "sp_mean", "Rn_mean") if v in daily.data_vars]
    )

    # Record the 0.1° ERA5 cell identity on the 'site' dim, straight from the
    # selected_pixels01_1 file: site_id ("E{lat}_{lon}"), the cell-centre lat/lon,
    # and the RAW ERA5 lat_idx/lon_idx used for the isel. (The isel already
    # inherited the same ERA5 cell centres, but we overwrite so the file carries
    # the exact selected_pixels01_1 metadata.)
    daily = daily.assign_coords(
        {
            "site_id": ("site", selected_meta["site_id"].values),
            "latitude": ("site", selected_meta["latitude"].values),
            "longitude": ("site", selected_meta["longitude"].values),
            "lat_idx": ("site", selected_meta["lat_idx"].values),
            "lon_idx": ("site", selected_meta["lon_idx"].values),
        }
    )

    encoding = _set_attrs_and_encoding(daily, year, complevel=complevel)
    daily.attrs["title"] = (
        f"ERA5-Land daily aggregations + derived variables, "
        f"0.1° pixel subset (raw ERA5 idx) — {year}"
    )
    daily.attrs["source"] = "build_daily_dataset_era5idx.py"
    daily.attrs["n_selected"] = int(daily.sizes["site"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ERA5_daily_pixelset_{year}.nc"
    print(f"  writing {out_path.name} ...")
    with ProgressBar(dt=5.0):
        daily.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

    daily.close()
    del daily
    gc.collect()

    print(f"  ✓ done in {time_mod.time()-t0:6.1f}s  →  {out_path}")
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``input_dir``, ``output_dir``,
        ``selected_pixels``, ``year_start``, ``year_end``, ``time_chunk``,
        ``lat_chunk``, ``lon_chunk``, ``complevel``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Folder with per-variable monthly hourly NetCDFs.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Where to write ERA5_daily_pixelset_<year>.nc.",
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels01_1.nc from make_selected_pixels_01.py "
        "(site dim: RAW ERA5 lat_idx/lon_idx).",
    )
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end", type=int, default=2025)
    p.add_argument("--time_chunk", type=int, default=24)
    p.add_argument("--lat_chunk", type=int, default=1801)
    p.add_argument("--lon_chunk", type=int, default=3600)
    p.add_argument("--complevel", type=int, default=1)
    return p.parse_args()


def main():
    """
    Run the 0.1° ERA5 daily-pixelset build over a year range.

    Loads ``selected_pixels01_1.nc`` (raw ERA5 indices), wraps its ``lat_idx`` /
    ``lon_idx`` as site-dim DataArrays, and iterates over
    ``range(args.year_start, args.year_end + 1)``. Per-year failures are
    caught, logged, and do not abort the run. A running ETA is printed.
    """
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    sel_path = Path(args.selected_pixels).resolve()

    # ── Load selected pixels ──
    selected = xr.open_dataset(sel_path)
    n_site = selected.sizes["site"]
    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Selected    : {sel_path.name}  ({n_site:,} pixels, RAW ERA5 idx)")
    print(f"Years       : {args.year_start} → {args.year_end}")
    print(f"Time chunk  : {args.time_chunk} h")
    print(f"Hourly tile : {args.lat_chunk} × {args.lon_chunk}")
    print(f"Compression : zlib level {args.complevel}")

    # selected_pixels01_1.nc already stores the RAW ERA5-Land 0.1° indices
    # (lat_idx 0..1800 for 90→-90, lon_idx 0..3599 for 0..360), baked by
    # make_selected_pixels_01.py. Use them directly — NO (idx+1)//2, NO
    # longitude roll (that mapping lives in build_daily_dataset_pixelset.py,
    # which consumes 0.05° indices instead). Wrap as 'site'-dim DataArrays so
    # .isel(latitude=…, longitude=…) returns a (time, site) cube.
    lat_idx_da = xr.DataArray(selected["lat_idx"].values.astype(np.int64), dims="site")
    lon_idx_da = xr.DataArray(selected["lon_idx"].values.astype(np.int64), dims="site")

    skipped: List[int] = []
    year_times: List[float] = []
    years_to_do = list(range(args.year_start, args.year_end + 1))
    n_total = len(years_to_do)

    for i, year in enumerate(years_to_do, start=1):
        t_year = time_mod.time()
        try:
            process_year(
                year,
                input_dir,
                output_dir,
                lat_idx_da,
                lon_idx_da,
                selected,
                time_chunk=args.time_chunk,
                lat_chunk=args.lat_chunk,
                lon_chunk=args.lon_chunk,
                complevel=args.complevel,
            )
            year_times.append(time_mod.time() - t_year)
        except FileNotFoundError as e:
            print(f"  ✗ {year} skipped — {e}")
            skipped.append(year)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

        n_done = i
        n_left = n_total - n_done
        if year_times and n_left > 0:
            avg = sum(year_times) / len(year_times)
            eta_s = avg * n_left
            mm, ss = divmod(int(eta_s), 60)
            hh, mm = divmod(mm, 60)
            print(
                f"  [progress] {n_done}/{n_total} years done  "
                f"avg/year={avg:5.1f}s  ETA={hh:d}h{mm:02d}m{ss:02d}s"
            )

    selected.close()
    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
