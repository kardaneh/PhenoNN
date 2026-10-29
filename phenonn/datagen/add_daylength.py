# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Add a photoperiod column (``daylength``, in hours) to the ERA5 pixelset files.

For every site and every day the value is the astronomical day length (sun
centre above the geometric horizon), a deterministic function of the site
LATITUDE and the DAY OF YEAR only — no extra data needed::

    δ  = 0.409 · sin( 2π/365 · J − 1.39 )          solar declination (rad, FAO-56)
    X  = −tan(φ) · tan(δ)                            φ = site latitude (rad)
    daylength = 24/π · arccos( clip(X, −1, 1) )      hours

In the 0.05° paradigm the ERA5 pixelset carries the NATIVE 0.05° site latitude
(``build_daily_dataset_pixelset`` overwrites the coord with the
``selected_pixels`` value), so the photoperiod is computed for the true 0.05°
cell — this script is resolution-agnostic and needs no change between the 0.1°
and 0.05° datasets.

Global validity
---------------
- Both hemispheres are handled by the SIGN of the latitude (φ < 0 in the
  Southern hemisphere flips the seasonal phase automatically).
- Polar day / night: near the poles X leaves [−1, 1]; the clip maps
  X ≤ −1 → 24 h (midnight sun) and X ≥ +1 → 0 h (polar night), so the formula
  stays valid at every latitude up to ±90°.

The definition is purely geometric (sun centre at the horizon); it does not add
the ~0.83° twilight/refraction correction, so it is a hair shorter than civil
daylength — the standard choice for a phenology driver.

Input / output
--------------
Input  (ERA5 pixelset, from ``build_daily_dataset_pixelset.py``)::

    {features_dir}/ERA5_daily_pixelset_{Y}.nc   latitude(site), time(datetime), …

Output (same file layout, one extra variable)::

    {out_dir}/ERA5_daily_pixelset_{Y}.nc        … + daylength(time, site)

``--out_dir`` may equal ``--features_dir`` to overwrite in place.

Notes
-----
Torch-free. Reads use ``engine="netcdf4"``; the write is isolated in a
subprocess (``_ncwrite``) to dodge the netCDF4 / h5py dual-libhdf5 conflict.

Feeding ``daylength`` to the model is a separate step (a config toggle + a read
in ``phenonn.data.lai_dataset``), not done here.

Examples
--------
::

    python -m phenonn.datagen.add_daylength \\
        --features_dir /path/to/data/era5_pixelset \\
        --out_dir      /path/to/data/era5_pixelset_dl \\
        --year_start 1992 --year_end 2019
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.datagen.config import FEATURES_FNAME
from phenonn.datagen._ncwrite import to_netcdf_subprocess


DAYLENGTH_VAR = "daylength"


def _day_of_year(ds: xr.Dataset) -> np.ndarray:
    """
    Day-of-year (1..366) for each step of the ``time`` dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a ``time`` coordinate.

    Returns
    -------
    doy : ndarray of shape (n_time,)
        Day-of-year of each time step, dtype float64. Uses the real dates when
        ``time`` decoded to ``datetime64`` (correct on leap years); falls back
        to a 1-based index if ``time`` is a bare integer.
    """
    tvals = np.asarray(ds["time"].values)
    if np.issubdtype(tvals.dtype, np.datetime64):
        return ds["time"].dt.dayofyear.values.astype(np.float64)
    return np.arange(ds.sizes["time"], dtype=np.float64) + 1.0


def _daylength_hours(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """
    Astronomical day length in hours for ``(time, site)``.

    Parameters
    ----------
    lat_deg : ndarray of shape (n_site,)
        Site latitudes in degrees North (Southern = negative).
    doy : ndarray of shape (n_time,)
        Day-of-year of each time step.

    Returns
    -------
    daylength : ndarray of shape (n_time, n_site)
        Astronomical day length in hours, dtype float64. NaN latitudes
        propagate to NaN.
    """
    phi = np.deg2rad(lat_deg)  # (n_site,)
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)  # (n_time,)
    x = -np.tan(phi)[None, :] * np.tan(decl)[:, None]  # (n_time, n_site)
    x = np.clip(x, -1.0, 1.0)  # polar day/night
    return 24.0 / np.pi * np.arccos(x)


# ── Per-year build ───────────────────────────────────────────────────────────
def _process_year(
    year: int,
    features_dir: Path,
    out_dir: Path,
    complevel: int,
) -> None:
    """
    Compute and write the augmented pixelset for one year.

    Parameters
    ----------
    year : int
        Year to process.
    features_dir : pathlib.Path
        Folder of ``ERA5_daily_pixelset_{Y}.nc`` files.
    out_dir : pathlib.Path
        Folder where the augmented files are written (may equal
        ``features_dir``).
    complevel : int
        zlib compression level.
    """
    fpath = features_dir / FEATURES_FNAME.format(year=year)
    if not fpath.exists():
        print(f"  ✗ {year} skipped — missing {fpath.name}")
        return

    with xr.open_dataset(fpath, engine="netcdf4") as ds:
        ds = ds.load()

    lat_name = (
        "latitude"
        if "latitude" in ds.coords
        else ("lat" if "lat" in ds.coords else None)
    )
    if lat_name is None:
        print(f"  ✗ {year} skipped — no latitude coordinate in {fpath.name}")
        return

    lat_deg = np.asarray(ds[lat_name].values, dtype=np.float64)  # (n_site,)
    doy = _day_of_year(ds)  # (n_time,)
    daylength = _daylength_hours(lat_deg, doy).astype(np.float32)  # (time, site)

    ds[DAYLENGTH_VAR] = (("time", "site"), daylength)
    ds[DAYLENGTH_VAR].attrs.update(
        {
            "long_name": "astronomical day length (photoperiod)",
            "units": "h",  # NOT "hours": that is a CF time unit → xarray would
            # decode this variable to timedelta64 on read (and re-
            # encode it as int64 nanoseconds), corrupting the scale.
            "formula": (
                "decl=0.409*sin(2pi/365*doy-1.39); "
                "daylength=24/pi*arccos(clip(-tan(lat)*tan(decl),-1,1))"
            ),
            "definition": "sun centre at the geometric horizon (no refraction)",
            "note": "hemisphere via latitude sign; polar day=24h / night=0h via clip",
        }
    )

    # Fresh encoding per var (drop source filter hints to avoid subprocess-pickle
    # conflicts) but PRESERVE the source chunking (full time, block of sites) so
    # RamLAIDataset's scattered per-site read still hits few chunks. See the same
    # note in add_soil_moisture_proxy.py.
    ref_var = next(iter(ds.data_vars))  # any (time, site) var
    dl_chunks = ds[ref_var].encoding.get("chunksizes")  # read before clearing
    encoding = {}
    for name, var in ds.data_vars.items():
        orig_chunks = var.encoding.get("chunksizes")
        var.encoding = {}
        enc = {"zlib": True, "complevel": complevel}
        if orig_chunks is not None:
            enc["chunksizes"] = tuple(orig_chunks)
        encoding[name] = enc
    if dl_chunks is not None:  # daylength is a fresh var
        encoding[DAYLENGTH_VAR]["chunksizes"] = tuple(dl_chunks)
    encoding[DAYLENGTH_VAR]["_FillValue"] = np.float32(np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(
        ds,
        out_dir / FEATURES_FNAME.format(year=year),
        encoding=encoding,
        engine="netcdf4",
    )
    print(f"  ✓ {year}  (+{DAYLENGTH_VAR}, {daylength.shape[1]:,} sites)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``features_dir``, ``out_dir``,
        ``year_start``, ``year_end``, ``complevel``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--features_dir", required=True, help="Folder of ERA5_daily_pixelset_{Y}.nc."
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Where to write the augmented files "
        "(may equal --features_dir to overwrite in place).",
    )
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args()


def main():
    """
    Run the day-length augmentation over a year range.

    Iterates ``range(args.year_start, args.year_end + 1)`` calling
    :func:`_process_year`; per-year failures are caught, logged, and do not
    abort the run.
    """
    args = parse_args()
    features_dir = Path(args.features_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"ERA5 pixelset in : {features_dir}")
    print(f"Augmented out    : {out_dir}")
    print(f"Years            : {args.year_start} → {args.year_end}")
    print(f"Adding           : {DAYLENGTH_VAR} (hours, latitude × day-of-year)\n")

    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, features_dir, out_dir, args.complevel)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
