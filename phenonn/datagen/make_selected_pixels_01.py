# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Collapse a native 0.05° ``selected_pixels.nc`` onto the ERA5-Land 0.1° grid.

A 0.1° ERA5-Land cell is kept **iff at least one** selected 0.05° cell falls
inside it, and each kept cell appears **exactly once** (no duplication). Four
0.05° cells share one 0.1° cell, so extracting ERA5 per 0.05° site does up to
4× redundant work; running the ERA5 extraction on this deduplicated 0.1° set
removes that redundancy (see ``build_daily_dataset_pixelset.py``).

Mapping (0.05° cell → containing 0.1° ERA5-Land cell)
-----------------------------------------------------
::

    0.05° THEIA grid : cell-registered, lat 89.975 → -89.975 (row 0..3599),
                       lon -179.975 → 179.975 (col 0..7199).
    ERA5-Land 0.1°   : node-registered, lat 90.0 → -90.0 (1801 rows, idx 0..1800),
                       lon 0.0..359.9 (3600 cols, idx 0..3599).

    lat_idx_era5 = (row05 + 1) // 2                  # 0..1800
    lon_idx_m180 = ((col05 + 1) // 2) % 3600         # index on a -180..180 grid
    lon_idx_era5 = (lon_idx_m180 + 1800) % 3600      # roll -180..180 → 0..360

Two consecutive 0.05° centres straddle one 0.1° node, hence the "+1 then //2".
This is the same mapping ``build_daily_dataset_pixelset.py`` applies internally;
here we bake it once and deduplicate.

Output
------
Two dims — the deduplicated 0.1° set and the original 0.05° sites:

``site`` = N01 unique 0.1° cells:

- ``site_id`` (str): ``"E{lat_idx:04d}_{lon_idx:04d}"``
- ``lat_idx`` (int32): raw ERA5-Land latitude index 0..1800 (90 → -90)
- ``lon_idx`` (int32): raw ERA5-Land longitude index 0..3599 (0..360)
- ``latitude`` (float64): ERA5 node-cell centre latitude
- ``longitude`` (float64): ERA5 node-cell centre longitude, -180..180

``site05`` = N05 original 0.05° sites (unchanged order):

- ``site_id05`` (str): original 0.05° site_id
- ``parent_site`` (int32): index into ``site`` of the 0.1° cell containing it

``lat_idx`` / ``lon_idx`` are the raw ERA5-Land indices, so a reader extracts
the meteo with no roll and no containment mapping::

    ds[var].isel(latitude=lat_idx, longitude=lon_idx)

To later broadcast the 0.1° ERA5 features back onto the 0.05° sites (to align
with the 0.05° LAI/PFT targets), index by ``parent_site``::

    feat_05 = feat_01.isel(site=parent_site)          # (time, site05)

and ``site_id05`` joins that back to the LAI/PFT pixelsets by site_id.

Notes
-----
Torch-free. The read uses ``engine="netcdf4"``; the write is isolated in a
subprocess (``_ncwrite``) to dodge the netCDF4 / h5py dual-libhdf5 conflict.

Examples
--------
::

    python -m phenonn.datagen.make_selected_pixels_01 \\
        --selected_pixels /data/ecarlin/pixelset1/selected_pixels005_1.nc \\
        --output          /data/ecarlin/pixelset1/selected_pixels01_1.nc
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.datagen._ncwrite import to_netcdf_subprocess


# ── ERA5-Land 0.1° grid ───────────────────────────────────────────────────────
ERA5_RES = 0.1
N_LAT_ERA5 = 1801  # node-registered, 90.0 → -90.0
N_LON_ERA5 = 3600  # node-registered, 0.0 → 359.9


def _map_to_era5(row05: np.ndarray, col05: np.ndarray):
    """
    Map 0.05° grid indices to the containing ERA5-Land 0.1° cell indices.

    Parameters
    ----------
    row05 : ndarray of shape (n_site,)
        Row indices on the native 0.05° grid (0..3599, top to bottom).
    col05 : ndarray of shape (n_site,)
        Column indices on the native 0.05° grid (0..7199, -180° to 180°).

    Returns
    -------
    lat_idx : ndarray of shape (n_site,)
        Raw ERA5-Land latitude index (0..1800, 90° to -90°), dtype int64.
    lon_idx : ndarray of shape (n_site,)
        Raw ERA5-Land longitude index (0..3599, 0° to 360°), dtype int64.
    """
    lat_idx = (row05 + 1) // 2  # 0..1800
    lon_m180 = ((col05 + 1) // 2) % N_LON_ERA5  # -180..180 index
    lon_idx = (lon_m180 + N_LON_ERA5 // 2) % N_LON_ERA5  # → 0..360 index
    return lat_idx, lon_idx


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``selected_pixels`` and ``output``.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="Native 0.05° selected_pixels.nc "
        "(site dim: site_id, lat_idx, lon_idx, latitude, longitude).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Where to write the deduplicated 0.1° selected_pixels.",
    )
    return p.parse_args()


def main():
    """
    Run the 0.05° → 0.1° collapse.

    Reads the native 0.05° selection, maps each site to its containing
    ERA5-Land 0.1° cell, deduplicates the resulting cells, and writes a new
    selection file carrying both the deduplicated 0.1° set (dim ``site``) and
    the original 0.05° sites with their parent index (dim ``site05``).

    Raises
    ------
    AssertionError
        If any computed ERA5 index falls outside the ERA5-Land grid bounds.
    """
    args = parse_args()
    sel_path = Path(args.selected_pixels).resolve()
    out_path = Path(args.output).resolve()

    with xr.open_dataset(sel_path, engine="netcdf4") as ds:
        site_id05 = np.asarray(ds["site_id"].values).astype(str)
        row05 = ds["lat_idx"].values.astype(np.int64)
        col05 = ds["lon_idx"].values.astype(np.int64)
    n05 = site_id05.size

    lat_idx, lon_idx = _map_to_era5(row05, col05)
    assert (
        lat_idx.min() >= 0 and lat_idx.max() <= N_LAT_ERA5 - 1
    ), f"lat_idx out of ERA5 range: [{lat_idx.min()}, {lat_idx.max()}]"
    assert (
        lon_idx.min() >= 0 and lon_idx.max() <= N_LON_ERA5 - 1
    ), f"lon_idx out of ERA5 range: [{lon_idx.min()}, {lon_idx.max()}]"

    # Deduplicate: one row per unique (lat_idx, lon_idx). np.unique's inverse
    # maps each 0.05° site to its row in the unique set = parent_site.
    key = lat_idx * N_LON_ERA5 + lon_idx
    uniq, parent_site = np.unique(key, return_inverse=True)
    parent_site = np.asarray(parent_site).ravel()  # 1-D across numpy versions
    lat_idx_u = (uniq // N_LON_ERA5).astype(np.int32)
    lon_idx_u = (uniq % N_LON_ERA5).astype(np.int32)
    n01 = uniq.size

    # ERA5 node-cell centres; longitude rolled back to -180..180 for readability
    # (lat_idx/lon_idx stay the RAW 0..360 ERA5 indices used for isel).
    lat_u = 90.0 - lat_idx_u * ERA5_RES
    lon_u = lon_idx_u * ERA5_RES
    lon_u = np.where(lon_u > 180.0, lon_u - 360.0, lon_u)
    site_id01 = np.array(
        [f"E{la:04d}_{lo:04d}" for la, lo in zip(lat_idx_u, lon_idx_u)], dtype=object
    )

    out = xr.Dataset(
        {
            "lat_idx": ("site", lat_idx_u),
            "lon_idx": ("site", lon_idx_u),
            "latitude": ("site", lat_u.astype(np.float64)),
            "longitude": ("site", lon_u.astype(np.float64)),
            "site_id": ("site", site_id01),
            "parent_site": ("site05", parent_site.astype(np.int32)),
            "site_id05": ("site05", site_id05.astype(object)),
        }
    )
    out["lat_idx"].attrs["long_name"] = (
        "ERA5-Land latitude index (raw, 0..1800, 90→-90)"
    )
    out["lon_idx"].attrs["long_name"] = (
        "ERA5-Land longitude index (raw, 0..3599, 0..360)"
    )
    out["lon_idx"].attrs["usage"] = "ds[var].isel(latitude=lat_idx, longitude=lon_idx)"
    out["latitude"].attrs["units"] = "degrees_north"
    out["longitude"].attrs["units"] = "degrees_east (-180..180)"
    out["parent_site"].attrs["long_name"] = (
        "index into `site` of the 0.1° cell containing this 0.05° site"
    )
    out["parent_site"].attrs["usage"] = "feat_05 = feat_01.isel(site=parent_site)"
    out.attrs.update(
        {
            "title": "ERA5-Land 0.1° selected pixels, deduplicated from a 0.05° set",
            "source": "phenonn.datagen/make_selected_pixels_01.py",
            "grid": "ERA5-Land 0.1° node-registered (lat 90→-90, lon 0..360)",
            "n_selected_05": int(n05),
            "n_selected_01": int(n01),
        }
    )

    # Per-0.1°-cell counts of covered 0.05° cells (diagnostic).
    counts = np.bincount(parent_site, minlength=n01)
    print(f"0.05° selected pixels : {n05:,}")
    print(
        f"0.1° unique cells     : {n01:,}  "
        f"({n05 - n01:,} duplicates removed, ratio {n05 / max(n01, 1):.2f}×)"
    )
    print(
        f"0.05° cells per 0.1° cell : min={counts.min()}  "
        f"mean={counts.mean():.2f}  max={counts.max()}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(out, out_path, encoding={}, engine="netcdf4")
    print(f"✓ wrote {out_path}")


if __name__ == "__main__":
    main()
