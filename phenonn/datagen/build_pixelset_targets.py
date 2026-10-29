# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Pre-extract full-grid LAI and PFT into *pixelset* files indexed by selected sites.

Outputs the same flat ``(site)``-indexed layout as
``ERA5_daily_pixelset_{Y}.nc``.

Why
---
``LAI_dekadal_{Y}.nc`` and ``PFTmap_{Y}.nc`` are FULL native 0.05° grids
(3600×7200). Reading a few hundred scattered sites from them with a vectorised
``isel(lat=…, lon=…)`` forces HDF5 to decompress almost every chunk, so the
same chunks are decompressed many times (thrashing). Over NFS this is minutes
per year. Extracting once to a ``(dekad, site)`` / ``(pft, site)`` file turns
every later read into a few kB.

Output
------
Same filenames as the grid files, written into the ``--out_*`` dirs:

- ``{out_target_dir}/LAI_dekadal_{Y}.nc`` — ``LAI(dekad, site)``
- ``{out_pft_dir}/PFTmap_{Y}.nc`` — ``pft_frac(pft, site)``

Both carry ``site_id`` / ``lat_idx`` / ``lon_idx`` / ``latitude`` /
``longitude`` coords so the data loader can map ``site_id → row`` exactly like
the ERA5 pixelset.

Point training at the new dirs::

    --target_dir {out_target_dir}  --pft_dir {out_pft_dir}

``phenonn.data.lai_dataset`` auto-detects the pixelset layout (a ``site`` dim).

Notes
-----
On the ``.venv_ERA5`` environment, netcdf4 and h5py ship separate bundled
libhdf5 that corrupt each other in one process. Importing ``phenon`` pulls
netcdf4's libhdf5 first, so BOTH grids are read with ``engine="netcdf4"`` (this
node opens the ORCHIDEE PFTmap fine that way) to keep the whole process on one
libhdf5. The write is still isolated in a fresh subprocess (``_ncwrite``).

Examples
--------
::

    python -m phenonn.datagen.build_pixelset_targets \\
        --selected_pixels /data/ecarlin/pixelset1/selected_pixels005_1.nc \\
        --target_dir      /path/to/full_grid/LAI \\
        --pft_dir         /path/to/full_grid/PFT \\
        --out_target_dir  /data/ecarlin/pixelset1/LAI_pixelset \\
        --out_pft_dir     /data/ecarlin/pixelset1/PFT_pixelset \\
        --year_start 1992 --year_end 2019
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from phenonn.datagen.config import N_DEKAD_YEAR, N_PFT, PFT_FNAME, TARGETS_FNAME
from phenonn.datagen._ncwrite import to_netcdf_subprocess


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Extract a single ``(pft, latitude, longitude)`` DataArray from a PFT file.

    Supports three layouts: ORCHIDEE ``maxvegetfrac(time_counter, veget, lat,
    lon)``; 15 separate ``pft{k}_frac`` variables; or a single variable with a
    ``pft`` dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        An open PFT file.

    Returns
    -------
    pft : xarray.DataArray
        A DataArray with dims ``(pft, latitude, longitude)``.

    Raises
    ------
    ValueError
        If no recognised PFT layout is found.

    Notes
    -----
    Inlined here (instead of importing from ``phenonn.data.lai_dataset``) so this
    preprocessing step does not pull in torch. Mirrors the loader in
    ``phenonn.data.lai_dataset`` / ``phenonn.datagen.compute_valid_pixels``.
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


def _read_selected(selected_path: Path):
    """
    Read a ``selected_pixels`` file into a dict of numpy arrays.

    Parameters
    ----------
    selected_path : pathlib.Path
        Path to a ``selected_pixels*.nc`` file.

    Returns
    -------
    sel : dict
        Dictionary with keys ``"site_id"`` (object array of str),
        ``"lat_idx"`` (int64), ``"lon_idx"`` (int64), ``"latitude"``
        (float32), ``"longitude"`` (float32).
    """
    ds = xr.open_dataset(selected_path)
    sel = {
        "site_id": np.asarray(ds["site_id"].values).astype(str),
        "lat_idx": ds["lat_idx"].values.astype(np.int64),
        "lon_idx": ds["lon_idx"].values.astype(np.int64),
        "latitude": np.asarray(ds["latitude"].values, dtype=np.float32),
        "longitude": np.asarray(ds["longitude"].values, dtype=np.float32),
    }
    ds.close()
    return sel


def _site_coords(sel) -> dict:
    """
    Build the ``site``-dim coordinates dict for an output Dataset.

    Parameters
    ----------
    sel : dict
        Selection dict as returned by :func:`_read_selected`.

    Returns
    -------
    coords : dict
        Mapping suitable for ``xarray.Dataset(coords=...)``, with one entry
        per site-level coordinate.
    """
    return {
        "site_id": ("site", sel["site_id"]),
        "lat_idx": ("site", sel["lat_idx"]),
        "lon_idx": ("site", sel["lon_idx"]),
        "latitude": ("site", sel["latitude"]),
        "longitude": ("site", sel["longitude"]),
    }


def _process_year(
    year: int,
    sel,
    target_dir: Path,
    pft_dir: Path,
    out_target_dir: Path,
    out_pft_dir: Path,
    complevel: int,
) -> None:
    """
    Extract the LAI and PFT pixelsets for one year.

    Parameters
    ----------
    year : int
        Year to process.
    sel : dict
        Selection dict as returned by :func:`_read_selected`.
    target_dir : pathlib.Path
        Folder of full-grid ``LAI_dekadal_{Y}.nc`` files.
    pft_dir : pathlib.Path
        Folder of full-grid ``PFTmap_{Y}.nc`` files.
    out_target_dir : pathlib.Path
        Folder where the LAI pixelset is written.
    out_pft_dir : pathlib.Path
        Folder where the PFT pixelset is written.
    complevel : int
        zlib compression level.

    Notes
    -----
    Reads use ``engine="netcdf4"`` for both grids (see the module docstring).
    The writes go through ``_ncwrite.to_netcdf_subprocess``.
    """
    lat_idx, lon_idx = sel["lat_idx"], sel["lon_idx"]

    # ── LAI grid → pixelset ──
    tpath = target_dir / TARGETS_FNAME.format(year=year)
    if tpath.exists():
        # engine="netcdf4" for BOTH reads: importing phenon pulls netcdf4's
        # libhdf5 into this process first, so a later h5netcdf read trips the
        # dual-libhdf5 conflict ("H5DSget_num_scales <0"). On this node netcdf4
        # opens the ORCHIDEE PFTmap fine, so the whole process stays on netcdf4
        # (the documented fix). The write is still isolated in a subprocess.
        with xr.open_dataset(tpath, engine="netcdf4") as tgt:
            # One sequential decompression pass of the whole grid, then a numpy
            # fancy-index — no scattered isel, no chunk-cache thrashing.
            lai_full = tgt["LAI"].transpose("dekad", "lat", "lon").values
            vals = lai_full[:, lat_idx, lon_idx].astype(np.float32)  # (36, n_site)
            dekad = (
                tgt["dekad"].values
                if "dekad" in tgt.coords
                else np.arange(N_DEKAD_YEAR)
            )
            lai_attrs = dict(tgt["LAI"].attrs)
        out = xr.Dataset(
            {"LAI": (("dekad", "site"), vals)},
            coords={"dekad": dekad, **_site_coords(sel)},
        )
        out["LAI"].attrs.update(lai_attrs)
        enc = {
            "LAI": {
                "zlib": True,
                "complevel": complevel,
                "_FillValue": np.float32(np.nan),
            }
        }
        out_target_dir.mkdir(parents=True, exist_ok=True)
        to_netcdf_subprocess(
            out,
            out_target_dir / TARGETS_FNAME.format(year=year),
            encoding=enc,
            engine="netcdf4",
        )
        n_nan = int(np.isnan(vals).all(axis=0).sum())
        print(f"  ✓ LAI    {year}  ({vals.shape[1]:,} sites, {n_nan} all-NaN)")
    else:
        print(f"  ✗ LAI    {year} skipped — missing {tpath.name}")

    # ── PFT grid → pixelset ──
    ppath = pft_dir / PFT_FNAME.format(year=year)
    if ppath.exists():
        with xr.open_dataset(ppath, engine="netcdf4", decode_times=False) as pds:
            pda = _load_pft_vector_array(pds).transpose("pft", "latitude", "longitude")
            pft_full = pda.values
            pv = pft_full[:, lat_idx, lon_idx].astype(np.float32)  # (15, n_site)
        outp = xr.Dataset(
            {"pft_frac": (("pft", "site"), pv)},
            coords={"pft": np.arange(N_PFT, dtype=np.int32), **_site_coords(sel)},
        )
        enc = {
            "pft_frac": {
                "zlib": True,
                "complevel": complevel,
                "_FillValue": np.float32(np.nan),
            }
        }
        out_pft_dir.mkdir(parents=True, exist_ok=True)
        to_netcdf_subprocess(
            outp,
            out_pft_dir / PFT_FNAME.format(year=year),
            encoding=enc,
            engine="netcdf4",
        )
        print(f"  ✓ PFT    {year}  ({pv.shape[1]:,} sites)")
    else:
        print(f"  ✗ PFT    {year} skipped — missing {ppath.name}")


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``selected_pixels``, ``target_dir``,
        ``pft_dir``, ``out_target_dir``, ``out_pft_dir``, ``year_start``,
        ``year_end``, ``complevel``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels.nc (site_id, lat_idx, lon_idx, " "latitude, longitude).",
    )
    p.add_argument("--target_dir", required=True, help="Grid LAI_dekadal_*.nc in.")
    p.add_argument("--pft_dir", required=True, help="Grid PFTmap_*.nc in.")
    p.add_argument("--out_target_dir", required=True, help="Pixelset LAI out.")
    p.add_argument("--out_pft_dir", required=True, help="Pixelset PFT out.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--complevel", type=int, default=1)
    return p.parse_args()


def main():
    """
    Run the pixelset extraction over a year range.

    Loads the selection, iterates over ``range(args.year_start,
    args.year_end + 1)`` calling :func:`_process_year`, and reports the
    skipped years at the end. Per-year failures are caught, logged, and do not
    abort the run.
    """
    args = parse_args()
    selected_path = Path(args.selected_pixels).resolve()
    target_dir = Path(args.target_dir).resolve()
    pft_dir = Path(args.pft_dir).resolve()
    out_target = Path(args.out_target_dir).resolve()
    out_pft = Path(args.out_pft_dir).resolve()

    sel = _read_selected(selected_path)
    print(f"Selected pixels : {selected_path}  ({len(sel['site_id']):,} sites)")
    print(f"Grid LAI in     : {target_dir}")
    print(f"Grid PFT in     : {pft_dir}")
    print(f"Pixelset LAI out: {out_target}")
    print(f"Pixelset PFT out: {out_pft}")
    print(f"Years           : {args.year_start} → {args.year_end}")

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(
                year, sel, target_dir, pft_dir, out_target, out_pft, args.complevel
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
