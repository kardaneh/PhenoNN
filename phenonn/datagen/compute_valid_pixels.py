# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Build the per-year mask of usable pixels.

A pixel is kept when all three conditions hold:

1. ``Σ_k PFT_k(year) > --pft_threshold`` — it is a land pixel;
2. ``PFT1 (bare soil) < --max_bare_frac`` — it is not mostly bare ground;
3. ``n_valid_dekads >= --min_valid_dekads`` (default 26 = 26/36 ≈ 70 % obs).

A dekad counts as valid when its LAI is finite (not THEIA nodata) AND it is
not rejected by the RMSE test. The RMSE test only rejects dekads whose RMSE is
PRESENT and too large relative to the pixel's yearly peak LAI, i.e.
``LAI_RMSE / max_year(LAI) >= --max_rmse`` (default 0.3); a MISSING RMSE (NaN)
keeps the dekad. Using the ratio rather than the absolute RMSE avoids a set
dominated by low-LAI pixels (absolute RMSE grows with LAI).

Output
------
``{output_dir}/valid_pixels_{YYYY}.nc`` with one int8 variable ``valid`` on the
native 0.05 degree LAI/PFT grid (same coords as the target file).

Notes
-----
The data loader (``phenonn.data.lai_dataset.RamLAIDataset``) uses this mask via
``list_valid_sites(valid_dir, years)`` to draw its random sample pool — no
NaN pixels ever get fed to the model.

Reads use ``engine="h5netcdf"`` so the whole process stays on the h5py family
(the ORCHIDEE PFTmap can only be opened by h5netcdf on the cluster venv).
Writes are isolated in a subprocess to dodge the netCDF4 / h5py dual-libhdf5
conflict.

Examples
--------
::

    python -m phenonn.datagen.compute_valid_pixels \\
        --target_dir  /path/to/full_grid/LAI \\
        --pft_dir     /path/to/full_grid/PFT \\
        --output_dir  /data/ecarlin/pixelset1/valid_pixels \\
        --year_start  1992 --year_end 2019 \\
        --pft_threshold 0.99 --max_bare_frac 0.4 \\
        --min_valid_dekads 34 --max_rmse 0.3

The report selection uses PFT sum > 0.99, bare soil < 0.4, and at least 34
valid dekads out of 36. The script defaults are more permissive (0.5, 0.5,
26), so pass the report values explicitly when rebuilding that selection.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr

from phenonn.datagen.config import (
    N_DEKAD_YEAR,
    N_PFT,
    PFT_FNAME,
    TARGETS_FNAME,
    VALID_FNAME,
)
from phenonn.datagen._ncwrite import to_netcdf_subprocess


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Extract a single ``(pft, lat, lon)`` DataArray from a PFT file.

    Supports three layouts:

    1. ORCHIDEE PFTmap (native ESACCI / IPSL output):
       ``maxvegetfrac(time_counter=1, veget=15, lat, lon)`` — ``time_counter``
       is squeezed, ``veget`` is renamed to ``pft``.
    2. 15 separate variables ``pft1_frac..pft15_frac`` on
       ``(lat|latitude, lon|longitude)``.
    3. One 3-D variable on ``(pft, lat|latitude, lon|longitude)`` — fallback
       for ``pft_fraction``, ``PFT``, ``pft``, ``pft_frac``, ``fraction``.

    Parameters
    ----------
    ds : xarray.Dataset
        An open PFT file.

    Returns
    -------
    pft : xarray.DataArray
        A DataArray with dims ``(pft, lat, lon)`` and shape
        ``(N_PFT, n_lat, n_lon)``.

    Raises
    ------
    ValueError
        If no recognised PFT layout is found in ``ds``.

    Notes
    -----
    Inlined here (instead of importing from ``phenonn.data.lai_dataset``) so this
    preprocessing step does not pull in torch.
    """
    rename = {}
    for src, dst in [("latitude", "lat"), ("longitude", "lon"), ("veget", "pft")]:
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


def _process_year(
    year: int,
    target_dir: Path,
    pft_dir: Path,
    output_dir: Path,
    pft_threshold: float,
    max_bare_frac: float,
    min_valid_dekads: int,
    max_rmse: float,
    complevel: int,
) -> None:
    """
    Compute and write the validity mask for one year.

    Parameters
    ----------
    year : int
        Year to process.
    target_dir : pathlib.Path
        Folder of full-grid ``LAI_dekadal_{Y}.nc`` files.
    pft_dir : pathlib.Path
        Folder of full-grid ``PFTmap_{Y}.nc`` files.
    output_dir : pathlib.Path
        Folder where ``valid_pixels_{Y}.nc`` is written.
    pft_threshold : float
        Minimum sum of PFT fractions to call the pixel "land".
    max_bare_frac : float
        Maximum PFT1 (bare soil) fraction to keep a pixel.
    min_valid_dekads : int
        Minimum number of valid dekads per year.
    max_rmse : float
        Maximum relative RMSE (``LAI_RMSE / max_year(LAI)``) for a dekad to
        count as valid. A missing RMSE keeps the dekad.
    complevel : int
        zlib compression level for the output.

    Raises
    ------
    KeyError
        If the target file has no ``LAI_RMSE`` variable.

    Notes
    -----
    Reads use ``engine="h5netcdf"`` so the process stays on the h5py family;
    the write is isolated in a fresh subprocess (see ``_ncwrite``).
    """
    tgt_path = target_dir / TARGETS_FNAME.format(year=year)
    pft_path = pft_dir / PFT_FNAME.format(year=year)
    if not tgt_path.exists():
        print(f"  ✗ {year} skipped — missing {tgt_path.name}")
        return
    if not pft_path.exists():
        print(f"  ✗ {year} skipped — missing {pft_path.name}")
        return

    # Read EVERYTHING with the h5py family (h5netcdf) in this process. The
    # ORCHIDEE PFTmap can only be opened by h5netcdf, and on the cluster venv
    # the netcdf4 and h5py wheels bundle separate libhdf5 that corrupt each
    # other when both load in one process (here: netcdf4 read of the target
    # would then make the h5netcdf PFT read fail in H5DSget_num_scales).
    # The actual NetCDF write is already isolated in a fresh subprocess
    # (to_netcdf_subprocess), so the parent stays netcdf4-free. See
    # phenonn.datagen/_ncwrite.py.
    tgt = xr.open_dataset(tgt_path, engine="h5netcdf")
    pft_ds = xr.open_dataset(pft_path, engine="h5netcdf", decode_times=False)
    pft = _load_pft_vector_array(pft_ds)

    if "LAI_RMSE" not in tgt:
        raise KeyError(
            f"{tgt_path.name} has no LAI_RMSE variable — rerun preprocess_targets "
            f"(it now writes the RMSE band required by the dekad criterion)."
        )

    # Land pixels (Σ PFT > threshold) and bare-soil fraction (PFT1). Handle the
    # case where the PFT file is on coords slightly different from the target's
    # grid (reindex both with nearest).
    pft_sum = pft.sum(dim="pft")
    bare = pft.isel(pft=0)  # PFT1 = bare soil (ORCHIDEE ordering)
    if pft_sum.sizes.get("lat") != tgt.sizes.get("lat") or pft_sum.sizes.get(
        "lon"
    ) != tgt.sizes.get("lon"):
        rk = dict(lat=tgt["lat"], lon=tgt["lon"], method="nearest", tolerance=0.03)
        pft_sum = pft_sum.reindex(**rk)
        bare = bare.reindex(**rk)
    land_mask = pft_sum > pft_threshold
    bare_ok = bare < max_bare_frac

    # A dekad is valid when LAI is finite (not THEIA nodata) AND it is NOT
    # rejected by the RMSE test. The RMSE test only rejects dekads whose RMSE is
    # PRESENT and too large RELATIVE to the pixel's yearly peak LAI:
    # LAI_RMSE / max(LAI) >= max_rmse. A MISSING RMSE (NaN) does not disqualify a
    # valid LAI observation — the ratio avoids a set dominated by low-LAI pixels
    # (absolute RMSE grows with LAI, so a fixed cutoff would reject high-LAI
    # pixels and admit only sparse ones).
    max_lai = tgt["LAI"].max(dim="dekad")  # per-pixel yearly peak LAI
    rel_rmse = tgt["LAI_RMSE"] / max_lai  # NaN where RMSE missing / peak≈0
    # ~(rel_rmse >= max_rmse) is True when rel_rmse is NaN, so a missing RMSE is
    # kept; only a finite ratio >= max_rmse rejects the dekad.
    dekad_ok = tgt["LAI"].notnull() & ~(rel_rmse >= max_rmse)
    valid_count = dekad_ok.sum(dim="dekad")
    enough_obs = valid_count >= min_valid_dekads

    valid = (land_mask & bare_ok & enough_obs).astype("int8").rename("valid")
    valid.attrs.update(
        {
            "long_name": "Usable pixel for the LAI training set",
            "description": (
                "True when Σ PFT > pft_threshold AND PFT1 "
                "(bare soil) < max_bare_frac AND at least "
                "min_valid_dekads dekads have finite LAI not "
                "rejected by the RMSE test (a dekad is rejected "
                "only if LAI_RMSE is present and "
                "LAI_RMSE / max_year(LAI) >= max_rmse; a missing "
                "RMSE keeps the dekad)."
            ),
            "pft_threshold": float(pft_threshold),
            "max_bare_frac": float(max_bare_frac),
            "min_valid_dekads": int(min_valid_dekads),
            "max_rmse": float(max_rmse),
        }
    )

    out = valid.to_dataset()
    out["valid_count"] = valid_count.astype("int16")
    out["valid_count"].attrs["long_name"] = (
        "Number of dekads with finite LAI "
        "not rejected by the RMSE test "
        "(missing RMSE kept) (0..36)"
    )
    out["pft_sum"] = pft_sum.astype("float32")
    out["pft_sum"].attrs["long_name"] = "Sum of PFT fractions per pixel"
    out["bare_frac"] = bare.astype("float32")
    out["bare_frac"].attrs["long_name"] = "PFT1 (bare soil) fraction per pixel"

    encoding = {
        "valid": {"zlib": True, "complevel": complevel, "dtype": "int8"},
        "valid_count": {"zlib": True, "complevel": complevel, "dtype": "int16"},
        "pft_sum": {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "bare_frac": {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / VALID_FNAME.format(year=year)
    # Reads PFT via h5netcdf (h5py) earlier in this process, so a same-process
    # netcdf4 write trips "NetCDF: HDF error" on the cluster venv. Write from a
    # fresh h5py-free subprocess. See phenonn.datagen/_ncwrite.py.
    to_netcdf_subprocess(out, out_path, encoding=encoding, engine="netcdf4")
    n_valid = int(valid.sum())
    n_total = int(valid.size)
    print(
        f"  ✓ {out_path.name}  {n_valid:,} / {n_total:,} "
        f"({100 * n_valid / max(n_total, 1):.1f} %) valid pixels"
    )
    tgt.close()
    pft_ds.close()


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``target_dir``, ``pft_dir``,
        ``output_dir``, ``year_start``, ``year_end``, ``pft_threshold``,
        ``max_bare_frac``, ``min_valid_dekads``, ``max_rmse``, ``complevel``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target_dir", required=True)
    p.add_argument("--pft_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--year_start", type=int, default=1981)
    p.add_argument("--year_end", type=int, default=2025)
    p.add_argument(
        "--pft_threshold",
        type=float,
        default=0.5,
        help="Minimum Σ PFT to call the pixel 'land' (default 0.5).",
    )
    p.add_argument(
        "--max_bare_frac",
        type=float,
        default=0.5,
        help="Maximum PFT1 (bare soil) fraction to keep a pixel " "(default 0.5).",
    )
    p.add_argument(
        "--min_valid_dekads",
        type=int,
        default=26,
        help="Minimum valid dekads per year (default 26, ≈ 72%% of the year).",
    )
    p.add_argument(
        "--max_rmse",
        type=float,
        default=0.3,
        help="Max RELATIVE RMSE (LAI_RMSE / yearly max LAI) for a "
        "dekad to count as valid (default 0.3).",
    )
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args()


def main():
    """
    Run ``compute_valid_pixels`` over a year range.

    Iterates ``range(args.year_start, args.year_end + 1)``, calls
    :func:`_process_year` for each, and reports the skipped years at the end.
    Per-year failures are caught, logged, and do not abort the run.
    """
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    pft_dir = Path(args.pft_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    print(f"Target dir : {target_dir}")
    print(f"PFT dir    : {pft_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Years      : {args.year_start} → {args.year_end}")
    print(
        f"Criteria   : Σ PFT > {args.pft_threshold}  AND  "
        f"bare(PFT1) < {args.max_bare_frac}  AND  "
        f"n_valid ≥ {args.min_valid_dekads}/{N_DEKAD_YEAR}  "
        f"(dekad valid if LAI finite & RMSE/max(LAI) < {args.max_rmse})"
    )

    skipped: List[int] = []
    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(
                year,
                target_dir,
                pft_dir,
                output_dir,
                args.pft_threshold,
                args.max_bare_frac,
                args.min_valid_dekads,
                args.max_rmse,
                args.complevel,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")
            skipped.append(year)

    print("\nDone.")
    if skipped:
        print(f"Skipped years: {skipped}")


if __name__ == "__main__":
    main()
