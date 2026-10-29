# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Sample a fixed subset of pixels from the union of yearly valid masks.

Pipeline position
-----------------
::

    compute_valid_pixels.py  →  valid_pixels_{YYYY}.nc   (per-year masks)
                                      ↓
    select_pixels.py         →  selected_pixels005_1.nc (1 file, list of pixels)
                                      ↓
    make_selected_pixels_01.py ; build_pixelset_targets.py
                                      ↓
    selected_pixels01_1.nc        PFTmap_{Y}.nc, LAI_{Y}.nc
    build_daily_dataset_era5idx.py -> ERA5_daily_pixelset_{Y}.nc

Algorithm
---------
1. Open every ``valid_pixels_{YYYY}.nc`` found in ``--valid_dir``.
2. Compute the union of their ``valid`` masks (a pixel is kept if it is valid
   in at least one year), or the intersection if ``--min_years`` equals the
   number of files.
3. Randomly sample ``--fraction`` (default 0.1) of these qualifying pixels
   with a seeded RNG for reproducibility.
4. Write a single ``selected_pixels.nc`` containing ``lat_idx``, ``lon_idx``,
   ``latitude``, ``longitude`` and ``site_id`` of the selected pixels, in a
   single ``site`` dimension.

Output schema
-------------
``site = N_selected``; variables ``lat_idx`` (int32), ``lon_idx`` (int32),
``latitude`` (float32), ``longitude`` (float32), ``site_id`` (str,
``"pix_LLLL_OOOOO"``). Attributes: ``fraction``, ``seed``, ``min_years``,
``n_qualifying``, ``n_total``, ``n_selected``, ``source``, ``valid_files``.

Notes
-----
Torch-free. Uses xarray + numpy only; no subprocess write is needed because
the output is a tiny coordinate table, not a multi-GB cube.

Examples
--------
Run as a module::

    python -m phenonn.datagen.select_pixels \\
        --valid_dir   /path/to/data/valid_pixels \\
        --output_path /data/ecarlin/pixelset1/selected_pixels005_1.nc \\
        --fraction    0.1 \\
        --seed        42
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr


def encode_site_id(lat_idx: int, lon_idx: int) -> str:
    """
    Encode a site identifier from its grid indices.

    Parameters
    ----------
    lat_idx : int
        Latitude index on the native 0.05° grid (row, 0-based).
    lon_idx : int
        Longitude index on the native 0.05° grid (column, 0-based).

    Returns
    -------
    site_id : str
        Identifier of the form ``"pix_LLLL_OOOOO"``, with a 4-digit latitude
        index and a 5-digit longitude index.

    Notes
    -----
    Inlined here (instead of importing from ``phenonn.data.lai_dataset``) so this
    preprocessing step does not pull in torch.

    Examples
    --------
    >>> encode_site_id(12, 3456)
    'pix_0012_03456'
    """
    return f"pix_{int(lat_idx):04d}_{int(lon_idx):05d}"


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``valid_dir``, ``output_path``,
        ``fraction``, ``min_years`` and ``seed``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--valid_dir",
        required=True,
        help="Folder of valid_pixels_{Y}.nc produced by " "compute_valid_pixels.py",
    )
    p.add_argument(
        "--output_path",
        required=True,
        help="Output file (e.g. /data/sbarbu/selected_pixels.nc)",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of qualifying pixels to keep (default 0.1).",
    )
    p.add_argument(
        "--min_years",
        type=int,
        default=1,
        help="Keep only pixels valid in at least N years (default 1 "
        "= union). Pass N = number-of-files for intersection.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    return p.parse_args()


def main():
    """
    Run the pixel-selection pipeline.

    Reads every ``valid_pixels_{YYYY}.nc`` in ``args.valid_dir``, accumulates
    a per-pixel validity counter across years, keeps pixels with
    ``counter >= args.min_years``, samples ``args.fraction`` of them without
    replacement with a seeded RNG, and writes the selection to
    ``args.output_path``.

    Raises
    ------
    FileNotFoundError
        If no ``valid_pixels_*.nc`` files are found in ``args.valid_dir``.
    ValueError
        If ``args.min_years`` exceeds the number of valid_pixels files, or if
        the valid masks do not all share the same ``(lat, lon)`` shape.
    """
    args = parse_args()
    valid_dir = Path(args.valid_dir).resolve()
    files: List[str] = sorted(glob.glob(str(valid_dir / "valid_pixels_*.nc")))
    if not files:
        raise FileNotFoundError(f"No valid_pixels_*.nc files in {valid_dir!r}")
    print(f"Found {len(files)} valid_pixels file(s) in {valid_dir}.")

    # ── Open the first file to get reference grid + start the counter ──
    ref = xr.open_dataset(files[0])
    n_lat = ref.sizes["lat"]
    n_lon = ref.sizes["lon"]
    ref_lat = ref["lat"].values.astype(np.float32)
    ref_lon = ref["lon"].values.astype(np.float32)

    # int16 counter: how many years each pixel was valid.
    counter = ref["valid"].astype(bool).values.astype(np.int16)
    ref.close()

    for f in files[1:]:
        ds = xr.open_dataset(f)
        if ds.sizes["lat"] != n_lat or ds.sizes["lon"] != n_lon:
            raise ValueError(
                f"{os.path.basename(f)} has different grid shape — "
                f"valid masks must share the same (lat, lon) reference."
            )
        counter += ds["valid"].astype(bool).values.astype(np.int16)
        ds.close()

    min_years = max(1, int(args.min_years))
    if min_years > len(files):
        raise ValueError(
            f"--min_years {min_years} > number of valid_pixels files "
            f"({len(files)})."
        )

    qualifying = counter >= min_years
    n_total = int(qualifying.size)
    n_qualifying = int(qualifying.sum())
    n_sample = int(round(n_qualifying * args.fraction))
    print(f"Counter range : 0 … {int(counter.max())} years per pixel")
    print(f"min_years     : {min_years}/{len(files)} (years a pixel must be valid)")
    print(
        f"Qualifying    : {n_qualifying:,} pixels / {n_total:,} "
        f"({100 * n_qualifying / n_total:.2f} %)"
    )
    print(
        f"Target        : {n_sample:,} pixels "
        f"({args.fraction * 100:.1f} % of qualifying)"
    )

    # ── Sample without replacement ──
    lat_idx_all, lon_idx_all = np.where(qualifying)
    n_qualifying = int(lat_idx_all.size)
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(n_qualifying, size=n_sample, replace=False))
    lat_idx_sel = lat_idx_all[sel].astype(np.int32)
    lon_idx_sel = lon_idx_all[sel].astype(np.int32)
    lat_sel = ref_lat[lat_idx_sel]
    lon_sel = ref_lon[lon_idx_sel]
    site_ids = np.array(
        [encode_site_id(int(la), int(lo)) for la, lo in zip(lat_idx_sel, lon_idx_sel)],
        dtype=object,
    )

    # ── Build output ──
    out = xr.Dataset(
        {
            "lat_idx": (["site"], lat_idx_sel),
            "lon_idx": (["site"], lon_idx_sel),
            "latitude": (["site"], lat_sel),
            "longitude": (["site"], lon_sel),
            "site_id": (["site"], site_ids),
        },
        attrs={
            "fraction": float(args.fraction),
            "seed": int(args.seed),
            "min_years": min_years,
            "n_qualifying": n_qualifying,
            "n_total": n_total,
            "n_selected": n_sample,
            "source": "select_pixels.py",
            "valid_files": ", ".join(os.path.basename(f) for f in files),
        },
    )

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "lat_idx": {"zlib": True, "complevel": 4, "dtype": "int32"},
        "lon_idx": {"zlib": True, "complevel": 4, "dtype": "int32"},
        "latitude": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "longitude": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    out.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    print(f"✓ Saved {n_sample:,} pixels → {output_path}")


if __name__ == "__main__":
    main()
