# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Down-sample a ``selected_pixels.nc`` to match a reference spatial distribution.

Method
------
For each N°×N° block ``b`` on the globe:

- ``ref_frac[b] = ref_count[b] / ref_total``;
- for a chosen total ``T``: ``quota[b] = round(ref_frac[b] * T)``;
- ``keep[b] = min(target_count[b], quota[b])``.

``keep[b]`` sites are drawn at random from block ``b`` (seeded). Blocks where
the target has fewer sites than its quota are kept whole (they cannot be
added), so the realised total is ≤ ``T`` and the distribution matches the
reference wherever the target had enough pixels.

What it reads and writes
------------------------
Reads only ``latitude``, ``longitude`` and ``site_id`` from the two
``selected_pixels.nc`` files (schema of ``select_pixels.py``). Writes the same
schema. Torch-free.

The output reuses the existing 0.1° parent map (a superset), so training needs
no pixelset rebuild::

    --selected_pixels <output>  --parent_map selected_pixels01_1.nc

Notes
-----
``--output`` must differ from ``--reference`` and ``--target``: the script only
CREATES a new file and never edits its inputs.

Examples
--------
::

    python -m phenonn.datagen.rebalance_spatial \\
        --reference /path/to/data/pixelset1/all_pixels_selected.nc \\
        --target    /path/to/data/pixelset1/selected_pixels005_1.nc \\
        --block_deg 10 --target_total 1000000 \\
        --output    /path/to/data/pixelset1/selected_pixels005_balanced.nc \\
        --report_csv /path/to/data/pixelset1/rebalance_blocks.csv \\
        --seed 42
"""

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


def _latlon(path):
    """
    Open a selection file and return it with its latitude and longitude arrays.

    Parameters
    ----------
    path : str
        Path to a ``selected_pixels*.nc`` file.

    Returns
    -------
    ds : xarray.Dataset
        The open dataset (caller is responsible for closing it).
    lat : ndarray of shape (n_site,)
        Site latitudes, dtype float64.
    lon : ndarray of shape (n_site,)
        Site longitudes, dtype float64.
    """
    ds = xr.open_dataset(path)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return ds, lat, lon


def _block_ids(lat, lon, deg):
    """
    Assign each site to a flat block identifier on a regular N°×N° grid.

    Parameters
    ----------
    lat : ndarray of shape (n_site,)
        Site latitudes in degrees, in ``[-90, 90]``.
    lon : ndarray of shape (n_site,)
        Site longitudes in degrees, in ``[-180, 180]``.
    deg : float
        Block size in degrees.

    Returns
    -------
    block_id : ndarray of shape (n_site,)
        Flat block identifier, ``ib * ncol + jb``, dtype int64.
    nrow : int
        Number of block rows (``round(180 / deg)``).
    ncol : int
        Number of block columns (``round(360 / deg)``).
    """
    ncol = int(round(360.0 / deg))
    nrow = int(round(180.0 / deg))
    ib = np.clip(np.floor((lat + 90.0) / deg).astype(np.int64), 0, nrow - 1)
    jb = np.clip(np.floor((lon + 180.0) / deg).astype(np.int64), 0, ncol - 1)
    return ib * ncol + jb, nrow, ncol


def main():
    """
    Run the spatial rebalancing.

    Builds the per-block reference distribution, computes a per-block quota at
    the requested total, draws a seeded random subset of sites per block, and
    writes a new selection file with the same schema as the target.

    Raises
    ------
    SystemExit
        If ``--output`` equals ``--reference`` or ``--target`` (the script
        never edits its inputs).
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--reference",
        required=True,
        help="selected_pixels.nc whose spatial distribution is the TARGET "
        "shape (e.g. all-pixels 'analy').",
    )
    ap.add_argument(
        "--target",
        required=True,
        help="selected_pixels.nc to subset (e.g. valid set 'analysis').",
    )
    ap.add_argument("--block_deg", type=float, default=10.0)
    ap.add_argument(
        "--target_total",
        type=int,
        default=0,
        help="Total sites to aim for (0 = keep the target's current "
        "total, i.e. only trim over-represented blocks).",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--report_csv",
        default="",
        help="Optional: write per-block ref%%/target%%/kept table.",
    )
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # Never overwrite an input: the output must be a brand-new file.
    out_abs = os.path.abspath(a.output)
    if out_abs in (os.path.abspath(a.reference), os.path.abspath(a.target)):
        raise SystemExit(
            "--output must differ from --reference/--target "
            "(this script only CREATES a new file, never edits inputs)."
        )

    _, rlat, rlon = _latlon(a.reference)
    tds, tlat, tlon = _latlon(a.target)
    rblk, nrow, ncol = _block_ids(rlat, rlon, a.block_deg)
    tblk, _, _ = _block_ids(tlat, tlon, a.block_deg)
    nblk = nrow * ncol

    r_count = np.bincount(rblk, minlength=nblk).astype(np.int64)
    t_count = np.bincount(tblk, minlength=nblk).astype(np.int64)
    r_tot, t_tot = int(r_count.sum()), int(t_count.sum())
    r_frac = r_count / r_tot

    T = a.target_total if a.target_total > 0 else t_tot
    quota = np.round(r_frac * T).astype(np.int64)
    keep = np.minimum(t_count, quota)
    kept_tot = int(keep.sum())

    print(f"Reference : {a.reference}  ({r_tot:,} sites)")
    print(f"Target    : {a.target}  ({t_tot:,} sites)")
    print(f"Block     : {a.block_deg}°  ({nrow}×{ncol})   aim total T={T:,}")
    print(f"→ kept {kept_tot:,} sites (dropped {t_tot - kept_tot:,})\n")

    # Most over-represented blocks (target share − reference share).
    over = (t_count / max(t_tot, 1)) - r_frac
    order = np.argsort(over)[::-1]
    print(f"{'block(lat,lon)':>16} {'ref%':>7} {'tgt%':>7} {'tgt#':>9} {'keep#':>9}")
    for b in order[:12]:
        if t_count[b] == 0 and r_count[b] == 0:
            continue
        ib, jb = divmod(b, ncol)
        lat0 = -90 + ib * a.block_deg
        lon0 = -180 + jb * a.block_deg
        print(
            f"  ({lat0:+.0f},{lon0:+.0f})".rjust(16)
            + f"{100*r_frac[b]:7.2f}{100*t_count[b]/max(t_tot,1):7.2f}"
            f"{t_count[b]:9,}{keep[b]:9,}"
        )

    # ── Random per-block subsample (seeded) ──
    rng = np.random.RandomState(a.seed)
    keep_mask = np.zeros(tlat.size, dtype=bool)
    for b in np.flatnonzero(keep > 0):
        idx = np.flatnonzero(tblk == b)
        k = int(keep[b])
        if k >= idx.size:
            keep_mask[idx] = True
        else:
            keep_mask[rng.choice(idx, size=k, replace=False)] = True
    keep_idx = np.flatnonzero(keep_mask)
    keep_idx.sort()

    if a.report_csv:
        import csv

        with open(a.report_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "lat0",
                    "lon0",
                    "ref_count",
                    "ref_pct",
                    "target_count",
                    "target_pct",
                    "kept",
                ]
            )
            for b in range(nblk):
                if r_count[b] == 0 and t_count[b] == 0:
                    continue
                ib, jb = divmod(b, ncol)
                w.writerow(
                    [
                        -90 + ib * a.block_deg,
                        -180 + jb * a.block_deg,
                        int(r_count[b]),
                        round(100 * r_frac[b], 4),
                        int(t_count[b]),
                        round(100 * t_count[b] / max(t_tot, 1), 4),
                        int(keep[b]),
                    ]
                )
        print(f"\nPer-block report → {a.report_csv}")

    out = tds.isel(site=keep_idx)
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.output, engine="netcdf4", encoding=enc)
    print(f"\n✓ wrote {keep_idx.size:,} sites → {a.output}")
    print(
        "  Train:  --selected_pixels %s  --parent_map <selected_pixels01_1.nc>"
        % a.output
    )


if __name__ == "__main__":
    main()
