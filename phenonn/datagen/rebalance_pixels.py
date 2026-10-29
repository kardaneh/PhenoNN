# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Down-sample over-represented PFTs in a ``selected_pixels.nc``.

Motivation
----------
The 0.05° validity filter keeps disproportionately many tropical broadleaf
evergreen pixels (stable, always-green, clean LAI): in the valid set they are
~34 % of the pure cells, versus ~14 % over ALL land pixels. This script removes
a random subset of the sites whose DOMINANT PFT is one of ``--reduce_pft``
(default: tropical BL evergreen) to bring the pool back toward the global
proportions, targeting either a total site count or a target dominant share.

What it reads and writes
------------------------
It reads the PFT fractions per site from ONE PFT pixelset year (PFT is
~static), computes the argmax = dominant PFT per site, prints the before/after
dominant histogram, then writes a NEW ``selected_pixels.nc`` (same schema:
site dim with lat_idx / lon_idx / latitude / longitude / site_id) holding only
the kept sites.

The existing 0.1° parent map (``selected_pixels01_1.nc``) is a SUPERSET of
these sites, so training can reuse it unchanged::

    --selected_pixels <this output>  --parent_map selected_pixels01_1.nc

No ERA5/LAI/PFT pixelset rebuild is needed — the reduced pool simply loads
fewer sites into RAM (and a more balanced mix).

Notes
-----
Torch-free (xarray + numpy + pandas). PFT read uses ``engine="netcdf4"``,
``decode_times=False`` to stay on one libhdf5.

Examples
--------
::

    python -m phenonn.datagen.rebalance_pixels \\
        --selected_pixels /path/to/data/pixelset1/selected_pixels005_1.nc \\
        --pft_dir         /path/to/data/pixelset1/PFT_pixelset \\
        --pft_year 2010 \\
        --reduce_pft 2 \\
        --target_total 1000000 \\
        --output          /path/to/data/pixelset1/selected_pixels005_balanced.nc \\
        --seed 42
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from phenonn.utils.config import PFT_FNAME

# ORCHIDEE 15-PFT labels (copied to stay torch-free — mirrors utils.config.PFT_NAMES).
PFT_NAMES = [
    "bare soil",
    "trop BL evergreen",
    "trop BL raingreen",
    "temp NL evergreen",
    "temp BL evergreen",
    "temp BL summergreen",
    "bor NL evergreen",
    "bor BL summergreen",
    "bor NL summergreen",
    "temp C3 grass",
    "C4 grass",
    "C3 agriculture",
    "C4 agriculture",
    "trop C3 grass",
    "bor C3 grass",
]
N_PFT = 15


def _open_pft_frac(path):
    """
    Open a PFT pixelset and return its fraction array and site identifiers.

    Parameters
    ----------
    path : str
        Path to a ``PFTmap_{Y}.nc`` pixelset.

    Returns
    -------
    pft_frac : ndarray of shape (n_pft, n_site)
        PFT fractions, dtype float32.
    site_id : ndarray of shape (n_site,)
        Site identifiers, dtype ``object`` (strings).
    """
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
    da = da.transpose("pft", "site")
    return da.values.astype(np.float32), np.asarray(ds["site_id"].values).astype(str)


def _hist(dominant, mask=None):
    """
    Count sites per dominant PFT.

    Parameters
    ----------
    dominant : ndarray of shape (n_site,)
        Dominant PFT per site, values 1..N_PFT (0 = missing).
    mask : ndarray of shape (n_site,), optional
        Boolean mask restricting which sites are counted. Default is ``None``
        (count every site).

    Returns
    -------
    counts : ndarray of shape (N_PFT,)
        Number of sites per dominant PFT, dtype int64, index 0 = PFT 1.
    """
    d = dominant if mask is None else dominant[mask]
    return np.array([int((d == k).sum()) for k in range(1, N_PFT + 1)])


def _print_hist(title, counts):
    """
    Print a dominant-PFT histogram.

    Parameters
    ----------
    title : str
        Header line to print before the table.
    counts : ndarray of shape (N_PFT,)
        Number of sites per PFT (from :func:`_hist`).
    """
    tot = int(counts.sum())
    print(f"\n{title}  (total {tot:,})")
    for k, c in enumerate(counts, 1):
        pct = 100.0 * c / tot if tot else 0.0
        print(f"  {k:2d}. {PFT_NAMES[k-1]:20s} {c:>10,}  {pct:5.2f}%")


def main():
    """
    Run the PFT-dominance rebalancing.

    Loads the selection, joins the PFT fractions by site ID, computes the
    dominant PFT per site, down-samples the reducible sites to the chosen
    target, and writes a new selection file with the same schema as the input.

    The number of reducible sites to keep is set by exactly one of
    ``--target_total`` (keep roughly this many sites in total) or
    ``--target_share`` (down-sample so the reducible PFTs combined share is
    this fraction). Non-reducible sites are always kept. The reducible subset
    is drawn with a seeded RNG (``--seed``, default 42) so the draw is
    reproducible.

    Raises
    ------
    FileNotFoundError
        If the requested PFT year file does not exist in ``--pft_dir``.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selected_pixels", required=True)
    ap.add_argument(
        "--pft_dir", required=True, help="Folder of PFTmap_{Y}.nc (pixelset)."
    )
    ap.add_argument("--pft_year", type=int, required=True)
    ap.add_argument(
        "--reduce_pft",
        default="2",
        help="Comma-separated PFT numbers (1..15) to down-sample by "
        "dominance (default 2 = tropical BL evergreen).",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--target_total",
        type=int,
        help="Keep ~this many sites total by dropping reduce_pft sites.",
    )
    g.add_argument(
        "--target_share",
        type=float,
        help="Down-sample reduce_pft so their COMBINED dominant share "
        "equals this fraction (e.g. 0.15).",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    reduce_pft = {int(x) for x in a.reduce_pft.split(",")}

    # ── Load the selection and the PFT fractions, align by site_id ──
    sp = xr.open_dataset(a.selected_pixels).load()
    sp_ids = np.asarray(sp["site_id"].values).astype(str)
    n_sp = sp_ids.size
    print(f"Selection    : {a.selected_pixels}  ({n_sp:,} sites)")

    ppath = os.path.join(a.pft_dir, PFT_FNAME.format(year=a.pft_year))
    if not os.path.exists(ppath):
        raise FileNotFoundError(ppath)
    frac_all, pft_ids = _open_pft_frac(ppath)
    print(f"PFT year     : {ppath}  ({pft_ids.size:,} sites)")

    rows = pd.Index(pft_ids).get_indexer(pd.Index(sp_ids))  # sp order → PFT rows
    if (rows < 0).any():
        n_miss = int((rows < 0).sum())
        print(
            f"  ! {n_miss:,} selection sites missing from the PFT file "
            f"— kept as-is (treated non-reducible)"
        )
    frac = np.full((N_PFT, n_sp), np.nan, np.float32)
    ok = rows >= 0
    frac[:, ok] = frac_all[:, rows[ok]]

    # Dominant PFT per site (1..15); NaN-frac sites (missing) → dominant 0 (kept).
    # Fill NaN with -1 so argmax is safe on all-NaN columns (overridden by `ok`).
    dom = np.argmax(np.where(np.isnan(frac), -1.0, frac), axis=0) + 1
    dominant = np.where(ok, dom, 0).astype(np.int64)

    counts_before = _hist(dominant)
    _print_hist("Dominant PFT BEFORE", counts_before)

    # ── Decide how many reduce_pft sites to keep ──
    reducible = np.isin(dominant, list(reduce_pft))
    n_red = int(reducible.sum())
    n_other = n_sp - n_red
    print(
        f"\nReducible (dominant in {sorted(reduce_pft)}): {n_red:,} sites; "
        f"non-reducible: {n_other:,}"
    )

    if a.target_total is not None:
        keep_red = a.target_total - n_other
        if keep_red >= n_red:
            print(
                f"  target_total {a.target_total:,} ≥ non-reducible+all-reducible "
                f"({n_sp:,}) — nothing to drop."
            )
            keep_red = n_red
        keep_red = max(0, keep_red)
    else:  # target_share
        # keep_red / (keep_red + n_other) = share  →  keep_red = share/(1-share)*n_other
        s = a.target_share
        keep_red = int(round(s / (1.0 - s) * n_other)) if s < 1.0 else n_red
        keep_red = min(keep_red, n_red)
    print(
        f"Keeping {keep_red:,} of {n_red:,} reducible sites "
        f"(dropping {n_red - keep_red:,})."
    )

    # ── Random subset of the reducible sites (seeded) ──
    rng = np.random.RandomState(a.seed)
    red_idx = np.flatnonzero(reducible)
    keep_red_idx = (
        rng.choice(red_idx, size=keep_red, replace=False)
        if keep_red < n_red
        else red_idx
    )
    keep_mask = ~reducible
    keep_mask[keep_red_idx] = True
    keep_idx = np.flatnonzero(keep_mask)
    keep_idx.sort()

    _print_hist("Dominant PFT AFTER", _hist(dominant, keep_mask))

    # ── Write the reduced selection (same schema) ──
    out = sp.isel(site=keep_idx)
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(a.output, engine="netcdf4", encoding=enc)
    print(f"\n✓ wrote {keep_idx.size:,} sites → {a.output}")
    print(
        "  Train with:  --selected_pixels %s  --parent_map <selected_pixels01_1.nc>"
        % a.output
    )


if __name__ == "__main__":
    main()
