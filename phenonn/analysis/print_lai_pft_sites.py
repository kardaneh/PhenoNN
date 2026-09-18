#!/usr/bin/env python3
"""
print_lai_pft_sites.py
======================

Diagnostic PRINT-ONLY (no plot): for a few pixels, dump the 36 dekadal LAI
values and the 15 PFT fractions for every year.

Reads the pixelset cubes produced by build_pixelset_targets.py:
  target_dir/LAI_dekadal_{Y}.nc   LAI(dekad=36, site)
  pft_dir/PFTmap_{Y}.nc           pft_frac(pft=15, site)

Site alignment reuses the exact helpers the Dataset uses
(`_read_site_vectors` / `_open_pft_array`), so an unknown site_id prints as
NaN (never silently mismatched).

Usage
-----
    python -m plot_diagnositcs.print_lai_pft_sites \\
        --target_dir /data/.../targets \\
        --pft_dir    /data/.../pft \\
        --years      2000-2018 \\
        --sites      pix_0123_04567,pix_0088_01200,pix_0140_00999
"""

import argparse
import datetime
import os

import numpy as np
import xarray as xr

from phenonn.utils.config import (
    DEKAD_DAYS, N_DEKAD_YEAR, N_PFT, PFT_FNAME, PFT_NAMES, TARGETS_FNAME,
)
from phenonn.data.lai_dataset import _open_pft_array, _read_site_vectors


# Labels for the 36 dekads (days 5/15/25 of each month), e.g. "Jan05".
DEKAD_LABELS = [f"{datetime.date(2001, m, d):%b}{d:02d}"
                for m in range(1, 13) for d in DEKAD_DAYS]


def parse_years(s: str):
    """'2000-2018' or '2000,2005,2010' → list of ints."""
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target_dir", required=True,
                   help="Folder of LAI_dekadal_{Y}.nc.")
    p.add_argument("--pft_dir", required=True,
                   help="Folder of PFTmap_{Y}.nc.")
    p.add_argument("--years", required=True,
                   help="'2000-2018' or '2000,2005,2010'.")
    p.add_argument("--sites", required=True,
                   help="Comma-separated site_ids (pix_LLLL_OOOOO).")
    p.add_argument("--pft_min_frac", type=float, default=0.0,
                   help="Only list PFTs with fraction >= this (default 0.0 = "
                        "all non-zero).")
    return p.parse_args()


def main():
    args = parse_args()
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    years = parse_years(args.years)

    # ── Preload once: one file open per year, all requested sites at a time ──
    lai_by_year = {}   # year -> (n_site, 36)
    pft_by_year = {}   # year -> (n_site, 15)
    for y in years:
        tpath = os.path.join(args.target_dir, TARGETS_FNAME.format(year=y))
        if os.path.exists(tpath):
            with xr.open_dataset(tpath) as dt:
                lai_by_year[y] = _read_site_vectors(dt["LAI"], sites, "dekad")
        ppath = os.path.join(args.pft_dir, PFT_FNAME.format(year=y))
        if os.path.exists(ppath):
            pft_by_year[y] = _read_site_vectors(
                _open_pft_array(ppath), sites, "pft")

    # ── Print, grouped by site then year ──
    for si, site in enumerate(sites):
        print("=" * 74)
        print(f"SITE {site}")
        print("=" * 74)
        for y in years:
            lai = lai_by_year.get(y)
            pft = pft_by_year.get(y)
            if lai is None and pft is None:
                print(f"\n  ── {y} ── (no LAI and no PFT file)")
                continue

            # PFT fractions
            if pft is not None:
                fr = pft[si]
                items = [(PFT_NAMES[k], fr[k]) for k in range(N_PFT)
                         if np.isfinite(fr[k]) and fr[k] >= args.pft_min_frac
                         and fr[k] > 0.0]
                pft_str = ("  ".join(f"{name}={val:.3f}" for name, val in items)
                           or "(all zero / NaN)")
            else:
                pft_str = "(no PFT file)"

            # LAI vector + summary
            if lai is not None:
                v = lai[si]
                fin = v[np.isfinite(v)]
                summ = (f"n={fin.size}/{N_DEKAD_YEAR}  mean={fin.mean():.3f}  "
                        f"min={fin.min():.3f}  max={fin.max():.3f}"
                        if fin.size else "all-NaN")
            else:
                v, summ = None, "(no LAI file)"

            print(f"\n  ── {y} ──")
            print(f"  PFT: {pft_str}")
            print(f"  LAI: {summ}")
            if v is not None:
                for m in range(12):
                    seg = v[m * 3:(m + 1) * 3]
                    labs = DEKAD_LABELS[m * 3:(m + 1) * 3]
                    cells = "  ".join(f"{lab}={val:6.3f}"
                                      for lab, val in zip(labs, seg))
                    print(f"       {cells}")
        print()


if __name__ == "__main__":
    main()
