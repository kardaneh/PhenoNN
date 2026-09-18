#!/usr/bin/env python3
"""
rank_years_by_temp.py — rank the years from coldest to hottest.

Computes, per year, the mean of the daily-mean temperature (Tmean) over the whole
ERA5 pixelset (all sites × all days), then writes a JSON ordering the years from
coldest to hottest. Used by the `coldhot` split of run_year_split.py (train on the
coldest years, validate on the hottest).

Torch-free (xarray + numpy). Run on the machine that holds the data.

    python -m phenonn.analysis.year_generalization.rank_years_by_temp \\
        --features_dir /leonardo_work/EUHPC_D36_053/pixelset1/era5_pixelset \\
        --year_start 1992 --year_end 2019 \\
        --out year_temp_rank.json
"""

import argparse
import json
import os

import numpy as np
import xarray as xr

from phenonn.utils.config import FEATURES_FNAME


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_dir", required=True,
                   help="Folder of ERA5_daily_pixelset_{Y}.nc.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end",   type=int, required=True)
    p.add_argument("--var", default="Tmean", help="Temperature variable (default Tmean).")
    p.add_argument("--out", default="year_temp_rank.json")
    args = p.parse_args()

    means = {}
    for y in range(args.year_start, args.year_end + 1):
        path = os.path.join(args.features_dir, FEATURES_FNAME.format(year=y))
        if not os.path.exists(path):
            print(f"  ✗ {y} missing {os.path.basename(path)}")
            continue
        with xr.open_dataset(path) as ds:
            means[y] = float(np.asarray(ds[args.var].mean().values))
        print(f"  ✓ {y}  mean {args.var} = {means[y]:.3f}")

    order = sorted(means, key=lambda yr: means[yr])          # cold → hot
    out = {"years_cold_to_hot": order,
           "mean_tmean": {str(k): v for k, v in means.items()},
           "var": args.var}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\nColdest → hottest:")
    for yr in order:
        print(f"  {yr}  {means[yr]:.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
