#!/usr/bin/env python3
"""
drought.py — LAI impact of a precipitation deficit (drought).

Ceteris paribus: multiplies the precipitation-related channels (tp_sum and its
30-day proxy SMI) by `--precip_factor` (< 1) over the chosen season, everything
else held fixed, then reports ΔLAI vs the unperturbed prediction.

Usage
-----
    python -m phenonn.analysis.perturbation.drought \\
        --checkpoint runs_greedy/exp/checkpoints/best_model.pth \\
        --predict_sites val --precip_factor 0.5 --season growing \\
        --output_dir runs_perturb
"""

import argparse

from phenonn.analysis.perturbation._perturb import (
    ChannelPerturbation, Perturbation, SEASONS,
)
from phenonn.analysis.perturbation import run_perturbation as R

SCENARIO = "drought"
PRECIP_CHANNELS = ["tp_sum", "SMI"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common_args(p)
    p.add_argument("--precip_factor", type=float, default=0.5,
                   help="Multiply precip channels by this (<1 = drier). Default 0.5.")
    p.add_argument("--season", default="growing", choices=list(SEASONS),
                   help="DOY window of the deficit (default: growing = Apr–Oct).")
    args = p.parse_args()

    def make(norm_stats):
        chans = [ChannelPerturbation(f, mul=args.precip_factor)
                 for f in PRECIP_CHANNELS]
        return Perturbation(chans, norm_stats, season_ranges=SEASONS[args.season])

    R.run(args, make, SCENARIO,
          f"Precipitation ×{args.precip_factor:g} (tp_sum, SMI) over {args.season}.")


if __name__ == "__main__":
    main()
