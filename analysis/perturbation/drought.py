#!/usr/bin/env python3
"""
drought.py: LAI impact of a precipitation deficit (drought).

Ceteris paribus: multiplies the precipitation-related channels (tp_sum and its
30-day proxy SMI) by `--precip_factor` (< 1) over the chosen season, everything
else held fixed, then reports Delta LAI vs the unperturbed prediction.

Usage
-----
    python -m analysis.perturbation.drought \\
        --checkpoint runs/exp/checkpoints/run_best_model.pth.tar \\
        --predict_sites val --precip_factor 0.5 --season growing \\
        --output_dir runs/perturbation
"""

import argparse

from analysis.perturbation import run_perturbation as R
from analysis.perturbation._perturb import (
    SEASONS,
    ChannelPerturbation,
    Perturbation,
)

SCENARIO = "drought"
PRECIP_CHANNELS = ["tp_sum", "SMI"]


def main():
    """Run the drought scenario from command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    R.add_common_args(p)
    p.add_argument(
        "--precip_factor",
        type=float,
        default=0.5,
        help="Multiply precip channels by this (<1 = drier). Default 0.5.",
    )
    p.add_argument(
        "--season",
        default="growing",
        choices=list(SEASONS),
        help="DOY window of the deficit (default: growing = Apr-Oct).",
    )
    args = p.parse_args()

    def make(norm_stats):
        """Build the precipitation-deficit perturbation."""
        chans = [
            ChannelPerturbation(f, mul=args.precip_factor) for f in PRECIP_CHANNELS
        ]
        return Perturbation(chans, norm_stats, season_ranges=SEASONS[args.season],
            hemisphere=getattr(args, "hemisphere", "north"),
        )

    R.run(
        args,
        make,
        SCENARIO,
        f"Precipitation x{args.precip_factor:g} (tp_sum, SMI) over {args.season}.",
    )


if __name__ == "__main__":
    main()
