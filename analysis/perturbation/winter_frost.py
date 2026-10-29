#!/usr/bin/env python3
"""
winter_frost.py: LAI impact of a colder winter (frost).

Ceteris paribus: subtracts `--delta_t` C from the temperature channels
(Tmin, Tmax, Tmean) over the chosen season (default: winter DJF), everything
else held fixed, then reports Delta LAI vs the unperturbed prediction. Useful to
probe chilling / cold-hardening responses that only show up the following
spring.

`--delta_t` is the COOLING magnitude in C (applied as -delta_t).

Usage
-----
    python -m analysis.perturbation.winter_frost \\
        --checkpoint runs/exp/checkpoints/run_best_model.pth.tar \\
        --predict_sites val --delta_t 8 --season winter \\
        --output_dir runs/perturbation
"""

import argparse

from analysis.perturbation import run_perturbation as R
from analysis.perturbation._perturb import (
    SEASONS,
    ChannelPerturbation,
    Perturbation,
)

SCENARIO = "winter_frost"
TEMP_CHANNELS = ["Tmin", "Tmax", "Tmean"]


def main():
    """Run the winter-frost scenario from command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    R.add_common_args(p)
    p.add_argument(
        "--delta_t",
        type=float,
        default=8.0,
        help="Cooling magnitude in degrees Celsius. Default 8.",
    )
    p.add_argument(
        "--season",
        default="winter",
        choices=list(SEASONS),
        help="DOY window of the frost (default: winter = DJF).",
    )
    args = p.parse_args()

    def make(norm_stats):
        """Build the seasonal cooling perturbation."""
        chans = [ChannelPerturbation(f, add=-args.delta_t) for f in TEMP_CHANNELS]
        return Perturbation(
            chans,
            norm_stats,
            season_ranges=SEASONS[args.season],
            hemisphere=getattr(args, "hemisphere", "north"),
        )

    R.run(
        args,
        make,
        SCENARIO,
        f"Temperature -{args.delta_t:g} C (Tmin/Tmax/Tmean) over {args.season}.",
    )


if __name__ == "__main__":
    main()
