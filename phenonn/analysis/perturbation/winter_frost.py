#!/usr/bin/env python3
"""
winter_frost.py — LAI impact of a colder winter (frost).

Ceteris paribus: subtracts `--delta_t` °C from the temperature channels
(Tmin, Tmax, Tmean) over the chosen season (default: winter DJF), everything
else held fixed, then reports ΔLAI vs the unperturbed prediction. Useful to
probe chilling / cold-hardening responses that only show up the following
spring.

`--delta_t` is the COOLING magnitude in °C (applied as −delta_t).

Usage
-----
    python -m phenonn.analysis.perturbation.winter_frost \\
        --checkpoint runs_greedy/exp/checkpoints/best_model.pth \\
        --predict_sites val --delta_t 8 --season winter \\
        --output_dir runs_perturb
"""

import argparse

from phenonn.analysis.perturbation._perturb import (
    ChannelPerturbation, Perturbation, SEASONS,
)
from phenonn.analysis.perturbation import run_perturbation as R

SCENARIO = "winter_frost"
TEMP_CHANNELS = ["Tmin", "Tmax", "Tmean"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common_args(p)
    p.add_argument("--delta_t", type=float, default=8.0,
                   help="Cooling magnitude in °C (applied as −delta_t). Default 8.")
    p.add_argument("--season", default="winter", choices=list(SEASONS),
                   help="DOY window of the frost (default: winter = DJF).")
    args = p.parse_args()

    def make(norm_stats):
        chans = [ChannelPerturbation(f, add=-args.delta_t) for f in TEMP_CHANNELS]
        return Perturbation(chans, norm_stats, season_ranges=SEASONS[args.season])

    R.run(args, make, SCENARIO,
          f"Temperature −{args.delta_t:g}°C (Tmin/Tmax/Tmean) over {args.season}.")


if __name__ == "__main__":
    main()
