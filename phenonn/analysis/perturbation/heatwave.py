#!/usr/bin/env python3
"""
heatwave.py — LAI impact of a summer heatwave (short, intense warming).

Ceteris paribus: adds `--delta_t` °C to the three temperature channels
(Tmin, Tmax, Tmean) over the chosen season (default: summer JJA), everything
else held fixed, then reports ΔLAI vs the unperturbed prediction.

Usage
-----
    python -m phenonn.analysis.perturbation.heatwave \\
        --checkpoint runs_greedy/exp/checkpoints/best_model.pth \\
        --predict_sites val --delta_t 5 --season summer \\
        --output_dir runs_perturb
"""

import argparse

from phenonn.analysis.perturbation._perturb import (
    ChannelPerturbation, Perturbation, SEASONS,
)
from phenonn.analysis.perturbation import run_perturbation as R

SCENARIO = "heatwave"
TEMP_CHANNELS = ["Tmin", "Tmax", "Tmean"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common_args(p)
    p.add_argument("--delta_t", type=float, default=5.0,
                   help="°C added to Tmin/Tmax/Tmean. Default +5.")
    p.add_argument("--season", default="summer", choices=list(SEASONS),
                   help="DOY window of the heatwave (default: summer = JJA).")
    args = p.parse_args()

    def make(norm_stats):
        chans = [ChannelPerturbation(f, add=args.delta_t) for f in TEMP_CHANNELS]
        return Perturbation(chans, norm_stats, season_ranges=SEASONS[args.season])

    R.run(args, make, SCENARIO,
          f"Temperature {args.delta_t:+g}°C (Tmin/Tmax/Tmean) over {args.season}.")


if __name__ == "__main__":
    main()
