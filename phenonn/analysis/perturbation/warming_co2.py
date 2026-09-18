#!/usr/bin/env python3
"""
warming_co2.py — LAI impact of long-term warming + rising CO2.

Ceteris paribus: adds `--delta_t` °C to the temperature channels (Tmin, Tmax,
Tmean) AND `--delta_co2` ppm to the CO2 channel, over the whole year, everything
else held fixed, then reports ΔLAI vs the unperturbed prediction. This is the
combined "climate trend" scenario (temperature and CO2 both increasing in time).

Set `--delta_co2 0` for a pure-warming run, or `--delta_t 0` for a pure-CO2 run.

Usage
-----
    python -m phenonn.analysis.perturbation.warming_co2 \\
        --checkpoint runs_greedy/exp/checkpoints/best_model.pth \\
        --predict_sites val --delta_t 2 --delta_co2 100 \\
        --output_dir runs_perturb
"""

import argparse

from phenonn.analysis.perturbation._perturb import (
    ChannelPerturbation, Perturbation, SEASONS,
)
from phenonn.analysis.perturbation import run_perturbation as R

SCENARIO = "warming_co2"
TEMP_CHANNELS = ["Tmin", "Tmax", "Tmean"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    R.add_common_args(p)
    p.add_argument("--delta_t", type=float, default=2.0,
                   help="°C added to Tmin/Tmax/Tmean (all year). Default +2.")
    p.add_argument("--delta_co2", type=float, default=100.0,
                   help="ppm added to the CO2 channel (all year). Default +100.")
    p.add_argument("--season", default="year", choices=list(SEASONS),
                   help="DOY window of the change (default: year = all days).")
    args = p.parse_args()

    def make(norm_stats):
        chans = [ChannelPerturbation(f, add=args.delta_t) for f in TEMP_CHANNELS]
        if args.delta_co2 != 0.0:
            chans.append(ChannelPerturbation("co2", add=args.delta_co2))
        return Perturbation(chans, norm_stats, season_ranges=SEASONS[args.season])

    R.run(args, make, SCENARIO,
          f"Temperature {args.delta_t:+g}°C + CO2 {args.delta_co2:+g} ppm "
          f"over {args.season}.")


if __name__ == "__main__":
    main()
