#!/usr/bin/env python3
"""
warming_co2.py: LAI impact of long-term warming and rising CO2.

Ceteris paribus: adds ``--delta_t`` °C to the temperature channels
(Tmin, Tmax, Tmean) and changes the CO2 channel over the whole year, holding
everything else fixed, then reports ΔLAI against the unperturbed prediction.

Three CO2 modes
---------------
``absolute``
    Add ``--delta_co2`` ppm to the CO2 channel (the original behaviour).
``doubling``
    Apply the transient-climate relation ``C/C0 = 2^(ΔT/3)`` used in the
    paper protocol, i.e. multiply the physical CO2 value by
    ``2^(ΔT/3)``.  This is the combined warming + CO2 scenario.
``detrend``
    Replace the CO2 channel with a constant ``--co2_reference_ppm``.  This is
    the detrended-CO2 control: it removes the historical CO2 trend while
    keeping the warming perturbation.

Set ``--delta_t 0`` for a pure-CO2 run, or ``--co2_mode absolute
--delta_co2 0`` for a pure-warming run.

Usage
-----
    python -m analysis.perturbation.warming_co2 \\
        --checkpoint runs/exp/checkpoints/run_best_model.pth.tar \\
        --predict_sites val --delta_t 2 --co2_mode doubling \\
        --output_dir runs/perturbation
"""

import argparse

from analysis.perturbation import run_perturbation as R
from analysis.perturbation._perturb import (
    SEASONS,
    ChannelPerturbation,
    Perturbation,
)

SCENARIO = "warming_co2"
TEMP_CHANNELS = ["Tmin", "Tmax", "Tmean"]


def main():
    """Run the warming-plus-CO2 scenario from command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    R.add_common_args(p)
    p.add_argument(
        "--delta_t",
        type=float,
        default=2.0,
        help="Degrees Celsius added to Tmin/Tmax/Tmean. Default +2.",
    )
    p.add_argument(
        "--delta_co2",
        type=float,
        default=100.0,
        help="ppm added to the CO2 channel in 'absolute' mode. Default +100.",
    )
    p.add_argument(
        "--co2_mode",
        default="absolute",
        choices=["absolute", "doubling", "detrend"],
        help="How the CO2 channel is changed (default: absolute).",
    )
    p.add_argument(
        "--co2_reference_ppm",
        type=float,
        default=350.0,
        help="Constant CO2 value for 'detrend' mode (default 350 ppm).",
    )
    p.add_argument(
        "--co2_factor",
        type=float,
        default=None,
        help="Explicit multiplicative CO2 factor; overrides --co2_mode.",
    )
    p.add_argument(
        "--season",
        default="year",
        choices=list(SEASONS),
        help="DOY window of the change (default: year = all days).",
    )
    args = p.parse_args()

    def make(norm_stats):
        """Build the combined temperature and CO2 perturbation."""
        chans = [ChannelPerturbation(f, add=args.delta_t) for f in TEMP_CHANNELS]
        if args.co2_factor is not None:
            chans.append(ChannelPerturbation("co2", mul=args.co2_factor))
        elif args.co2_mode == "absolute":
            if args.delta_co2 != 0.0:
                chans.append(ChannelPerturbation("co2", add=args.delta_co2))
        elif args.co2_mode == "doubling":
            # Transient climate response: C/C0 = 2^(ΔT/3).
            factor = 2.0 ** (args.delta_t / 3.0)
            chans.append(ChannelPerturbation("co2", mul=factor))
        elif args.co2_mode == "detrend":
            chans.append(ChannelPerturbation("co2", set_to=args.co2_reference_ppm))
        return Perturbation(
            chans,
            norm_stats,
            season_ranges=SEASONS[args.season],
            hemisphere=getattr(args, "hemisphere", "north"),
        )

    if args.co2_factor is not None:
        co2_txt = f"CO2 ×{args.co2_factor:g}"
    elif args.co2_mode == "doubling":
        co2_txt = f"CO2 ×{2.0 ** (args.delta_t / 3.0):.3f} (2^(ΔT/3))"
    elif args.co2_mode == "detrend":
        co2_txt = f"CO2 fixed at {args.co2_reference_ppm:g} ppm (detrended)"
    else:
        co2_txt = f"CO2 {args.delta_co2:+g} ppm"

    R.run(
        args,
        make,
        SCENARIO,
        f"Temperature {args.delta_t:+g} C + {co2_txt} over {args.season}.",
    )


if __name__ == "__main__":
    main()
