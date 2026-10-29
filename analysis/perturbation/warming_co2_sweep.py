#!/usr/bin/env python3
"""Sweep warming and CO2 perturbations into the F10 response table.

This is the input generator for the F10 table.  It runs the same baseline and
perturbed inference as :mod:`analysis.perturbation.warming_co2`, but over a
grid of temperature offsets and CO2 modes, and writes one row per scenario:

``scenario, delta_t, co2_mode, co2, mean_delta_lai,
median_dsos_10, median_deos_10, median_dlos_10, n_site_years, flag``

The phenology deltas are computed from the 36-dekad LAI curves with the same
10 %% threshold used by the paper table.  A row is flagged ``out-of-range``
when the perturbed CO2 falls outside the training-year CO2 range stored in the
checkpoint.

Usage
-----
    python -m analysis.perturbation.warming_co2_sweep \
        --checkpoint runs/exp/checkpoints/run_best_model.pth.tar \
        --predict_sites test --predict_years 2010-2019 \
        --delta_t_values 0,1,2,3,4 \
        --co2_modes absolute,detrend \
        --output_dir runs/perturbation
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analysis.perturbation import run_perturbation as R
from analysis.perturbation._perturb import (
    SEASONS,
    ChannelPerturbation,
    Perturbation,
)
from phenonn.evaluation import metrics as M
from phenonn.evaluation.leakage import parse_years

TEMP_CHANNELS = ["Tmin", "Tmax", "Tmean"]
DEFAULT_FRACTION = 0.10


def _parse_values(spec: str) -> list:
    """Parse ``"0,1,2"`` into a list of floats."""
    return [float(value) for value in str(spec).split(",") if value.strip()]


def _parse_modes(spec: str) -> list:
    """Parse and validate a comma-separated CO2-mode list."""
    modes = [value.strip() for value in str(spec).split(",") if value.strip()]
    allowed = {"absolute", "doubling", "detrend"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise SystemExit(f"unknown CO2 mode(s): {', '.join(unknown)}")
    return modes


def _co2_label(delta_t: float, mode: str, delta_co2: float, reference: float) -> str:
    """Human-readable CO2 perturbation for the table."""
    if mode == "doubling":
        return f"CO2 ×{2.0 ** (delta_t / 3.0):.3f}"
    if mode == "detrend":
        return f"CO2 fixed at {reference:g} ppm"
    return f"CO2 {delta_co2:+g} ppm"


def _make_perturbation(
    norm_stats,
    delta_t: float,
    mode: str,
    args,
) -> Perturbation:
    """Build the temperature + CO2 perturbation for one sweep point."""
    channels = [ChannelPerturbation(name, add=delta_t) for name in TEMP_CHANNELS]
    if mode == "absolute":
        if args.delta_co2 != 0.0:
            channels.append(ChannelPerturbation("co2", add=args.delta_co2))
    elif mode == "doubling":
        channels.append(ChannelPerturbation("co2", mul=2.0 ** (delta_t / 3.0)))
    elif mode == "detrend":
        channels.append(ChannelPerturbation("co2", set_to=args.co2_reference_ppm))
    else:  # pragma: no cover - validated by _parse_modes
        raise ValueError(f"unknown CO2 mode {mode!r}")
    return Perturbation(
        channels,
        norm_stats,
        season_ranges=SEASONS[args.season],
        hemisphere=getattr(args, "hemisphere", "north"),
    )


def _phenology_deltas(df: pd.DataFrame, fraction: float = DEFAULT_FRACTION) -> dict:
    """Median ΔSOS / ΔEOS / ΔLOS across site-years for one threshold."""
    tag = f"{int(round(100 * fraction))}"
    rows = []
    for _key, group in df.groupby(["site_id", "year"], sort=True):
        group = group.sort_values("doy")
        doy = group["doy"].to_numpy()
        base = M.sos_eos_curve(doy, group["lai_base"].to_numpy(), fractions=(fraction,))
        pert = M.sos_eos_curve(doy, group["lai_pert"].to_numpy(), fractions=(fraction,))
        rows.append(
            {
                f"dsos_{tag}": pert[f"sos_{tag}"] - base[f"sos_{tag}"],
                f"deos_{tag}": pert[f"eos_{tag}"] - base[f"eos_{tag}"],
                f"dlos_{tag}": pert[f"los_{tag}"] - base[f"los_{tag}"],
            }
        )
    if not rows:
        return {
            f"median_dsos_{tag}": float("nan"),
            f"median_deos_{tag}": float("nan"),
            f"median_dlos_{tag}": float("nan"),
        }
    table = pd.DataFrame(rows)
    return {
        f"median_dsos_{tag}": float(np.nanmedian(table[f"dsos_{tag}"])),
        f"median_deos_{tag}": float(np.nanmedian(table[f"deos_{tag}"])),
        f"median_dlos_{tag}": float(np.nanmedian(table[f"dlos_{tag}"])),
    }


def _co2_value(lut: dict, year: int):
    """Look up a CO2 value with int/str key tolerance."""
    for key in (year, int(year), str(int(year))):
        if key in lut:
            return float(lut[key])
    return None


def _out_of_range(
    ctx,
    df: pd.DataFrame,
    delta_t: float,
    mode: str,
    args,
) -> str:
    """Return ``"out-of-range"`` when perturbed CO2 leaves the training span."""
    lut = ctx.get("co2_lut") or {}
    train_years = parse_years(ctx.get("train_args", {}).get("train_years", ""))
    values = [
        _co2_value(lut, int(year))
        for year in train_years
        if _co2_value(lut, int(year)) is not None
    ]
    if not values:
        return ""
    low, high = min(values), max(values)
    factor = 2.0 ** (delta_t / 3.0)
    for year in sorted(df["year"].unique()):
        base = _co2_value(lut, int(year))
        if base is None:
            continue
        if mode == "absolute":
            perturbed = base + args.delta_co2
        elif mode == "doubling":
            perturbed = base * factor
        else:
            perturbed = args.co2_reference_ppm
        if perturbed < low - 1e-9 or perturbed > high + 1e-9:
            return "out-of-range"
    return ""


def _summarise(
    ctx,
    df: pd.DataFrame,
    delta_t: float,
    mode: str,
    args,
) -> dict:
    """Build one row of the F10 table from a perturbation table."""
    row = {
        "scenario": f"{mode}_dt{delta_t:+g}",
        "delta_t": float(delta_t),
        "co2_mode": mode,
        "co2": _co2_label(delta_t, mode, args.delta_co2, args.co2_reference_ppm),
        "mean_delta_lai": float(df["delta"].mean()),
        "n_site_years": int(df.groupby(["site_id", "year"]).ngroups),
        "flag": _out_of_range(ctx, df, delta_t, mode, args),
    }
    row.update(_phenology_deltas(df, args.sos_fraction))
    return row


def main() -> None:
    """Run the warming + CO2 sweep from command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    R.add_common_args(parser)
    parser.add_argument(
        "--delta_t_values",
        default="0,1,2,3,4",
        help="Comma-separated warming offsets in °C. Default: 0,1,2,3,4.",
    )
    parser.add_argument(
        "--co2_modes",
        default="absolute,detrend",
        help="Comma-separated CO2 modes: absolute, doubling, detrend.",
    )
    parser.add_argument(
        "--delta_co2",
        type=float,
        default=100.0,
        help="ppm added in 'absolute' mode. Default +100.",
    )
    parser.add_argument(
        "--co2_reference_ppm",
        type=float,
        default=350.0,
        help="Constant CO2 value for 'detrend' mode. Default 350 ppm.",
    )
    parser.add_argument(
        "--season",
        default="year",
        choices=list(SEASONS),
        help="DOY window of the change (default: year).",
    )
    parser.add_argument(
        "--sos_fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="LAI threshold fraction for SOS/EOS (default 0.10).",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Aggregate CSV path (default: <output_dir>/warming_co2.csv).",
    )
    args = parser.parse_args()

    delta_values = _parse_values(args.delta_t_values)
    modes = _parse_modes(args.co2_modes)
    if not delta_values or not modes:
        raise SystemExit("provide at least one delta_t and one CO2 mode")

    ctx = R.load_model_and_data(args)
    rows = []
    for mode in modes:
        for delta_t in delta_values:
            perturbation = _make_perturbation(ctx.norm_stats, delta_t, mode, args)
            scenario = f"{mode}_dt{delta_t:+g}"
            df = R.run_from_context(
                ctx,
                args,
                perturbation,
                scenario,
                write_outputs=False,
            )
            rows.append(_summarise(ctx, df, delta_t, mode, args))

    out_csv = args.output_csv or os.path.join(args.output_dir, "warming_co2.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    print(f"\nWarming + CO2 table -> {out_csv}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
