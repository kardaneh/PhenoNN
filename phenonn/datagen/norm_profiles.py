# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Named normalisation profiles for the N1/N2/N3 experiments.

A profile lists only the channels that differ from the legacy rule; the stats
builder expands it into a full ``_transforms`` block. Keeping the profiles here
means the experiments are data, not branches in the training code.

- ``n1``: remove the harmful ``log1p`` on radiation, bound ``daylength`` with
  min-max, and keep the PFT fractions raw.
- ``n2``: ``n1`` plus ``sqrt`` on ``VPD_mean`` (audit: skew 1.86 → 0.62
  against 0.91 with ``log1p``).
- ``n3``: all 28 inputs on ``minmax`` (fixed physical bounds for temperature,
  daylength, CO2 and the PFT fractions; p1-p99 for the fluxes).
"""

from __future__ import annotations

from typing import Dict

from phenonn.utils.config import DYNAMIC_FEATURES, N_PFT

N1: Dict[str, object] = {
    "ssrd_sum": "zscore",
    "strd_sum": "zscore",
    "daylength": {"kind": "minmax", "bounds": (0.0, 24.0)},
    "PFT": "none",
}

N2: Dict[str, object] = {**N1, "VPD_mean": "sqrt_zscore"}

# Global min-max on all 28 inputs, in the style of DeepPhenoMem: one family of
# transforms for every channel, so PFT/co2 are not left on a different scale.
# Fixed physical bounds where they exist; p1-p99 (clipped) for the fluxes.
N3: Dict[str, object] = {
    "ALL_DYNAMIC": {"kind": "minmax", "percentiles": (1, 99)},
    "PFT": {"kind": "minmax", "bounds": (0.0, 1.0)},
    "Tmin": {"kind": "minmax", "bounds": (-50.0, 50.0)},
    "Tmax": {"kind": "minmax", "bounds": (-50.0, 50.0)},
    "Tmean": {"kind": "minmax", "bounds": (-50.0, 50.0)},
    "daylength": {"kind": "minmax", "bounds": (0.0, 24.0)},
    "co2": {"kind": "minmax", "bounds": (300.0, 450.0)},
}

# Precipitation-only variants: everything else stays legacy.
P_SQRT: Dict[str, object] = {"tp_sum": {"kind": "sqrt_zscore"}}
P_CBRT: Dict[str, object] = {"tp_sum": {"kind": "cbrt_zscore"}}
P_ASINH: Dict[str, object] = {"tp_sum": {"kind": "asinh_zscore", "scale": 1.0}}

PROFILES: Dict[str, Dict[str, object]] = {
    "legacy": {},
    "n1": N1,
    "n2": N2,
    "n3": N3,
    "p_sqrt": P_SQRT,
    "p_cbrt": P_CBRT,
    "p_asinh": P_ASINH,
}


def resolve_profile(name: str) -> Dict[str, object]:
    """Expand a profile shorthand into per-channel overrides.

    Parameters
    ----------
    name : str
        One of ``PROFILES``.

    Returns
    -------
    dict
        ``{channel: kind or spec}``; ``PFT`` expands to ``pft1_frac``...
        ``pft15_frac`` and ``ALL_DYNAMIC`` to every dynamic channel.

    Raises
    ------
    ValueError
        If the profile name is unknown.
    """
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    out: Dict[str, object] = {}
    for key, value in PROFILES[name].items():
        if key == "PFT":
            for k in range(1, N_PFT + 1):
                out[f"pft{k}_frac"] = value
        elif key == "ALL_DYNAMIC":
            for feature in DYNAMIC_FEATURES:
                out[feature] = value
        else:
            out[key] = value
    return out
