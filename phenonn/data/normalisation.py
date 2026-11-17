# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Per-channel input normalisation specs.

A stats file keeps the historical ``{feature: {"mean", "std"}}`` entries and
may add a ``_transforms`` block that selects the map applied to each channel::

    "_transforms": {
        "ssrd_sum":  {"kind": "zscore"},
        "daylength": {"kind": "minmax", "lo": 0.0, "hi": 24.0, "clip": true},
        "pft1_frac": {"kind": "none"}
    }

Channels absent from the block keep the default rule: ``log1p`` then z-score
for ``LOG_TRANSFORM_FEATURES``, z-score otherwise. A stats file without
``_transforms`` therefore reproduces the previous behaviour exactly.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from phenonn.utils.config import LOG_TRANSFORM_FEATURES

KINDS = (
    "zscore",
    "log1p_zscore",
    "sqrt_zscore",
    "cbrt_zscore",
    "asinh_zscore",
    "minmax",
    "none",
)

_ZSCORE_FAMILY = (
    "zscore",
    "log1p_zscore",
    "sqrt_zscore",
    "cbrt_zscore",
    "asinh_zscore",
)

# Channels that can be negative: log/sqrt transforms are invalid on them.
SIGNED = {
    "Tmin",
    "Tmax",
    "Tmean",
    "Rn_tot",
    "botta_forcing",
    "t_rising",
    "t_falling",
}

_EPS = 1e-8


def default_kind(name: str) -> str:
    """Return the default transform kind of a channel."""
    return "log1p_zscore" if name in LOG_TRANSFORM_FEATURES else "zscore"


def get_spec(norm_stats, name: str) -> Dict[str, object]:
    """Return the effective spec of one channel.

    Parameters
    ----------
    norm_stats : mapping or None
        Parsed stats file, optionally carrying ``_transforms``.
    name : str
        Channel name (``Tmin``, ``co2``, ``pft1_frac``, ``LAI``, ...).

    Returns
    -------
    dict
        ``{"kind", "name"}`` plus ``mean``/``std`` (z-score family) or
        ``lo``/``hi``/``clip`` (min-max).
    """
    if not isinstance(norm_stats, dict):
        return {"kind": "none", "name": name}

    base = norm_stats.get(name)
    base = base if isinstance(base, dict) else {}
    overrides = norm_stats.get("_transforms") or {}
    override = overrides.get(name, {}) if isinstance(overrides, dict) else {}
    if isinstance(override, str):
        override = {"kind": override}

    if override.get("kind"):
        kind = override["kind"]
    elif base:
        kind = default_kind(name)
    else:
        kind = "none"

    spec: Dict[str, object] = {"kind": kind, "name": name}
    for key in ("mean", "std", "lo", "hi", "clip", "scale"):
        if key in override:
            spec[key] = override[key]
        elif key in base:
            spec[key] = base[key]

    if kind in _ZSCORE_FAMILY:
        spec["mean"] = float(spec.get("mean", 0.0))
        std = abs(float(spec.get("std", 1.0)))
        spec["std"] = std if std > _EPS else 1.0
        if kind == "asinh_zscore":
            spec["scale"] = float(spec.get("scale", 1.0))
    elif kind == "minmax":
        spec["lo"] = float(spec.get("lo", 0.0))
        spec["hi"] = float(spec.get("hi", 1.0))
        spec["clip"] = bool(spec.get("clip", True))
    return spec


def forward_numpy(x: np.ndarray, spec: Mapping[str, object]) -> np.ndarray:
    """Apply a spec to a numpy array (same dtype behaviour as before)."""
    kind = spec["kind"]
    if kind == "none":
        return x
    if kind == "zscore":
        return (x - spec["mean"]) / spec["std"]
    if kind == "log1p_zscore":
        return (np.log1p(np.clip(x, 0.0, None)) - spec["mean"]) / spec["std"]
    if kind == "sqrt_zscore":
        return (np.sqrt(np.clip(x, 0.0, None)) - spec["mean"]) / spec["std"]
    if kind == "cbrt_zscore":
        return (np.cbrt(np.clip(x, 0.0, None)) - spec["mean"]) / spec["std"]
    if kind == "asinh_zscore":
        return (np.arcsinh(x / spec["scale"]) - spec["mean"]) / spec["std"]
    if kind == "minmax":
        values = np.clip(x, spec["lo"], spec["hi"]) if spec["clip"] else x
        span = float(spec["hi"]) - float(spec["lo"])
        if span <= _EPS:
            return np.zeros_like(values)
        return (values - spec["lo"]) / span
    raise ValueError(f"unknown transform kind {kind!r}")


def inverse_numpy(x: np.ndarray, spec: Mapping[str, object]) -> np.ndarray:
    """Invert :func:`forward_numpy`."""
    kind = spec["kind"]
    if kind == "none":
        return x
    if kind == "minmax":
        return x * (float(spec["hi"]) - float(spec["lo"])) + float(spec["lo"])
    values = x * spec["std"] + spec["mean"]
    if kind == "zscore":
        return values
    if kind == "log1p_zscore":
        return np.expm1(values)
    if kind == "sqrt_zscore":
        return np.square(values)
    if kind == "cbrt_zscore":
        return np.power(values, 3)
    if kind == "asinh_zscore":
        return np.sinh(values) * spec["scale"]
    raise ValueError(f"unknown transform kind {kind!r}")


def inverse_torch(x, spec: Mapping[str, object]):
    """Invert a spec with torch ops (used by the PFT mixing wrapper)."""
    import torch

    kind = spec["kind"]
    if kind == "none":
        return x
    if kind == "minmax":
        return x * (float(spec["hi"]) - float(spec["lo"])) + float(spec["lo"])
    values = x * float(spec["std"]) + float(spec["mean"])
    if kind == "zscore":
        return values
    if kind == "log1p_zscore":
        return torch.expm1(values)
    if kind == "sqrt_zscore":
        return torch.square(values)
    if kind == "cbrt_zscore":
        return torch.pow(values, 3)
    if kind == "asinh_zscore":
        return torch.sinh(values) * float(spec["scale"])
    raise ValueError(f"unknown transform kind {kind!r}")


def physical_zero(spec: Mapping[str, object]) -> float:
    """Return ``forward(0)``: the physical zero in the model's input space."""
    kind = spec["kind"]
    if kind == "minmax":
        span = float(spec["hi"]) - float(spec["lo"])
        return 0.0 if span <= _EPS else (0.0 - float(spec["lo"])) / span
    if kind in _ZSCORE_FAMILY:
        return -float(spec["mean"]) / float(spec["std"])
    return 0.0


def validate_specs(specs: Mapping[str, object]) -> Dict[str, dict]:
    """Check a ``_transforms`` block and normalise shorthands to dicts."""
    clean: Dict[str, dict] = {}
    for name, spec in specs.items():
        spec = {"kind": spec} if isinstance(spec, str) else dict(spec)
        kind = spec.get("kind")
        if kind not in KINDS:
            raise ValueError(f"{name}: unknown transform kind {kind!r}")
        if name in SIGNED and kind in ("log1p_zscore", "sqrt_zscore"):
            raise ValueError(
                f"{name}: {kind} is invalid on a signed channel (use zscore/minmax)"
            )
        clean[name] = spec
    return clean


def finalise_specs(
    specs: Mapping[str, object],
    percentiles: Optional[Mapping[str, Mapping[str, float]]] = None,
    moments: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
) -> Dict[str, dict]:
    """Resolve each spec against the moments of the map it selects.

    A channel can have different statistics in the raw, ``log1p`` and ``sqrt``
    spaces. The builder therefore passes ``moments[name][kind]`` so the spec
    carries the ``mean``/``std`` that match its own transform; reusing the flat
    (default) entry would scale a channel with the wrong statistics.
    """
    out: Dict[str, dict] = {}
    for name, spec in specs.items():
        spec = {"kind": spec} if isinstance(spec, str) else dict(spec)
        if spec.get("kind") in _ZSCORE_FAMILY:
            if (
                spec.get("kind") == "asinh_zscore"
                and float(spec.get("scale", 1.0)) != 1.0
            ):
                raise ValueError(
                    f"{name}: stored asinh moments assume scale=1.0, "
                    f"got {spec['scale']}"
                )
            if "mean" not in spec or "std" not in spec:
                table = (moments or {}).get(name, {}).get(spec["kind"])
                if not table:
                    raise ValueError(
                        f"{name}: no {spec['kind']} moments available to finalise"
                    )
                spec.setdefault("mean", table["mean"])
                spec.setdefault("std", table["std"])
        elif spec.get("kind") == "minmax":
            if "bounds" in spec:
                lo, hi = spec.pop("bounds")
            elif "percentiles" in spec:
                qs = spec.pop("percentiles")
                table = (percentiles or {}).get(name)
                if not table:
                    raise ValueError(f"{name}: no percentiles available for minmax")
                lo = table[f"p{qs[0]:g}"]
                hi = table[f"p{qs[1]:g}"]
            else:
                lo, hi = spec.get("lo"), spec.get("hi")
            if lo is None or hi is None:
                raise ValueError(f"{name}: minmax needs bounds, percentiles or lo/hi")
            spec["lo"] = float(lo)
            spec["hi"] = float(hi)
            spec.setdefault("clip", True)
        out[name] = spec
    return validate_specs(out)


def describe_specs(norm_stats, names: Optional[Iterable[str]] = None) -> str:
    """Summarise the effective transforms, e.g. ``9 zscore, 4 log1p_zscore``."""
    if not isinstance(norm_stats, dict):
        return "none"
    if names is None:
        names = [
            key
            for key, value in norm_stats.items()
            if key != "_transforms" and isinstance(value, dict) and "mean" in value
        ]
    counts = Counter(get_spec(norm_stats, name)["kind"] for name in names)
    return ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
