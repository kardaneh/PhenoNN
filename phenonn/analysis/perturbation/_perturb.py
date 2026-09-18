"""
Physical-space input perturbation for trained PhenoNN models.

A `Perturbation` installs a forward-pre-hook on a built model (exactly like the
feature-ablation study — no change to `phenonn/`). For each targeted meteo
channel the hook, on the ALREADY-NORMALIZED input tensor x (B, C, L):

    z  → physical  (invert z-score, and expm1 for the log1p features)
       → perturb   (physical += add ; physical *= mul)  [ceteris paribus]
       → z'        (re-apply log1p + z-score)

so a "+5 °C" or "×0.5 precipitation" change is expressed in real units and only
the requested channels move (VPD / PET / Rn / etc. are held fixed). An optional
day-of-year window restricts the change to a season (heatwave = summer, frost =
winter); positions outside the window keep their original value.

The 720/730-day input window ends on DOY 365 of the target year, so position i
maps to a DOY on a 365-day calendar with the LAST position = DOY 365 (see
`doy_of_positions`).
"""

import numpy as np
import torch

from phenonn.utils.config import ALL_FEATURES, LOG_TRANSFORM_FEATURES


def doy_of_positions(seq_length: int) -> np.ndarray:
    """1-based DOY (365-day calendar) for each of the `seq_length` window
    positions, with the LAST position = DOY 365 and wrapping backwards."""
    back = (seq_length - 1 - np.arange(seq_length)) % 365
    return 365 - back                       # 1 (oldest wrap) .. 365 (last day)


def season_mask(seq_length: int, ranges) -> np.ndarray:
    """Boolean (seq_length,) mask True inside any (lo, hi) inclusive DOY range.
    A range with lo > hi wraps the new year (e.g. DJF = (335, 59)). `ranges`
    empty / None → all True (perturb the whole window)."""
    if not ranges:
        return np.ones(seq_length, dtype=bool)
    doy = doy_of_positions(seq_length)
    m = np.zeros(seq_length, dtype=bool)
    for lo, hi in ranges:
        if lo <= hi:
            m |= (doy >= lo) & (doy <= hi)
        else:                               # wrap around Dec→Jan
            m |= (doy >= lo) | (doy <= hi)
    return m


# Named DOY windows (non-leap calendar) for the seasonal scenarios.
SEASONS = {
    "year":   None,
    "summer": [(152, 243)],                 # Jun 1 – Aug 31 (JJA)
    "winter": [(335, 59)],                  # Dec 1 – Feb 28 (DJF, wraps)
    "spring": [(60, 151)],                  # Mar 1 – May 31 (MAM)
    "autumn": [(244, 334)],                 # Sep 1 – Nov 30 (SON)
    "growing": [(91, 304)],                 # Apr 1 – Oct 31
}


class ChannelPerturbation:
    """One channel's change in physical units: physical' = physical * mul + add."""

    def __init__(self, feature: str, add: float = 0.0, mul: float = 1.0) -> None:
        if feature not in ALL_FEATURES:
            raise SystemExit(f"unknown feature {feature!r}; not in ALL_FEATURES")
        self.feature = feature
        self.ch = ALL_FEATURES.index(feature)
        self.add = float(add)
        self.mul = float(mul)
        self.is_log = feature in LOG_TRANSFORM_FEATURES


class Perturbation:
    """A set of channel perturbations + an optional seasonal window, installable
    as a forward-pre-hook on a built model."""

    def __init__(self, channels, norm_stats, season_ranges=None) -> None:
        if not channels:
            raise SystemExit("Perturbation needs at least one channel.")
        self.channels = list(channels)
        self.norm_stats = norm_stats                 # None → inputs are raw
        self.season_ranges = season_ranges
        self._mask_cache: dict = {}

    def _mask(self, seq_length: int, device) -> torch.Tensor:
        if seq_length not in self._mask_cache:
            m = season_mask(seq_length, self.season_ranges).astype(np.float32)
            self._mask_cache[seq_length] = torch.from_numpy(m)
        return self._mask_cache[seq_length].to(device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return a perturbed copy of the normalized input x (B, C, L)."""
        seq_length = x.shape[2]
        mask = self._mask(seq_length, x.device)      # (L,)
        keep = 1.0 - mask
        x = x.clone()
        for cp in self.channels:
            z = x[:, cp.ch, :]                       # (B, L), normalized
            if self.norm_stats is not None:
                st = self.norm_stats.get(cp.feature)
                if st is None:
                    raise SystemExit(
                        f"norm_stats has no entry for {cp.feature!r}.")
                mu = float(st["mean"])
                sd = max(float(st["std"]), 1e-8)
                phys = torch.expm1(z * sd + mu) if cp.is_log else z * sd + mu
            else:                                    # raw inputs: z IS physical
                mu, sd = 0.0, 1.0
                phys = z
            phys_new = phys * cp.mul + cp.add
            if self.norm_stats is not None and cp.is_log:
                z_new = (torch.log1p(phys_new.clamp(min=0.0)) - mu) / sd
            elif self.norm_stats is not None:
                z_new = (phys_new - mu) / sd
            else:
                z_new = phys_new
            x[:, cp.ch, :] = z * keep + z_new * mask
        return x

    def hook(self, _module, inputs):
        t = inputs[0]
        if not torch.is_tensor(t):
            return None
        return (self.apply(t),) + tuple(inputs[1:])

    def describe(self) -> str:
        season = "year" if not self.season_ranges else str(self.season_ranges)
        parts = []
        for cp in self.channels:
            op = []
            if cp.mul != 1.0:
                op.append(f"×{cp.mul:g}")
            if cp.add != 0.0:
                op.append(f"{cp.add:+g}")
            parts.append(f"{cp.feature} {' '.join(op) or '(no-op)'}")
        return f"[{'; '.join(parts)}] over {season}"
