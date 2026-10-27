"""
Physical-space input perturbation for trained PhenoNN models.

A ``Perturbation`` installs a forward-pre-hook on a built model, exactly like
the feature-ablation study. It does not change ``phenonn/``. For each targeted
meteo channel the hook receives the already-normalized input tensor
``x`` with shape ``(B, C, L)`` and performs this sequence:

1. Convert normalized values to physical units. Invert the z-score and apply
   ``expm1`` to log1p features.
2. Apply the requested change in physical units: ``physical = physical * mul +
   add``.
3. Convert back to normalized units with log1p and z-score.

A ``+5 C`` or ``x0.5 precipitation`` change is therefore expressed in real
units, and only the requested channels move. VPD, PET, radiation, and the other
channels are held fixed. An optional day-of-year window restricts the change to
a season, for example summer for a heatwave or winter for frost. Positions
outside the window keep their original value.

The 720 or 730 day input window ends on DOY 365 of the target year, so position
``i`` maps to a DOY on a 365 day calendar with the last position equal to DOY
365. See :func:`doy_of_positions`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from phenonn.utils.config import ALL_FEATURES, LOG_TRANSFORM_FEATURES

__all__ = [
    "SEASONS",
    "ChannelPerturbation",
    "Perturbation",
    "doy_of_positions",
    "resolve_hemisphere",
    "season_mask",
]


def doy_of_positions(seq_length: int) -> np.ndarray:
    """Return the 1-based day of year for each input-window position.

    Parameters
    ----------
    seq_length : int
        Number of daily positions in the model input window.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(seq_length,)``. The last position is DOY 365,
        and earlier positions wrap backwards on a 365 day calendar.
    """
    back = (seq_length - 1 - np.arange(seq_length)) % 365
    return 365 - back  # 1 (oldest wrap) .. 365 (last day)


def season_mask(
    seq_length: int,
    ranges: Sequence[tuple[int, int]] | None,
    hemisphere: str = "north",
) -> np.ndarray:
    """Build a boolean mask for one or more inclusive day-of-year ranges.

    Parameters
    ----------
    seq_length : int
        Number of daily positions in the model input window.
    ranges : sequence of tuple of int, optional
        Inclusive ``(start, end)`` day-of-year ranges. A range with
        ``start > end`` wraps across the new year, for example
        ``(335, 59)`` for December to February. ``None`` or an empty sequence
        selects the whole window.
    hemisphere : {"north", "south"}
        Southern-hemisphere seasons are shifted by 182 days so that the
        ``summer`` window still means the local summer.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(seq_length,)`` that is true inside any of the
        requested ranges.
    """
    if not ranges:
        return np.ones(seq_length, dtype=bool)
    doy = doy_of_positions(seq_length)
    if str(hemisphere).lower() in ("south", "sh", "southern"):
        doy = ((doy - 1 + 182) % 365) + 1
    mask = np.zeros(seq_length, dtype=bool)
    for lo, hi in ranges:
        if lo <= hi:
            mask |= (doy >= lo) & (doy <= hi)
        else:  # Wrap around December to January.
            mask |= (doy >= lo) | (doy <= hi)
    return mask


def resolve_hemisphere(latitude) -> str:
    """Return ``"north"`` or ``"south"`` for a scalar latitude."""
    try:
        return "south" if float(latitude) < 0.0 else "north"
    except (TypeError, ValueError):
        return "north"


# Named DOY windows (non-leap calendar) for the seasonal scenarios.
SEASONS = {
    "year": None,
    "summer": [(152, 243)],  # Jun 1 to Aug 31 (JJA)
    "winter": [(335, 59)],  # Dec 1 to Feb 28 (DJF, wraps)
    "spring": [(60, 151)],  # Mar 1 to May 31 (MAM)
    "autumn": [(244, 334)],  # Sep 1 to Nov 30 (SON)
    "growing": [(91, 304)],  # Apr 1 to Oct 31
}


class ChannelPerturbation:
    """One channel's change in physical units.

    The new value is ``physical * mul + add``.

    Parameters
    ----------
    feature : str
        Feature name that must appear in ``ALL_FEATURES``.
    add : float, optional
        Additive change in physical units. The default is ``0.0``.
    mul : float, optional
        Multiplicative change in physical units. The default is ``1.0``.
    """

    def __init__(
        self, feature: str, add: float = 0.0, mul: float = 1.0, set_to: float | None = None
    ) -> None:
        if feature not in ALL_FEATURES:
            raise SystemExit(f"unknown feature {feature!r}; not in ALL_FEATURES")
        self.feature = feature
        self.ch = ALL_FEATURES.index(feature)
        self.add = float(add)
        self.mul = float(mul)
        self.set_to = None if set_to is None else float(set_to)
        self.is_log = feature in LOG_TRANSFORM_FEATURES


class Perturbation:
    """A set of channel changes and an optional seasonal window.

    The object can be installed as a forward-pre-hook on a built model. It
    stores one mask per input length and device on first use.

    Parameters
    ----------
    channels : sequence of ChannelPerturbation
        Channel changes to apply. At least one is required.
    norm_stats : mapping or None
        Per-feature normalization statistics. Use ``None`` when the model
        receives raw physical inputs.
    season_ranges : sequence of tuple of int, optional
        Inclusive day-of-year ranges for the change. ``None`` applies it to the
        whole window.
    """

    def __init__(
        self,
        channels,
        norm_stats,
        season_ranges=None,
        hemisphere="north",
        position_mask=None,
    ) -> None:
        if not channels:
            raise SystemExit("Perturbation needs at least one channel.")
        self.channels = list(channels)
        self.norm_stats = norm_stats  # None means the inputs are raw.
        self.season_ranges = season_ranges
        # "north", "south" or "auto".  "auto" requires per-sample latitudes in
        # ``apply`` so mixed-hemisphere datasets use the correct local season.
        self.hemisphere = str(hemisphere).lower()
        # Explicit per-position mask (length = seq_length) for impulse studies.
        self.position_mask = (
            None if position_mask is None else np.asarray(position_mask, dtype=bool)
        )
        self._mask_cache = {}

    def _mask(self, seq_length: int, device) -> torch.Tensor:
        """Return the seasonal mask for one input length and device."""
        if seq_length not in self._mask_cache:
            if self.position_mask is not None:
                if self.position_mask.size != seq_length:
                    raise ValueError(
                        f"position_mask has length {self.position_mask.size}, "
                        f"expected {seq_length}"
                    )
                mask = self.position_mask.astype(np.float32)
            else:
                hemi = (
                    self.hemisphere
                    if self.hemisphere in ("north", "south")
                    else "north"
                )
                mask = season_mask(
                    seq_length, self.season_ranges, hemisphere=hemi
                ).astype(np.float32)
            self._mask_cache[seq_length] = torch.from_numpy(mask)
        return self._mask_cache[seq_length].to(device)

    def _sample_mask(self, seq_length: int, device, lats) -> torch.Tensor:
        """Return a ``(B, L)`` mask for per-sample latitudes."""
        base = torch.as_tensor(
            doy_of_positions(seq_length).astype(np.float32), device=device
        ).unsqueeze(0)  # (1, L)
        lat = torch.as_tensor(lats, dtype=torch.float32, device=device).reshape(-1, 1)
        shift = torch.where(lat < 0.0, torch.full_like(lat, 182.0), torch.zeros_like(lat))
        doy = torch.remainder(base - 1.0 + shift, 365.0) + 1.0  # (B, L)
        if not self.season_ranges:
            return torch.ones_like(doy)
        mask = torch.zeros_like(doy)
        for lo, hi in self.season_ranges:
            if lo <= hi:
                m = ((doy >= lo) & (doy <= hi)).float()
            else:
                m = ((doy >= lo) | (doy <= hi)).float()
            mask = torch.maximum(mask, m)
        return mask

    def apply(self, x: torch.Tensor, lats=None) -> torch.Tensor:
        """Return a perturbed copy of the normalized input ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input tensor with shape ``(B, C, L)``.
        lats : array-like, optional
            Per-sample latitudes.  Required when ``hemisphere="auto"``.

        Returns
        -------
        torch.Tensor
            A new tensor with the same shape as ``x``. Channels not listed in
            ``channels`` are unchanged.
        """
        seq_length = x.shape[2]
        if self.hemisphere == "auto" and lats is not None:
            mask = self._sample_mask(seq_length, x.device, lats)  # (B, L)
        else:
            mask = self._mask(seq_length, x.device).unsqueeze(0)  # (1, L)
        keep = 1.0 - mask
        x = x.clone()
        for channel_perturbation in self.channels:
            z = x[:, channel_perturbation.ch, :]  # (B, L), normalized
            if self.norm_stats is not None:
                stats = self.norm_stats.get(channel_perturbation.feature)
                if stats is None:
                    raise SystemExit(
                        f"norm_stats has no entry for {channel_perturbation.feature!r}."
                    )
                mean = float(stats["mean"])
                std = max(float(stats["std"]), 1e-8)
                if channel_perturbation.is_log:
                    physical = torch.expm1(z * std + mean)
                else:
                    physical = z * std + mean
            else:  # Raw inputs: z is already physical.
                mean, std = 0.0, 1.0
                physical = z

            if channel_perturbation.set_to is not None:
                physical_new = torch.full_like(physical, channel_perturbation.set_to)
            else:
                physical_new = (
                    physical * channel_perturbation.mul + channel_perturbation.add
                )
            if self.norm_stats is not None and channel_perturbation.is_log:
                z_new = (torch.log1p(physical_new.clamp(min=0.0)) - mean) / std
            elif self.norm_stats is not None:
                z_new = (physical_new - mean) / std
            else:
                z_new = physical_new
            x[:, channel_perturbation.ch, :] = z * keep + z_new * mask
        return x

    def hook(self, _module, inputs):
        """Forward-pre-hook that applies the perturbation to the first input."""
        tensor = inputs[0]
        if not torch.is_tensor(tensor):
            return None
        return (self.apply(tensor),) + tuple(inputs[1:])

    def describe(self) -> str:
        """Return a short human-readable description of the perturbation."""
        season = "year" if not self.season_ranges else str(self.season_ranges)
        parts = []
        for channel_perturbation in self.channels:
            operations = []
            if channel_perturbation.set_to is not None:
                operations.append(f"={channel_perturbation.set_to:g}")
            if channel_perturbation.mul != 1.0:
                operations.append(f"x{channel_perturbation.mul:g}")
            if channel_perturbation.add != 0.0:
                operations.append(f"{channel_perturbation.add:+g}")
            description = " ".join(operations) or "(no-op)"
            parts.append(f"{channel_perturbation.feature} {description}")
        hemi = "" if self.hemisphere == "north" else f" [{self.hemisphere}]"
        return f"[{'; '.join(parts)}] over {season}{hemi}"
