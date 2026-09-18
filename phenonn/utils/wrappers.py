# Model wrapper for single-day prediction.
#
# The RTnn models all output (batch, output_channel, seq_length).
# For GCC prediction we want (batch, 1) — the value on the last day
# of the input window. This wrapper extracts that last timestep
# without modifying any of the existing model code.

import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from phenonn.utils.config import N_PFT, PFT_START


def _build_obs_positions() -> list:
    """
    0-indexed positions within a 365-day (non-leap) year for days 5, 15, 25
    of every month — the 36 days on which LAI observations are available.
    """
    positions = []
    for month in range(1, 13):
        for day in [5, 15, 25]:
            doy = datetime.date(2001, month, day).timetuple().tm_yday
            positions.append(doy - 1)
    return positions


_OBS_POSITIONS = _build_obs_positions()  # 36 positions


class SingleDayWrapper(nn.Module):
    """
    Wraps any RTnn model to output a single scalar per sample.

    Takes the last timestep of the wrapped model's sequence output,
    yielding shape (batch, output_channel) instead of
    (batch, output_channel, seq_length).

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (batch, C_in, L) -> (batch, C_out, L).

    Examples
    --------
    >>> from rtnn.models.rnn import RNN_LSTM
    >>> base = RNN_LSTM(feature_channel=14, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = SingleDayWrapper(base)
    >>> x = torch.randn(32, 14, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, feature_channels, seq_length).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, output_channel).
        """
        # Full sequence output: (batch, output_channel, seq_length)
        out = self.base_model(x)
        # Take last timestep: (batch, output_channel)
        return out[:, :, -1]

class permuteWrapper(nn.Module):
    """
    Permutes input dimensions to match expected order for RTnn models.

    RTnn models expect input shape (batch, feature_channels, seq_length).
    If your data is in (batch, seq_length, feature_channels), this wrapper
    permutes the dimensions before passing to the base model.

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (batch, C_in, L) -> (batch, C_out, L).

    Examples
    --------
    >>> from rtnn.models.rnn import RNN_LSTM
    >>> base = RNN_LSTM(feature_channel=14, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = permuteWrapper(base)
    >>> x = torch.randn(32, 365, 14)  # Note seq_length and feature_channels swapped
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # (B, features, seq) → (B, seq, features)
        x = x.permute(0, 2, 1)
        out = self.base_model(x)
        # (B, seq, output) → (B, output, seq)
        return out.permute(0, 2, 1)


class LastNDaysWrapper(nn.Module):
    """
    Wraps any RTnn model to output the last N timesteps.

    Takes the last N timesteps of the wrapped model's sequence output,
    yielding shape (batch, output_channel, N) instead of
    (batch, output_channel, seq_length).

    Used for gradient-aware loss where we need GCC(t) and GCC(t-1)
    while keeping the full input window for context.

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (batch, C_in, L) -> (batch, C_out, L).
    n_days : int
        Number of trailing timesteps to keep (default: 2).

    Examples
    --------
    >>> base = RNN_LSTM(feature_channel=16, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = LastNDaysWrapper(base, n_days=2)
    >>> x = torch.randn(32, 16, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1, 2])
    """

    def __init__(self, base_model: nn.Module, n_days: int = 2) -> None:
        super().__init__()
        self.base_model = base_model
        self.n_days = n_days

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channels, seq_length)

        Returns
        -------
        (batch, output_channel, n_days)
        """
        out = self.base_model(x)          # (B, C_out, L)
        return out[:, :, -self.n_days:]    # (B, C_out, n_days)


class Every10DaysWrapper(nn.Module):
    """
    Wraps any RTnn model to output predictions at the 36 LAI observation days.

    Extracts the last 365 timesteps of the model output, then selects the
    36 positions corresponding to days 5, 15, 25 of each month in a
    non-leap year.

    Input  : (B, C_in, L)  with L ≥ 365
    Output : (B, C_out, 36)

    Parameters
    ----------
    base_model : nn.Module
        Any model with forward signature (B, C_in, L) -> (B, C_out, L).

    Examples
    --------
    >>> base = RNN_LSTM(feature_channel=31, output_channel=1,
    ...                 hidden_size=128, num_layers=2)
    >>> model = Every10DaysWrapper(base)
    >>> x = torch.randn(32, 31, 720)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1, 36])
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        pos = torch.tensor(_OBS_POSITIONS, dtype=torch.long)
        self.register_buffer("obs_positions", pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channels, seq_length)  — seq_length ≥ 365

        Returns
        -------
        (batch, output_channel, 36)
        """
        out = self.base_model(x)                       # (B, C_out, L)
        last_year = out[:, :, -365:]                   # (B, C_out, 365)
        return last_year[:, :, self.obs_positions]     # (B, C_out, 36)


class DailyWrapper(nn.Module):
    """
    Daily twin of Every10DaysWrapper: returns the model output at EVERY day of
    the last year (365) instead of only the 36 dekad positions.

    Input  : (B, C_in, L)  with L ≥ 365
    Output : (B, C_out, 365)

    Used when training on a daily-interpolated LAI target (loss on all 365 days).
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)[:, :, -365:]         # (B, C_out, 365)


"""
PFT-aware output wrappers for PhenoNN models.

Two wrappers are exported:
  - `Every10DaysWrapper`    : re-exported from phenonn.utils.wrappers for convenience.
  - `PFTMixingWrapper`      : weighted sum across the 15-channel PFT
                              decomposition produced by a base model whose
                              `output_channel = 15`.

The PFTMixingWrapper is functionally identical to the one in
LaiNN/phenocam/main_bis.py
"""




class PFTMixingWrapper(nn.Module):
    """
    Wraps a base model whose output has 15 channels (one "pure" LAI per PFT)
    so that it returns the area-weighted global LAI at the 36 observation
    days of the last year.

    Forward pipeline
    ----------------
      1. base_model(x)              → (B, 15, L)
      2. last 365 days              → (B, 15, 365)   (skipped if sparse_output)
      3. pick the 36 obs positions  → (B, 15, 36)
      4. extract PFT fractions from x and denormalize → (B, 15, 1)
      5. optionally force PFT 1 (bare soil) LAI to 0
      6. optionally renormalize so PFT fractions sum to 1
      7. weighted sum across the PFT dimension → (B, 1, 36)

    Parameters mirror the LaiNN implementation; see its docstring for the
    rationale behind each option (zero_pft1_lai, lai_normalized, etc.).
    """

    def __init__(
        self,
        base_model: nn.Module,
        norm_stats=None,
        pft_start: int = PFT_START,
        n_pft: int = N_PFT,
        renormalize: bool = True,
        sparse_output: bool = False,
        zero_pft1_lai: bool = True,
        lai_normalized: bool = True,
        meteo_only: bool = False,
        nonneg: bool = False,
        daily: bool = False,
    ) -> None:
        super().__init__()
        self.base_model    = base_model
        self.pft_start     = pft_start
        self.n_pft         = n_pft
        self.renormalize   = renormalize
        self.sparse_output = sparse_output
        # When True, the mixed LAI is returned at EVERY day of the last year
        # (B, 1, 365) instead of the 36 dekad positions — daily-target training.
        self.daily         = daily
        self.zero_pft1_lai = zero_pft1_lai
        # When True the base model was built to see only the meteo/cyclic/co2
        # channels (0 : pft_start); its pure-LAI outputs L_k then depend on
        # climate alone and the PFT fractions enter only as mixing weights below.
        self.meteo_only    = meteo_only

        # When True, each per-PFT pure LAI is soft-floored at physical 0 in the
        # forward (softplus anchored at z0 = -mean/std, the physical zero in the
        # z-scored LAI space). Guarantees L_k ≥ 0 physically while keeping the
        # additive mixing consistent. z0 = 0 when LAI is not normalized.
        self.nonneg = nonneg
        if nonneg and lai_normalized and norm_stats is not None and "LAI" in norm_stats:
            self.nonneg_z0 = -float(norm_stats["LAI"]["mean"]) / max(
                float(norm_stats["LAI"]["std"]), 1e-8)
        else:
            self.nonneg_z0 = 0.0

        pos = torch.tensor(_OBS_POSITIONS, dtype=torch.long)
        self.register_buffer("obs_positions", pos)

        # PFT-1 (bare soil) override masks. Non-persistent buffers so older
        # checkpoints (without these) load cleanly.
        lai_mask = torch.ones(1, n_pft, 1, dtype=torch.float32)
        lai_bias = torch.zeros(1, n_pft, 1, dtype=torch.float32)
        if zero_pft1_lai:
            lai_mask[0, 0, 0] = 0.0
            if lai_normalized and norm_stats is not None and "LAI" in norm_stats:
                lai_mean = float(norm_stats["LAI"]["mean"])
                lai_std  = float(norm_stats["LAI"]["std"])
                lai_bias[0, 0, 0] = -lai_mean / max(lai_std, 1e-8)
        self.register_buffer("lai_pft_mask", lai_mask, persistent=False)
        self.register_buffer("lai_pft_bias", lai_bias, persistent=False)

        # Inverse z-score of the PFT input channels (so the wrapper recovers
        # physical area fractions before computing the weighted sum).
        has_pft_norm = (
            norm_stats is not None
            and f"pft1_frac" in norm_stats
        )
        if has_pft_norm:
            means = torch.tensor(
                [norm_stats[f"pft{k}_frac"]["mean"] for k in range(1, n_pft + 1)],
                dtype=torch.float32,
            ).view(1, n_pft, 1)
            stds = torch.tensor(
                [norm_stats[f"pft{k}_frac"]["std"]  for k in range(1, n_pft + 1)],
                dtype=torch.float32,
            ).view(1, n_pft, 1)
        else:
            means = torch.zeros(1, n_pft, 1, dtype=torch.float32)
            stds  = torch.ones(1, n_pft, 1, dtype=torch.float32)
        self.register_buffer("pft_means", means)
        self.register_buffer("pft_stds",  stds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In meteo_only mode the base model expects only the meteo/cyclic/co2
        # channels; the PFT block (read below for the weights) is stripped here.
        base_in = x[:, : self.pft_start, :] if self.meteo_only else x
        out = self.base_model(base_in)                          # (B, 15, L) or (B, 15, 36)
        if self.nonneg:
            # Soft floor at physical 0: L_k^norm ≥ z0 ⟺ L_k^phys ≥ 0.
            out = self.nonneg_z0 + F.softplus(out - self.nonneg_z0)
        if self.sparse_output:
            lai_pure = out
        elif self.daily:
            lai_pure = out[:, :, -365:]                          # (B, 15, 365)
        else:
            last_year = out[:, :, -365:]
            lai_pure  = last_year[:, :, self.obs_positions]      # (B, 15, 36)

        # PFT-1 override (bare soil → physical 0)
        lai_pure = lai_pure * self.lai_pft_mask + self.lai_pft_bias

        # Recover PFT physical fractions from the last input timestep
        pft_norm = x[:, self.pft_start : self.pft_start + self.n_pft, -1:]
        pft_real = pft_norm * self.pft_stds + self.pft_means
        pft_real = pft_real.clamp(min=0.0)
        if self.renormalize:
            tot = pft_real.sum(dim=1, keepdim=True).clamp(min=1e-6)
            pft_real = pft_real / tot

        return (lai_pure * pft_real).sum(dim=1, keepdim=True)    # (B, 1, 36)
