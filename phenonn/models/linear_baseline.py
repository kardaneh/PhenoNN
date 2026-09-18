# Linear regression baseline for LAI prediction.
#
# Two variants:
#   - LinearBaseline: flattens the full (C, L) window and applies a single
#     Linear layer. This is equivalent to ordinary least squares regression
#     on all C×L input values. No hidden layers, no activation, no nonlinearity.
#
#   - PerDayLinearBaseline: applies a Linear(C, 1) independently to each
#     timestep, then takes the last day's output. This tests whether a
#     simple per-day linear combination of the 16 features (without any
#     temporal context) can predict LAI.
#
# Both follow the SingleDayWrapper convention: input (B, C, L) → output (B, 1).

import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    """
    Full-window linear regression baseline.

    Flattens the entire (feature_channel × seq_length) input into one vector,
    applies a single Linear layer to produce a scalar prediction.

    This is mathematically equivalent to:
        gcc_pred = w @ flatten(input) + b

    where w has C×L weights. It can learn things like "temperature on day 300
    matters more than temperature on day 50" because each (channel, timestep)
    pair gets its own weight.

    Parameters
    ----------
    feature_channel : int
        Number of input feature channels.
    seq_length : int
        Window length in days (365).

    Examples
    --------
    >>> model = LinearBaseline(feature_channel=16, seq_length=365)
    >>> x = torch.randn(32, 16, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """

    def __init__(self, feature_channel: int, seq_length: int = 365):
        super().__init__()
        self.feature_channel = feature_channel
        self.seq_length = seq_length
        self.linear = nn.Linear(feature_channel * seq_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channel, seq_length)

        Returns
        -------
        (batch, 1)
        """
        # Flatten: (B, C, L) → (B, C*L)
        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class PerDayLinearBaseline(nn.Module):
    """
    Per-day linear regression baseline (no temporal context).

    Applies the same Linear(C, 1) to every timestep independently, then
    returns the prediction for the last day. This tests whether today's
    feature values alone (without any history) can predict today's LAI.

    If this baseline scores well, it means the temporal context from the
    365-day window isn't adding much — the model is mostly using the
    current day's meteorology.

    Parameters
    ----------
    feature_channel : int
        Number of input feature channels.

    Examples
    --------
    >>> model = PerDayLinearBaseline(feature_channel=16)
    >>> x = torch.randn(32, 16, 365)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 1])
    """

    def __init__(self, feature_channel: int):
        super().__init__()
        self.linear = nn.Linear(feature_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, feature_channel, seq_length)

        Returns
        -------
        (batch, 1)
        """
        # Permute to (B, L, C), take last day, apply linear
        x = x.permute(0, 2, 1)   # (B, L, C)
        last_day = x[:, -1, :]   # (B, C)
        return self.linear(last_day)
