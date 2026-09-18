"""
AttnLSTM — Transformer-stress front end + LSTM decoder.

Same two-stage factorisation as ``BiTransformerV2`` (see Bitransformer.py), but
the stage-2 Transformer is replaced by a unidirectional (causal) LSTM:

  Stage 1  weather features → vectorial "stress" per timestep
           (Conv1d(k=3) → sinusoidal PE → LayerNorm → TransformerEncoder → Linear)
  Stage 2  [stress, PFT fractions] → LAI per timestep
           (LSTM(hidden_size, num_layers) → Linear)

The LSTM is left-to-right only, so output day ``t`` sees the stress of days
``≤ t`` (no future leakage), matching the causal stage-1 mask.

Interface identical to BiTransformerV2 so it stays a drop-in replacement through
``permuteWrapper``:
  Input : (B, L, input_dim)  with the last ``n_pft`` channels = PFT fractions
  Output: (B, L, output_dim)
"""

import torch
import torch.nn as nn

from phenonn.models.transformer import SinusoidalPositionalEncoding


class AttnLSTM(nn.Module):
    """
    Transformer-stress stage 1 + causal LSTM stage 2 for LAI prediction.

    Parameters mirror BiTransformerV2's stage-1 ones; stage 2 is an LSTM.

    Parameters
    ----------
    input_dim : int
        Total input channel count (= n_weather + n_pft).
    output_dim : int
        Final output channels (1 for scalar LAI; 15 for PFT mixing).
    d_model : int
        Embedding dimension of the stage-1 transformer (weather → stress).
    lstm_hidden : int
        Hidden size of the stage-2 LSTM (decoupled from ``d_model``).
    n_pft : int
        Number of trailing PFT channels in ``input_dim``. The first
        ``input_dim - n_pft`` channels are the weather/derived features.
    stress_dim : int
        Dimensionality of the intermediate "stress" representation.
    nr_blocks_stage1 : int
        Number of transformer-encoder layers in stage 1 (weather → stress).
    lstm_layers : int
        Number of stacked LSTM layers in stage 2 (stress+PFT → LAI).
    nhead : int
        Number of stage-1 attention heads. Must divide ``d_model``.
    feed_forward_trans : int
        Stage-1 FFN multiplier (d_ff = feed_forward_trans * d_model).
    dropout_trans : float
        Dropout in stage 1.
    dropout_lstm : float
        Dropout between LSTM layers (ignored when lstm_layers == 1).
    seq_length : int
        Maximum input length (used to precompute the stage-1 causal mask).
    causal : bool
        If True, apply a triangular causal mask to the stage-1 transformer.
        The stage-2 LSTM is causal by construction regardless of this flag.

    Examples
    --------
    >>> model = AttnLSTM(input_dim=31, output_dim=1, d_model=128, n_pft=15,
    ...                  stress_dim=8, nr_blocks_stage1=2, lstm_layers=2,
    ...                  nhead=4, seq_length=720)
    >>> x = torch.randn(8, 720, 31)
    >>> model(x).shape
    torch.Size([8, 720, 1])
    """

    def __init__(
        self,
        input_dim: int = 31,
        output_dim: int = 1,
        d_model: int = 128,
        lstm_hidden: int = 128,
        n_pft: int = 15,
        stress_dim: int = 8,
        nr_blocks_stage1: int = 2,
        lstm_layers: int = 2,
        nhead: int = 4,
        feed_forward_trans: int = 4,
        dropout_trans: float = 0.05,
        dropout_lstm: float = 0.0,
        seq_length: int = 720,
        causal: bool = True,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        n_weather = input_dim - n_pft
        if n_weather <= 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be > n_pft ({n_pft})"
            )
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal PE")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_pft = n_pft
        self.n_weather = n_weather
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.stress_dim = stress_dim
        self.seq_length = seq_length
        self.causal = causal

        # ── Stage 1: weather → stress (identical to BiTransformerV2) ──
        self.weather_conv = nn.Conv1d(
            in_channels=n_weather,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        nn.init.kaiming_normal_(
            self.weather_conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

        self.pos_enc1  = SinusoidalPositionalEncoding(
            d_model, max_len=max(seq_length, 5000)
        )
        self.norm_emb1 = nn.LayerNorm(d_model)
        self.drop_emb1 = nn.Dropout(dropout_trans)

        stage1_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feed_forward_trans * d_model,
            dropout=dropout_trans,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stage1      = nn.TransformerEncoder(stage1_layer,
                                                 num_layers=nr_blocks_stage1)
        self.stage1_norm = nn.LayerNorm(d_model)
        self.to_stress   = nn.Linear(d_model, stress_dim)

        # ── Stage 2: stress + PFT → LAI (causal LSTM) ──
        self.lstm = nn.LSTM(
            input_size=stress_dim + n_pft,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
        )
        self.to_output = nn.Linear(lstm_hidden, output_dim)

        # ── Stage-1 causal mask (precomputed) ──
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_length)
            self.register_buffer("causal_mask", mask)
        else:
            self.causal_mask = None

    def _get_mask(self, L: int):
        """Return a (L, L) causal mask matching the actual sequence length."""
        if self.causal_mask is None:
            return None
        if L == self.causal_mask.size(0):
            return self.causal_mask
        return self.causal_mask[:L, :L]

    def forward(self, x: torch.Tensor, return_stress: bool = False):
        """
        Parameters
        ----------
        x : (B, L, input_dim). Last ``n_pft`` channels = PFT fractions.
        return_stress : if True, also return the (B, L, stress_dim) stress.

        Returns
        -------
        out : (B, L, output_dim)
        stress : (B, L, stress_dim), optional
        """
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} channels, got "
                f"{x.size(-1)}."
            )
        B, L, _ = x.shape
        mask = self._get_mask(L)

        if self.n_pft > 0:
            weather = x[:, :, : -self.n_pft]  # (B, L, n_weather)
            pft     = x[:, :, -self.n_pft :]  # (B, L, n_pft)
        else:
            weather = x
            pft     = x[:, :, :0]             # (B, L, 0)

        # ── Stage 1: weather → stress ──
        h = weather.transpose(1, 2)                  # (B, n_weather, L)
        h = self.weather_conv(h).transpose(1, 2)     # (B, L, d_model)
        h = self.pos_enc1(h)
        h = self.norm_emb1(h)
        h = self.drop_emb1(h)
        h = self.stage1(h, mask=mask)                # (B, L, d_model)
        h = self.stage1_norm(h)
        stress = self.to_stress(h)                   # (B, L, stress_dim)

        # ── Stage 2: [stress, PFT] → LAI ──
        y = torch.cat([stress, pft], dim=-1)          # (B, L, stress_dim + n_pft)
        y, _ = self.lstm(y)                           # (B, L, lstm_hidden)
        out = self.to_output(y)                       # (B, L, output_dim)

        if return_stress:
            return out, stress
        return out
