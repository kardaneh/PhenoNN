"""
BiTransformer V2 — corrected and refactored version of `Transformerbis.BiTransformer`.

Architecture (concept unchanged)
--------------------------------
  Stage 1   weather features (meteo + cyclic + derived)
            → vectorial "stress" representation per timestep
  Stage 2   [stress, PFT fractions]
            → LAI prediction per timestep

Why a V2 was needed
-------------------
The original `BiTransformer` had two critical bugs that silently broke its
intended semantics:

1. **Batch / time dimensions swapped in stage 1**. It called
   `nn.Transformer(x, x)` on a tensor of shape `(B, L, d_model)` while
   `nn.Transformer` defaults to `batch_first=False` — so it interpreted
   batch as sequence and time as batch. The "attention" was computed
   across batch elements, not across time. This caused cross-sample
   information leakage and a non-deterministic forward at inference.

2. **Positional encoding instantiated but never used**. The
   `self.positional_encoder` module was created in `__init__` and never
   called in `forward`. Stage 1 was therefore a "bag of features" with no
   sense of time — broken for time series.

Plus several other suboptimalities (1-dim stress bottleneck,
encoder+decoder pair fed with `(x, x)`, no Conv1d for local context,
post-norm without LayerNorm post-embedding, ReLU rather than GELU,
masking inconsistent between stages, …).

V2 fixes all of the above:
  • `batch_first=True` everywhere
  • Sinusoidal positional encoding applied both stages
  • Stress is a small VECTOR (`stress_dim`, default 8) instead of a scalar
  • Stage 1 is a single `TransformerEncoder` stack (not Transformer
    encoder+decoder)
  • Conv1d(k=3, replicate) input projections capture local 3-day context
  • Pre-norm transformer blocks (`norm_first=True`) + GELU activation
  • LayerNorm right after embedding + positional encoding
  • Causal masking consistent between the two stages (configurable)
  • Sinusoidal PE shared with `CausalSparseTransformer` (imported from
    Transformer.py, single source of truth)

Interface (unchanged) so it stays a drop-in replacement when used through
`permuteWrapper`:
  Input : (B, L, input_dim)  with the last `n_pft` channels = PFT fractions
  Output: (B, L, output_dim)
"""

from typing import Optional

import torch
import torch.nn as nn

from phenonn.models.transformer import SinusoidalPositionalEncoding


class BiTransformerV2(nn.Module):
    """
    Two-stage Transformer for LAI prediction with explicit weather → stress
    → PFT-conditioned LAI factorisation.

    Parameters
    ----------
    input_dim : int
        Total channel count of the input (= n_weather + n_pft).
    output_dim : int
        Final output channels (1 for scalar LAI; 15 if using PFT mixing).
    d_model : int
        Embedding dimension of stage 1 (weather → stress).
    d_model2 : int, optional
        Embedding dimension of stage 2 (stress+PFT → LAI). Defaults to
        ``d_model`` when None, keeping the original single-width behaviour.
    n_pft : int
        Number of trailing PFT channels in `input_dim`. The first
        `input_dim - n_pft` channels are treated as weather/derived features.
    stress_dim : int
        Dimensionality of the intermediate "stress" representation passed
        from stage 1 to stage 2 (default 8).
    nr_blocks_stage1 : int
        Number of transformer-encoder layers in stage 1 (weather → stress).
    nr_blocks_stage2 : int
        Number of transformer-encoder layers in stage 2 (stress+PFT → LAI).
    nhead : int
        Number of attention heads. Must divide `d_model`.
    feed_forward_trans, feed_forward_encoder : int
        Inner FFN multiplier for stage 1 and stage 2 respectively.
        d_ff_stage_i = feed_forward_i * d_model.
    dropout_trans, dropout_encoder : float
        Dropout for stage 1 and stage 2.
    seq_length : int
        Maximum input length (used to precompute the causal mask buffer).
    causal : bool
        If True, apply a triangular causal mask to BOTH stages so position
        `t` can only attend to positions ≤ t. Default False to match the
        original BiTransformer's stage-1 behaviour, but recommended True
        for "no future weather leakage".

    Examples
    --------
    >>> model = BiTransformerV2(
    ...     input_dim=31, output_dim=1, d_model=128, n_pft=15,
    ...     stress_dim=8, nr_blocks_stage1=2, nr_blocks_stage2=3,
    ...     nhead=4, seq_length=720,
    ... )
    >>> x = torch.randn(8, 720, 31)
    >>> out = model(x)            # (8, 720, 1)
    >>> out, stress = model(x, return_stress=True)
    >>> stress.shape              # (8, 720, 8)
    torch.Size([8, 720, 8])
    """

    def __init__(
        self,
        input_dim: int = 31,
        output_dim: int = 1,
        d_model: int = 128,
        d_model2: Optional[int] = None,
        n_pft: int = 15,
        stress_dim: int = 8,
        nr_blocks_stage1: int = 2,
        nr_blocks_stage2: int = 3,
        nhead: int = 4,
        feed_forward_trans: int = 4,
        feed_forward_encoder: int = 4,
        dropout_trans: float = 0.05,
        dropout_encoder: float = 0.05,
        seq_length: int = 720,
        causal: bool = False,
    ) -> None:
        super().__init__()

        if d_model2 is None:
            d_model2 = d_model
        if d_model % nhead != 0 or d_model2 % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) and d_model2 ({d_model2}) must both be "
                f"divisible by nhead ({nhead})"
            )
        n_weather = input_dim - n_pft
        if n_weather <= 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be > n_pft ({n_pft})"
            )
        if d_model % 2 != 0 or d_model2 % 2 != 0:
            raise ValueError("d_model and d_model2 must be even for sinusoidal PE")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_pft = n_pft
        self.n_weather = n_weather
        self.d_model = d_model
        self.d_model2 = d_model2
        self.stress_dim = stress_dim
        self.seq_length = seq_length
        self.causal = causal

        # ── Stage 1: weather → stress ──
        # Conv1d(k=3, replicate) gives each timestep its 3-day local context
        # before any attention is computed (echoes Informer's TokenEmbedding).
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

        self.pos_enc1   = SinusoidalPositionalEncoding(
            d_model, max_len=max(seq_length, 5000)
        )
        self.norm_emb1  = nn.LayerNorm(d_model)
        self.drop_emb1  = nn.Dropout(dropout_trans)

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

        # ── Stage 2: stress + PFT → LAI ──
        self.fuse_proj   = nn.Linear(stress_dim + n_pft, d_model2)
        self.pos_enc2    = SinusoidalPositionalEncoding(
            d_model2, max_len=max(seq_length, 5000)
        )
        self.norm_emb2   = nn.LayerNorm(d_model2)
        self.drop_emb2   = nn.Dropout(dropout_encoder)

        stage2_layer = nn.TransformerEncoderLayer(
            d_model=d_model2,
            nhead=nhead,
            dim_feedforward=feed_forward_encoder * d_model2,
            dropout=dropout_encoder,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stage2      = nn.TransformerEncoder(stage2_layer,
                                                 num_layers=nr_blocks_stage2)
        self.stage2_norm = nn.LayerNorm(d_model2)
        self.to_output   = nn.Linear(d_model2, output_dim)

        # ── Causal mask (precomputed, shared between the two stages) ──
        # PyTorch convention: additive mask, -inf = blocked, 0 = allowed.
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
        # Crop if a shorter sequence is passed at inference
        return self.causal_mask[:L, :L]

    def forward(
        self, x: torch.Tensor, return_stress: bool = False
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, L, input_dim). The last `n_pft` channels must be the
            PFT fractions; everything before is weather + derived features.
        return_stress : bool
            If True, also return the intermediate stress tensor
            (B, L, stress_dim) for inspection / regularisation.

        Returns
        -------
        out : (B, L, output_dim)
        stress : (B, L, stress_dim), optional, when return_stress=True
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
            pft     = x[:, :, -self.n_pft :]  # (B, L, n_pft) — constant in time
        else:
            # n_pft=0: the whole input is weather, stage 2 is not PFT-conditioned
            # (avoids the x[..., :-0] == empty-slice pitfall).
            weather = x
            pft     = x[:, :, :0]             # (B, L, 0)

        # ── Stage 1: weather → stress ──
        h = weather.transpose(1, 2)                  # (B, n_weather, L) for Conv1d
        h = self.weather_conv(h).transpose(1, 2)     # (B, L, d_model)
        h = self.pos_enc1(h)
        h = self.norm_emb1(h)
        h = self.drop_emb1(h)
        h = self.stage1(h, mask=mask)                # (B, L, d_model)
        h = self.stage1_norm(h)
        stress = self.to_stress(h)                   # (B, L, stress_dim)

        # ── Stage 2: stress + PFT → LAI ──
        y = torch.cat([stress, pft], dim=-1)          # (B, L, stress_dim + n_pft)
        y = self.fuse_proj(y)                         # (B, L, d_model)
        y = self.pos_enc2(y)
        y = self.norm_emb2(y)
        y = self.drop_emb2(y)
        y = self.stage2(y, mask=mask)                # (B, L, d_model)
        y = self.stage2_norm(y)
        out = self.to_output(y)                       # (B, L, output_dim)

        if return_stress:
            return out, stress
        return out
