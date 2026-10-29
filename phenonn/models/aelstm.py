# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh, Stefan Barbu
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Attention-Enhanced LSTM (AELSTM).

An LSTM encoder followed by ``n_attn_blocks`` transformer-style
self-attention blocks, and a final 1D convolution projecting to the
desired output channels.

The intuition is that the LSTM produces a contextualised feature vector at
every timestep, and the attention layer re-weights those vectors so that
timesteps carrying the strongest predictive signal (green-up, drought,
snowmelt) contribute more to each output timestep, while uninformative
periods (mid-winter plateaus) are attenuated. Stacking several attention
blocks lets the model capture hierarchical temporal patterns with an
increasing receptive field.

Input and output shapes::

    Input  : (B, feature_channel, L)
    Output : (B, output_channel,  L)   # sequence-to-sequence, same length

The interface matches the rest of the pipeline, so the model is a drop-in
replacement for ``RNN_LSTM`` under ``Every10DaysWrapper`` and
``PFTMixingWrapper``.
"""

import torch
import torch.nn as nn

from typing import Tuple


class SelfAttentionBlock(nn.Module):
    """
    One transformer-style self-attention block with pre-norm and residuals.

    The block applies two sublayers in sequence, each preceded by
    ``LayerNorm`` and followed by a residual connection:

    1. **Multi-head self-attention**, optional additive mask.
    2. **Feed-forward network**: ``Linear -> GELU -> Linear -> Dropout``.

    Pre-norm formulation (norm before the sublayer) keeps training stable
    when several blocks are stacked.

    Parameters
    ----------
    d_model : int
        Hidden dimension. Must equal the LSTM ``hidden_size`` of the model
        that contains this block.
    nhead : int
        Number of attention heads. Must divide ``d_model``.
    ff_expansion : int
        Feed-forward inner dimension, ``ff_expansion * d_model``.
    dropout : float
        Dropout rate applied to the attention output and inside the
        feed-forward sublayer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_expansion * d_model),
            nn.GELU(),
            nn.Linear(ff_expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, L, D)``.
        attn_mask : torch.Tensor, optional
            Additive attention mask of shape ``(L, L)``. For a causal block,
            ``-inf`` above the diagonal so timestep ``t`` only attends to
            timesteps ``<= t``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, L, D)``.
        """
        # Self-attention sublayer (pre-norm + residual)
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop_attn(attn_out)

        # Feed-forward sublayer (pre-norm + residual)
        x = x + self.ff(self.norm_ff(x))
        return x


class AELSTM(nn.Module):
    """
    Attention-Enhanced LSTM for LAI sequence prediction.

    Pipeline
    --------

    The input is permuted to ``(B, L, C_in)``, passed through the LSTM to
    produce ``(B, L, H)``, then through ``n_attn_blocks`` self-attention
    blocks (still ``(B, L, H)``). The output is permuted back to
    ``(B, H, L)`` and projected to ``(B, C_out, L)`` by a ``Conv1d`` with
    ``kernel_size=1``.

    Parameters
    ----------
    feature_channel : int
        Number of input features per timestep (``C_in``).
    output_channel : int
        Number of output channels (``C_out``). 1 for the scalar model, 15
        for the PFT-decomposed setup.
    hidden_size : int
        LSTM hidden size, also used as the attention ``d_model``.
    num_layers : int
        Number of stacked LSTM layers.
    n_attn_blocks : int
        Number of self-attention blocks placed between the LSTM and the
        output projection.
    nhead : int
        Number of attention heads in each block. Must divide ``hidden_size``.
    ff_expansion : int
        Feed-forward expansion factor inside each attention block.
    dropout : float
        Dropout applied between LSTM layers. No effect when ``num_layers=1``.
    dropout_att : float
        Dropout applied inside each self-attention block (attention output
        and feed-forward sublayer).
    seq_length : int
        Maximum input length. Used to precompute the causal attention mask
        once as a buffer; cropped to the actual length in ``forward``.

    Examples
    --------
    >>> model = AELSTM(
    ...     feature_channel=31, output_channel=1,
    ...     hidden_size=128, num_layers=2,
    ...     n_attn_blocks=2, nhead=4,
    ... )
    >>> x = torch.randn(8, 31, 720)
    >>> y = model(x)
    >>> y.shape
    torch.Size([8, 1, 720])
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        hidden_size: int,
        num_layers: int,
        n_attn_blocks: int = 2,
        nhead: int = 4,
        ff_expansion: int = 4,
        dropout: float = 0.0,
        dropout_att: float = 0.0,
        seq_length: int = 720,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel

        self.lstm = nn.LSTM(
            input_size=feature_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    d_model=hidden_size,
                    nhead=nhead,
                    ff_expansion=ff_expansion,
                    dropout=dropout_att,
                )
                for _ in range(n_attn_blocks)
            ]
        )

        # Final LayerNorm after the attention stack — standard practice.
        self.norm_out = nn.LayerNorm(hidden_size)

        self.final = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=output_channel,
            kernel_size=1,
            bias=True,
        )

        # Causal attention mask, precomputed once (additive, -inf above the
        # diagonal). Non-persistent so it stays out of the state_dict and old
        # checkpoints load cleanly; cropped to the actual length in forward.
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(seq_length),
            persistent=False,
        )

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return zero hidden and cell states for the LSTM.

        Parameters
        ----------
        batch_size : int
            Batch size of the current forward pass.
        device : torch.device
            Target device for the returned tensors.

        Returns
        -------
        tuple of torch.Tensor
            ``(h_0, c_0)``, each of shape
            ``(num_layers, batch_size, hidden_size)``.
        """
        h = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            requires_grad=False,
        )
        return h, h.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, feature_channel, L)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, output_channel, L)``.
        """
        # (B, C, L) → (B, L, C) for the LSTM
        x = x.permute(0, 2, 1)

        hidden = self.init_hidden(x.size(0), x.device)
        out, _ = self.lstm(x, hidden)  # (B, L, H)

        # Causal mask so each timestep attends only to the past (and itself);
        # without it the self-attention would leak future days into day t.
        # Precomputed buffer, cropped to the actual length (and cast to the
        # current dtype, e.g. under bf16 autocast).
        causal_mask = self.causal_mask[: out.size(1), : out.size(1)]
        if causal_mask.dtype != out.dtype:
            causal_mask = causal_mask.to(out.dtype)
        for block in self.attn_blocks:
            out = block(out, attn_mask=causal_mask)  # (B, L, H)

        out = self.norm_out(out)
        out = out.permute(0, 2, 1)  # (B, H, L)
        return self.final(out)  # (B, C_out, L)
