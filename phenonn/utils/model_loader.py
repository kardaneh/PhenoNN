"""
Build the trainable model from a CLI args namespace.

Two factories:
  - `build_model(args)`       : base model + Every10DaysWrapper  → (B, 1, 36)
  - `build_model_pft(args, norm_stats)` : base model with 15 output channels
                                + PFTMixingWrapper → (B, 1, 36)

Both end up with the same output shape so the rest of the pipeline (loss,
train loop, validation) is identical regardless of the mixing mode.
"""

import torch.nn as nn

from phenonn.utils.config import FEATURE_CHANNELS, N_PFT, PFT_START
from phenonn.models.rnn import RNN_LSTM, RNN_GRU
from phenonn.models.transformer import EncoderTorch, CausalSparseTransformer
from phenonn.models.transformerbis import BiTransformer
from phenonn.models.bitransformer import BiTransformerV2
from phenonn.models.attn_lstm import AttnLSTM
from phenonn.models.aelstm import AELSTM
from phenonn.models.fcn import FCN
from phenonn.models.linear_baseline import LinearBaseline, PerDayLinearBaseline
from phenonn.utils.wrappers import (
    DailyWrapper, Every10DaysWrapper, PFTMixingWrapper, permuteWrapper,
)


def _resolve_d_model(args) -> int:
    """Transformer embedding dim for bitransformer_v2 / attnlstm stage-1:
    --d_model, falling back to --hidden_size when unset (also covers old
    checkpoints saved before --d_model existed)."""
    return getattr(args, "d_model", None) or args.hidden_size


def _ff1(args) -> int:
    """Stage-1 FFN multiplier: --feed_forward_trans1, falling back to the old
    --feed_forward_trans name (old checkpoints)."""
    v = getattr(args, "feed_forward_trans1", None)
    return v if v is not None else getattr(args, "feed_forward_trans", 4)


def _ff2(args) -> int:
    """Stage-2 FFN multiplier: --feed_forward_trans2, falling back to the old
    --feed_forward_encoder name (old checkpoints)."""
    v = getattr(args, "feed_forward_trans2", None)
    return v if v is not None else getattr(args, "feed_forward_encoder", 4)


def _nl1(args) -> int:
    """Stage-1 transformer layer count: --num_layers1, falling back to the old
    --bitrans_stage1_layers name (old checkpoints)."""
    v = getattr(args, "num_layers1", None)
    return v if v is not None else getattr(args, "bitrans_stage1_layers", 2)


def _do1(args) -> float:
    """Stage-1 (transformer) dropout: --dropout1, falling back to the old
    --dropout_trans name (old checkpoints)."""
    v = getattr(args, "dropout1", None)
    return v if v is not None else getattr(args, "dropout_trans", 0.0)


def _do2(args) -> float:
    """Generic / stage-2 dropout: --dropout2, falling back to the old
    --dropout name (old checkpoints)."""
    v = getattr(args, "dropout2", None)
    return v if v is not None else getattr(args, "dropout", 0.0)


# ── Standard (single-output) factory ────────────────────────────────────────


def build_model(args) -> nn.Module:
    """
    Build a base model with output_channel=1 and wrap it so the final shape
    is (B, 1, 36). Used when --pft_mixing is OFF.
    """
    t = args.type.lower()

    if t == "linear":
        return LinearBaseline(
            feature_channel=FEATURE_CHANNELS,
            seq_length=args.seq_length,
        )
    if t == "linear_perday":
        return PerDayLinearBaseline(feature_channel=FEATURE_CHANNELS)

    if t == "lstm":
        base = RNN_LSTM(
            feature_channel=FEATURE_CHANNELS, output_channel=1,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
        )
    elif t == "gru":
        base = RNN_GRU(
            feature_channel=FEATURE_CHANNELS, output_channel=1,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
        )
    elif t == "transformer":
        base = EncoderTorch(
            feature_channel=FEATURE_CHANNELS, output_channel=1,
            embed_size=args.embed_size, num_layers=args.num_layers,
            heads=args.nhead, forward_expansion=args.forward_expansion,
            seq_length=args.seq_length, dropout=_do2(args), causal=False,
        )
    elif t == "bitransformer":
        base = permuteWrapper(BiTransformer(
            input_dim=FEATURE_CHANNELS, d_model=args.hidden_size,
            feed_forward_trans=_ff1(args),
            feed_forward_encoder=_ff2(args),
            output_dim=1, nr_blocks=args.num_layers,
            dropout_trans=_do1(args), dropout_encoder=_do2(args),
            n_pft=N_PFT,
        ))
    elif t == "aelstm":
        base = AELSTM(
            feature_channel=FEATURE_CHANNELS, output_channel=1,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            n_attn_blocks=args.n_attn_blocks, nhead=args.nhead,
            ff_expansion=args.forward_expansion,
            dropout=_do2(args), dropout_att=args.dropout_att,
            seq_length=args.seq_length,
        )
    elif t == "transformer_dec":
        # CausalSparseTransformer outputs (B, 1, 36) directly — bypass wrapper
        return CausalSparseTransformer(
            feature_channel=FEATURE_CHANNELS, output_channel=1,
            d_model=args.hidden_size, n_heads=args.nhead,
            e_layers=args.num_layers, d_layers=args.d_layers,
            d_ff=args.forward_expansion * args.hidden_size,
            seq_length=args.seq_length, dropout=_do2(args),
            causal=True, num_obs_days=36,
        )
    elif t == "bitransformer_v2":
        base = permuteWrapper(BiTransformerV2(
            input_dim=FEATURE_CHANNELS, output_dim=1,
            d_model=_resolve_d_model(args), d_model2=args.hidden_size,
            n_pft=N_PFT,
            stress_dim=args.stress_dim,
            nr_blocks_stage1=_nl1(args),
            nr_blocks_stage2=args.num_layers,
            nhead=args.nhead,
            feed_forward_trans=_ff1(args),
            feed_forward_encoder=_ff2(args),
            dropout_trans=_do1(args), dropout_encoder=_do2(args),
            seq_length=args.seq_length, causal=True,
        ))
    elif t == "attnlstm":
        base = permuteWrapper(AttnLSTM(
            input_dim=FEATURE_CHANNELS, output_dim=1,
            d_model=_resolve_d_model(args), lstm_hidden=args.hidden_size,
            n_pft=N_PFT,
            stress_dim=args.stress_dim,
            nr_blocks_stage1=_nl1(args),
            lstm_layers=args.num_layers,
            nhead=args.nhead,
            feed_forward_trans=_ff1(args),
            dropout_trans=_do1(args), dropout_lstm=_do2(args),
            seq_length=args.seq_length, causal=True,
        ))
    elif t in ("fcn", "fullyconnected"):
        base = FCN(
            feature_channel=FEATURE_CHANNELS, output_channel=1,
            num_layers=args.num_layers, hidden_size=args.hidden_size,
            seq_length=args.seq_length, dim_expand=0,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.type}")

    if getattr(args, "daily_lai", False):
        return DailyWrapper(base)                 # (B, 1, 365) — daily target
    return Every10DaysWrapper(base)


# ── PFT-decomposed factory ──────────────────────────────────────────────────


def build_model_pft(args, norm_stats) -> nn.Module:
    """
    Build a base model with output_channel=N_PFT (= 15) and wrap it in
    PFTMixingWrapper. Identical to LaiNN/phenocam/main_bis.build_model_pft.

    With `args.pft_meteo_only`, the base model sees ONLY the meteo/cyclic/co2
    channels (feature_channel=PFT_START, no trailing PFT input): each pure-LAI
    output L_k then depends on climate alone, and the PFT fractions enter only
    as mixing weights inside PFTMixingWrapper. For the BiTransformers this means
    n_pft=0 (no PFT concat in their conditioning stage).
    """
    t = args.type.lower()
    meteo_only = bool(getattr(args, "pft_meteo_only", False))
    nonneg = bool(getattr(args, "pft_nonneg", False))
    base_channels = PFT_START if meteo_only else FEATURE_CHANNELS
    n_pft_in = 0 if meteo_only else N_PFT     # trailing PFT channels the base sees

    if t == "lstm":
        base = RNN_LSTM(
            feature_channel=base_channels, output_channel=N_PFT,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
        )
    elif t == "gru":
        base = RNN_GRU(
            feature_channel=base_channels, output_channel=N_PFT,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
        )
    elif t == "transformer":
        base = EncoderTorch(
            feature_channel=base_channels, output_channel=N_PFT,
            embed_size=args.embed_size, num_layers=args.num_layers,
            heads=args.nhead, forward_expansion=args.forward_expansion,
            seq_length=args.seq_length, dropout=_do2(args), causal=False,
        )
    elif t == "bitransformer":
        base = permuteWrapper(BiTransformer(
            input_dim=base_channels, d_model=args.hidden_size,
            feed_forward_trans=_ff1(args),
            feed_forward_encoder=_ff2(args),
            output_dim=N_PFT, nr_blocks=args.num_layers,
            dropout_trans=_do1(args), dropout_encoder=_do2(args),
            n_pft=n_pft_in,
        ))
    elif t == "aelstm":
        base = AELSTM(
            feature_channel=base_channels, output_channel=N_PFT,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            n_attn_blocks=args.n_attn_blocks, nhead=args.nhead,
            ff_expansion=args.forward_expansion,
            dropout=_do2(args), dropout_att=args.dropout_att,
            seq_length=args.seq_length,
        )
    elif t == "transformer_dec":
        base = CausalSparseTransformer(
            feature_channel=base_channels, output_channel=N_PFT,
            d_model=args.hidden_size, n_heads=args.nhead,
            e_layers=args.num_layers, d_layers=args.d_layers,
            d_ff=args.forward_expansion * args.hidden_size,
            seq_length=args.seq_length, dropout=_do2(args),
            causal=True, num_obs_days=36,
        )
        return PFTMixingWrapper(
            base, norm_stats, sparse_output=True, meteo_only=meteo_only,
            nonneg=nonneg,
            lai_normalized=getattr(args, "normalize_lai", True),
        )
    elif t == "bitransformer_v2":
        base = permuteWrapper(BiTransformerV2(
            input_dim=base_channels, output_dim=N_PFT,
            d_model=_resolve_d_model(args), d_model2=args.hidden_size,
            n_pft=n_pft_in,
            stress_dim=args.stress_dim,
            nr_blocks_stage1=_nl1(args),
            nr_blocks_stage2=args.num_layers,
            nhead=args.nhead,
            feed_forward_trans=_ff1(args),
            feed_forward_encoder=_ff2(args),
            dropout_trans=_do1(args), dropout_encoder=_do2(args),
            seq_length=args.seq_length, causal=True,
        ))
    elif t == "attnlstm":
        base = permuteWrapper(AttnLSTM(
            input_dim=base_channels, output_dim=N_PFT,
            d_model=_resolve_d_model(args), lstm_hidden=args.hidden_size,
            n_pft=n_pft_in,
            stress_dim=args.stress_dim,
            nr_blocks_stage1=_nl1(args),
            lstm_layers=args.num_layers,
            nhead=args.nhead,
            feed_forward_trans=_ff1(args),
            dropout_trans=_do1(args), dropout_lstm=_do2(args),
            seq_length=args.seq_length, causal=True,
        ))
    else:
        raise ValueError(f"Unsupported model type for PFT mixing: {args.type}")

    return PFTMixingWrapper(
        base, norm_stats, meteo_only=meteo_only, nonneg=nonneg,
        lai_normalized=getattr(args, "normalize_lai", True),
        daily=getattr(args, "daily_lai", False),
    )
