"""PhenoNN training CLI arguments (RAM working-set trainer).

This module is dependency-light: it imports argparse and the protocol
normaliser only, so ``phenonn train --help`` does not import torch.
"""

from __future__ import annotations

import argparse

from phenonn.evaluation.leakage import normalise_protocol


def parse_args(argv=None):
    """
    Parse the command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── Data ──
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument("--valid_dir", default="")
    p.add_argument("--selected_pixels", default="")
    p.add_argument("--parent_map", default="")
    p.add_argument("--stats_path", default="")
    p.add_argument("--co2_path", default="")
    p.add_argument(
        "--no_normalize_lai",
        dest="normalize_lai",
        action="store_false",
        default=True,
    )
    p.add_argument("--daily_lai", action="store_true", default=False)
    p.add_argument("--daily_target_dir", default="")
    p.add_argument(
        "--threaded_feature_read",
        action="store_true",
        default=False,
    )

    # ── Sub-grid ──
    p.add_argument("--row_min", type=int, default=0)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=0)
    p.add_argument("--col_max", type=int, default=-1)

    # ── Years ──
    p.add_argument("--train_years", default="")
    p.add_argument("--val_years", default="")
    p.add_argument(
        "--test_years",
        default="",
        help="Held-out test years, e.g. '2010-2019'. Required by protocol B "
        "and used only for leakage checks; never for early stopping.",
    )
    p.add_argument(
        "--protocol",
        default="none",
        type=normalise_protocol,
        choices=["A", "B", "none"],
        help="Split protocol: A = reproduction (val=test decade, biased), "
        "B = unbiased paper split, none = ad-hoc (leakage checks still run).",
    )
    p.add_argument(
        "--norm_years",
        default="",
        help="Years used to build --stats_path. Checked against train_years.",
    )
    p.add_argument(
        "--allow_leakage",
        action="store_true",
        help="Downgrade leakage errors to warnings (documented diagnostics only).",
    )
    p.add_argument(
        "--selection_rule",
        default="val_loss",
        choices=["val_loss", "val_r2"],
        help="Checkpoint selection criterion (paper: val_loss).",
    )

    # ── Per-epoch sampling ──
    p.add_argument("--n_sites_per_epoch", type=int, default=500)
    p.add_argument("--n_years_per_epoch", type=int, default=3)
    p.add_argument("--n_val_sites", type=int, default=200)
    p.add_argument("--val_fraction_of_grid", type=float, default=0.1)
    p.add_argument("--subset", type=float, default=1.0)

    # ── Anomaly mode ──
    p.add_argument("--anomaly_mode", action="store_true")
    p.add_argument("--clim_years", default="")
    p.add_argument("--clim_target_dir", default="")

    # ── Model ──
    p.add_argument(
        "--type",
        default="lstm",
        choices=[
            "lstm",
            "gru",
            "transformer",
            "transformer_dec",
            "bitransformer",
            "bitransformer_v2",
            "attnlstm",
            "aelstm",
            "fcn",
            "linear",
            "linear_perday",
        ],
    )
    p.add_argument("--pft_mixing", action="store_true")
    p.add_argument("--pft_meteo_only", action="store_true")
    p.add_argument("--pft_nonneg", action="store_true")
    p.add_argument("--seq_length", type=int, default=730)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_layers1", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout2", type=float, default=0.0)
    p.add_argument("--dropout1", type=float, default=0.0)
    p.add_argument("--dropout_att", type=float, default=0.0)
    p.add_argument("--feed_forward_trans1", type=int, default=4)
    p.add_argument("--feed_forward_trans2", type=int, default=4)
    p.add_argument("--n_attn_blocks", type=int, default=2)
    p.add_argument("--stress_dim", type=int, default=8)
    p.add_argument("--d_layers", type=int, default=2)
    p.add_argument("--embed_size", type=int, default=64)

    # ── Training ──
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument(
        "--loss_type",
        default="huber",
        choices=["mse", "mae", "huber", "smoothl1", "nmse", "nmae", "gradient"],
    )
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--gradient_loss_weight", type=float, default=0.5)
    p.add_argument("--peak_penalty_weight", type=float, default=0.0)
    p.add_argument("--corr_loss_weight", type=float, default=0.0)
    p.add_argument("--amp_loss_weight", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=11)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--no_fused_adam",
        dest="fused_adam",
        action="store_false",
        default=True,
    )

    # ── Output / resume ──
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--experiment", default="exp_big")
    p.add_argument("--prefix", default="run", help="Prefix for checkpoint filenames.")
    p.add_argument(
        "--save_checkpoint_name",
        default="model",
        help="Base name for checkpoint files.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")

    return p.parse_args(argv)
