#!/usr/bin/env python3
"""
run_daily_extra.py — extra DAILY runs with explicit per-config overrides.

Unlike the grid-based run_sweep_inproc, each config here is a full dict of
train_full_ram-arg overrides, so we can vary anything (hidden_size, stress_dim,
the dropouts, n_attn_blocks, num_layers1, nhead, pft_mixing, …). One GPU = one
GROUP (2–3 configs), trained sequentially on ONE shared daily load.

Groups (all daily, loss mse, patience 7 — set in the launcher COMMON):
  big_aelstm / big_bitr   : larger models, dropout 0.2 everywhere.
  nomix_aelstm / nomix_bitr : same but pft_mixing OFF.
  params_aelstm / params_bitr : vary the other architecture params.

    python -m phenonn.training.run_daily_extra --group big_aelstm \\
        <train_full_ram flags incl. --daily_lai --daily_target_dir …>
"""

import argparse
import copy
import os

import numpy as np
import torch

from phenonn.training import train_full_ram as T
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils
# Reuse the (tested) mean-imputation ablation hook — single source of truth.
from phenonn.analysis.feature_ablation.run_ablate_inproc import (
    _channels, _install, _restore,
)

# Each entry: "name" + any train_full_ram arg overrides. pft_mixing/pft_meteo_only
# default ON via the launcher COMMON; the nomix groups turn mixing OFF per config.
GROUPS = {
    # ── Larger models, dropout 0.2 everywhere ──
    "big_aelstm": [
        {"name": "big_aelstm_h768_do0.2",  "type": "aelstm",
         "hidden_size": 768,  "dropout2": 0.2, "dropout_att": 0.2},
        {"name": "big_aelstm_h1024_do0.2", "type": "aelstm",
         "hidden_size": 1024, "dropout2": 0.2, "dropout_att": 0.2},
    ],
    "big_bitr": [
        {"name": "big_bitr_h768_s1_do0.2",  "type": "bitransformer_v2",
         "hidden_size": 768,  "stress_dim": 1, "dropout1": 0.2, "dropout2": 0.2},
        {"name": "big_bitr_h1024_s1_do0.2", "type": "bitransformer_v2",
         "hidden_size": 1024, "stress_dim": 1, "dropout1": 0.2, "dropout2": 0.2},
    ],
    # ── Without PFT mixing ──
    "nomix_aelstm": [
        {"name": "nomix_aelstm_h256", "type": "aelstm",
         "hidden_size": 256, "pft_mixing": False},
        {"name": "nomix_aelstm_h512", "type": "aelstm",
         "hidden_size": 512, "pft_mixing": False},
    ],
    "nomix_bitr": [
        {"name": "nomix_bitr_h256_s1", "type": "bitransformer_v2",
         "hidden_size": 256, "stress_dim": 1, "pft_mixing": False},
        {"name": "nomix_bitr_h512_s1", "type": "bitransformer_v2",
         "hidden_size": 512, "stress_dim": 1, "pft_mixing": False},
    ],
    # ── hal (1 GPU) : aelstm puis bitransformer_v2, sur UN seul chargement ──
    "hal_pair": [
        {"name": "hal_aelstm_h256",   "type": "aelstm",
         "hidden_size": 256, "dropout2": 0.1},
        {"name": "hal_bitr_h256_s1",  "type": "bitransformer_v2",
         "hidden_size": 256, "stress_dim": 1, "dropout1": 0.1, "dropout2": 0.1},
    ],
    # ── Vary the other architecture params ──
    "params_aelstm": [
        {"name": "params_aelstm_ab1",     "type": "aelstm",
         "hidden_size": 256, "n_attn_blocks": 1},
        {"name": "params_aelstm_ab3",     "type": "aelstm",
         "hidden_size": 256, "n_attn_blocks": 3},
        {"name": "params_aelstm_nh8_ff8", "type": "aelstm",
         "hidden_size": 256, "nhead": 8, "forward_expansion": 8},
    ],
    "params_bitr": [
        {"name": "params_bitr_nl1_1",   "type": "bitransformer_v2",
         "hidden_size": 256, "stress_dim": 1, "num_layers1": 1},
        {"name": "params_bitr_nl1_3",   "type": "bitransformer_v2",
         "hidden_size": 256, "stress_dim": 1, "num_layers1": 3},
        {"name": "params_bitr_nh8_ff8", "type": "bitransformer_v2",
         "hidden_size": 256, "stress_dim": 1, "nhead": 8, "feed_forward_trans1": 8},
    ],
}


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--ablate_features", default="",
                    help="Comma list of NON-PFT features to remove from the input "
                         "(mean-imputation: their z-scored channel is zeroed). "
                         "Applies to every config of the group; the experiment "
                         "name gets a '_no_…' suffix so results never collide "
                         "with the non-ablated runs.")
    known, rest = ap.parse_known_args()
    configs = GROUPS[known.group]

    feats = [f.strip() for f in known.ablate_features.split(",") if f.strip()]
    channels = _channels(feats) if feats else []
    tag = ("_no_" + "_".join(f.replace("_sum", "") for f in feats)) if feats else ""

    base_args = T.parse_args(rest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, f"daily_extra_{known.group}")
    FileUtils.makedir(sweep_dir)
    load_logger = Logger(console_output=True, file_output=True,
                         log_file=os.path.join(sweep_dir, "load.log"))
    load_logger.show_header(f"PhenoNN daily-extra — {known.group} "
                            f"({len(configs)} configs)")
    if feats:
        load_logger.info(f"Ablated features: {feats}  channels={channels}")

    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, load_logger, device)

    # One ablation for the whole group (no-op when --ablate_features is empty).
    orig = _install(T, channels) if channels else None
    try:
        _train_group(configs, base_args, known.group, tag, load_logger, shared,
                     ablate=",".join(feats))
    finally:
        if orig is not None:
            _restore(T, orig)


def _train_group(configs, base_args, group, tag, load_logger, shared, ablate=""):
    for entry in configs:
        args_i = copy.deepcopy(base_args)
        for k, v in entry.items():
            if k != "name":
                setattr(args_i, k, v)
        # Recorded in config.json AND the checkpoint's "args" so inference
        # (predict / perturbation) can re-apply the SAME ablation.
        args_i.ablate_features = ablate
        args_i.experiment = f"dailyx_{entry['name']}{tag}"

        if os.path.exists(os.path.join(args_i.output_dir, args_i.experiment,
                                       "config.json")):
            load_logger.info(f"[skip] {args_i.experiment} — already completed.")
            continue

        log_dir = os.path.join(args_i.output_dir, args_i.experiment, "logs")
        FileUtils.makedir(log_dir)
        cfg_logger = Logger(console_output=True, file_output=True,
                            log_file=os.path.join(log_dir, "train.log"))
        cfg_logger.show_header(f"[{group}] {args_i.experiment}")
        _seed_all(args_i.seed)
        T.train_one_config(args_i, shared, cfg_logger)


if __name__ == "__main__":
    main()
