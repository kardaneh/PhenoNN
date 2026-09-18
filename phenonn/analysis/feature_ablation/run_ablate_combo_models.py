#!/usr/bin/env python3
"""
run_ablate_combo_models.py — one extra GPU: lstm & bitransformer_v2 at two sizes
each, all trained with the COMBO ablation (tp_sum + ssrd_sum + strd_sum +
VPD_max + Tmean removed), on ONE shared dataset load.

Complements run_ablate_inproc (which fixes the model and varies the ablated
group): here the ablation is fixed (the combo) and the model/size vary. Same
mean-imputation hook, same no-core-change design.

    python -m phenonn.analysis.feature_ablation.run_ablate_combo_models \\
        <train_full_ram flags: --loss_type mse --patience 7 --pft_mixing …>
"""

import copy
import os

import numpy as np
import torch

from phenonn.training import train_full_ram as T
from phenonn.utils.config import ALL_FEATURES
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils
from phenonn.analysis.feature_ablation.run_ablate_inproc import (
    COMBO_FEATURES, _install, _restore,
)

# (model, hidden_size, stress_dim|None) — two sizes each for lstm & bitransformer.
CONFIGS = [
    ("lstm",             256, None),
    ("lstm",             512, None),
    ("bitransformer_v2", 256, 1),
    ("bitransformer_v2", 512, 1),
]


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    base_args = T.parse_args()                     # standard train_full_ram CLI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, "ablate_combo_models")
    FileUtils.makedir(sweep_dir)
    logger = Logger(console_output=True, file_output=True,
                    log_file=os.path.join(sweep_dir, "load.log"))
    logger.show_header(f"PhenoNN combo-ablation model/size — {len(CONFIGS)} configs")

    channels = [ALL_FEATURES.index(f) for f in COMBO_FEATURES]
    logger.info(f"Combo removed   : {COMBO_FEATURES}  channels={channels}")

    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, logger, device)

    orig = _install(T, channels)                   # combo hook for every build
    try:
        for typ, hid, stress in CONFIGS:
            args_i = copy.deepcopy(base_args)
            args_i.type = typ
            args_i.hidden_size = hid
            if stress is not None:
                args_i.stress_dim = stress
            args_i.experiment = (f"ablate_combo_{typ}_h{hid}"
                                 + (f"_s{stress}" if stress is not None else ""))

            done = os.path.exists(os.path.join(args_i.output_dir,
                                               args_i.experiment, "config.json"))
            if done:
                logger.info(f"[skip] {args_i.experiment} — already completed.")
                continue

            log_dir = os.path.join(args_i.output_dir, args_i.experiment, "logs")
            FileUtils.makedir(log_dir)
            cfg_logger = Logger(console_output=True, file_output=True,
                                log_file=os.path.join(log_dir, "train.log"))
            cfg_logger.show_header(f"combo ablation {typ} h{hid}"
                                   + (f" s{stress}" if stress is not None else ""))
            _seed_all(args_i.seed)
            T.train_one_config(args_i, shared, cfg_logger)
    finally:
        _restore(T, orig)


if __name__ == "__main__":
    main()
