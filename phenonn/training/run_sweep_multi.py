#!/usr/bin/env python3
"""
run_sweep_multi.py
==================

Single-load, MULTI-MODEL in-process sweep: load the nonanomaly RAM datasets
ONCE, then train every config of EVERY model on that one resident dataset.

Why
---
The nonanomaly working-set (features + targets/PFT) is IDENTICAL for all models:
load_shared() depends only on subset / seed / years / sites / n_sites_per_epoch /
num_epochs, never on the model. So one load_shared() can serve the whole 4-model
sweep (run_sweep_inproc reloads once PER model; this driver reloads once TOTAL).

Anomaly is NOT supported here (anomaly_mode changes the targets → it would need
its own load); use run_sweep_inproc --group anomaly for that.

    python -m phenonn.training.run_sweep_multi \\
        --models lstm,aelstm,attnlstm,bitransformer_v2 <train_full_ram flags>

Every flag other than --models is passed straight through to
train_full_ram.parse_args. Completed configs (their exp dir already holds the
config.json that train_one_config writes LAST) are skipped, so a re-run resumes.
"""

import argparse
import copy
import os

import numpy as np
import torch

from phenonn.training import train_full_ram as T
from phenonn.training.jobs_tuning.sweep_plan import plan
from phenonn.training.jobs_tuning.tuning_grid import MODELS
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils

# Overrides that are Namespace attributes to setattr; "name" is bookkeeping.
_META_KEYS = {"name"}
_GROUP = "nonanomaly"


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_done(args_i):
    """train_one_config writes exp_dir/config.json LAST → its presence = done."""
    return os.path.exists(os.path.join(args_i.output_dir, args_i.experiment,
                                       "config.json"))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="Comma list of models to sweep on the ONE shared load.")
    known, rest = ap.parse_known_args()
    models = [m.strip() for m in known.models.split(",") if m.strip()]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}; choose from {MODELS}")

    base_args = T.parse_args(rest)          # standard train_full_ram CLI
    if base_args.anomaly_mode:
        raise SystemExit("run_sweep_multi is nonanomaly-only (single shared "
                         "load). Use run_sweep_inproc --group anomaly instead.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, "sweep_multi_nonanomaly")
    FileUtils.makedir(sweep_dir)
    load_logger = Logger(console_output=True, file_output=True,
                         log_file=os.path.join(sweep_dir, "load.log"))
    total = sum(len(plan(m, _GROUP)) for m in models)
    load_logger.show_header(f"PhenoNN multi-model sweep — {models} "
                            f"[{_GROUP}] ({total} configs, ONE shared load)")

    # ── Load the resident datasets ONCE for ALL models ──
    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, load_logger, device)

    # ── Train every config of every model on the shared data ──
    for model in models:
        entries = plan(model, _GROUP)
        for i, entry in enumerate(entries):
            args_i = copy.deepcopy(base_args)
            args_i.type = model
            for k, v in entry.items():
                if k not in _META_KEYS:
                    setattr(args_i, k, v)
            args_i.experiment = f"{model}_{_GROUP}_a{i:02d}_{entry['name']}"

            if _is_done(args_i):
                load_logger.info(f"[skip] {args_i.experiment} — already completed.")
                continue

            log_dir = os.path.join(args_i.output_dir, args_i.experiment, "logs")
            FileUtils.makedir(log_dir)
            cfg_logger = Logger(console_output=True, file_output=True,
                                log_file=os.path.join(log_dir, "train.log"))
            cfg_logger.show_header(f"{model} [{_GROUP}] idx {i}: {entry['name']}")

            # Reset seeds before each build so model init is reproducible and
            # independent of how many configs ran before it in this process.
            _seed_all(args_i.seed)
            T.train_one_config(args_i, shared, cfg_logger)


if __name__ == "__main__":
    main()
