#!/usr/bin/env python3
"""
run_sweep_inproc.py
===================

In-process hyperparameter sweep: load the RAM datasets ONCE, then train every
plan entry of one (model, group) slice in the SAME process, reusing the
resident data.

Why
---
`train_full_ram` loads the working-set (features + targets/PFT) into RAM at
start-up — the slowest part of a run (dominated by a per-year full-coordinate
scan, independent of the subset). Launching one process per config re-pays it
every time. The sweep plan (sweep_plan.py) varies only lr / hidden_size /
num_layers / dropout / pft_mixing — none of which change the loaded dataset — so
we call `train_full_ram.load_shared` once and hand the same `SharedData` to
`train_full_ram.train_one_config` for each config.

Load groups
-----------
anomaly_mode DOES change the dataset (targets become anomalies), so it cannot be
mixed into one shared load. The plan is therefore split into groups:
  --group nonanomaly : run WITHOUT --anomaly_mode (main mixing + no-mixing).
  --group anomaly    : run WITH --anomaly_mode --clim_years <train years>.
This driver enforces that --group matches the CLI --anomaly_mode.

Trade-off: configs run SEQUENTIALLY on one GPU. Give each GPU a slice with
--indices (submit_sweep.sh does this) to keep parallelism across GPUs while
amortising the load per GPU. Completed configs (their exp dir already holds the
config.json that train_one_config writes LAST) are skipped, so a requeued job
resumes where it stopped instead of retraining.

Usage
-----
    python -m phenonn.training.run_sweep_inproc --model lstm --group nonanomaly \\
        --indices 0-8 <all the usual train_full_ram flags>

Every flag other than --model / --group / --indices is passed straight through
to train_full_ram.parse_args.
"""

import argparse
import copy
import os

import numpy as np
import torch

from phenonn.training import train_full_ram as T
from phenonn.training.jobs_tuning.sweep_plan import GROUPS, plan
from phenonn.training.jobs_tuning.tuning_grid import MODELS
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils

# Overrides that are Namespace attributes to setattr; everything else in a plan
# entry ("name") is bookkeeping, not an arg.
_META_KEYS = {"name"}


def _parse_indices(spec, n):
    """'' → all 0..n-1; '0-8' → inclusive range; '0,3,7' → explicit list."""
    if not spec:
        return list(range(n))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        idxs = list(range(int(a), int(b) + 1))
    else:
        idxs = [int(x) for x in spec.split(",")]
    for i in idxs:
        if not (0 <= i < n):
            raise SystemExit(f"index {i} out of range 0..{n - 1}")
    return idxs


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
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--group", required=True, choices=GROUPS)
    ap.add_argument("--indices", default="",
                    help="Plan entries for THIS process: '0-8', '0,3,7', or "
                         "empty for all. Slice across GPUs to keep parallelism.")
    known, rest = ap.parse_known_args()

    base_args = T.parse_args(rest)          # standard train_full_ram CLI

    # The group must match the CLI: anomaly plan ⇒ --anomaly_mode load, and only
    # that group may carry it (else the shared load would be wrong for the runs).
    if known.group in ("anomaly", "anom_loss", "anom_loss_nomix") and not base_args.anomaly_mode:
        raise SystemExit(f"--group {known.group} requires --anomaly_mode (and "
                         "--clim_years on the train years) on the CLI.")
    if known.group in ("nonanomaly", "big", "loss", "daily") and base_args.anomaly_mode:
        raise SystemExit(f"--group {known.group} must NOT be run with --anomaly_mode.")

    entries = plan(known.model, known.group)
    idxs = _parse_indices(known.indices, len(entries))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, f"sweep_{known.model}_{known.group}")
    FileUtils.makedir(sweep_dir)
    load_logger = Logger(console_output=True, file_output=True,
                         log_file=os.path.join(sweep_dir, "load.log"))
    load_logger.show_header(f"PhenoNN in-process sweep — {known.model} "
                            f"[{known.group}] ({len(idxs)}/{len(entries)} configs)")

    # ── Load the resident datasets ONCE for this group ──
    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, load_logger, device)

    # ── Train each plan entry on the shared data ──
    for i in idxs:
        entry = entries[i]
        args_i = copy.deepcopy(base_args)
        args_i.type = known.model
        for k, v in entry.items():
            if k not in _META_KEYS:
                setattr(args_i, k, v)
        args_i.experiment = f"{known.model}_{known.group}_a{i:02d}_{entry['name']}"

        if _is_done(args_i):
            load_logger.info(f"[skip] {args_i.experiment} — already completed.")
            continue

        log_dir = os.path.join(args_i.output_dir, args_i.experiment, "logs")
        FileUtils.makedir(log_dir)
        cfg_logger = Logger(console_output=True, file_output=True,
                            log_file=os.path.join(log_dir, "train.log"))
        cfg_logger.show_header(f"{known.model} [{known.group}] idx {i}: {entry['name']}")

        # Reset seeds before each build so model init is reproducible and
        # independent of how many configs ran before it in this process.
        _seed_all(args_i.seed)
        T.train_one_config(args_i, shared, cfg_logger)


if __name__ == "__main__":
    main()
