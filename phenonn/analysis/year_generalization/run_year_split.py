#!/usr/bin/env python3
"""
run_year_split.py — train one YEAR-SPLIT of the temporal-generalization study.

Same sites, OVERLAP mode; only the train/val YEAR lists change (18 train / 10 val,
matching the chronological baseline). Splits:
  chrono         : train 1992-2009, val 2010-2019 (baseline: validate on the future)
  rand1 … rand4  : 18 random train years, the other 10 for val (seeds 1001..1004)
  coldhot        : train on the 18 coldest years, val on the 10 hottest
                   (needs year_temp_rank.json from rank_years_by_temp.py)

Compare ΔR² = R²(split) − R²(chrono) to see the effect of predicting the future
vs random years vs a hotter climate.

    python -m phenonn.analysis.year_generalization.run_year_split --split rand1 \\
        <train_full_ram flags: fixed model config, --val_fraction_of_grid 100 …>
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from phenonn.training import train_full_ram as T
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils

ALL_YEARS = list(range(1992, 2020))         # 28 years
N_TRAIN = 18                                # → 10 val, like the chrono baseline


def _rand_split(seed):
    yy = ALL_YEARS[:]
    random.Random(seed).shuffle(yy)
    return sorted(yy[:N_TRAIN]), sorted(yy[N_TRAIN:])


def resolve_split(name, temp_rank_json):
    if name == "chrono":
        return list(range(1992, 2010)), list(range(2010, 2020))
    if name.startswith("rand") and name[4:].isdigit():
        return _rand_split(1000 + int(name[4:]))
    if name == "coldhot":
        if not os.path.exists(temp_rank_json):
            raise SystemExit(f"coldhot needs {temp_rank_json!r} — run "
                             f"rank_years_by_temp.py first.")
        with open(temp_rank_json) as f:
            order = json.load(f)["years_cold_to_hot"]
        return sorted(order[:N_TRAIN]), sorted(order[N_TRAIN:])
    raise SystemExit(f"unknown split {name!r} "
                     f"(chrono | rand1..4 | coldhot)")


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--split", required=True)
    ap.add_argument("--temp_rank_json", default="year_temp_rank.json")
    known, rest = ap.parse_known_args()

    tr, va = resolve_split(known.split, known.temp_rank_json)
    args = T.parse_args(rest)
    args.train_years = ",".join(map(str, tr))
    args.val_years = ",".join(map(str, va))
    args.experiment = f"yeargen_{known.split}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    exp_dir = os.path.join(args.output_dir, args.experiment)
    FileUtils.makedir(os.path.join(exp_dir, "logs"))
    logger = Logger(console_output=True, file_output=True,
                    log_file=os.path.join(exp_dir, "logs", "train.log"))
    logger.show_header(f"Year-split [{known.split}] — {args.type}")
    logger.info(f"Train years : {args.train_years}")
    logger.info(f"Val years   : {args.val_years}")

    if os.path.exists(os.path.join(exp_dir, "config.json")):
        logger.info(f"[skip] {args.experiment} — already completed.")
        return

    _seed_all(args.seed)
    shared = T.load_shared(args, logger, device)
    T.train_one_config(args, shared, logger)


if __name__ == "__main__":
    main()
