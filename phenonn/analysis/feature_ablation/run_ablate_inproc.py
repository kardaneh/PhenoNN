#!/usr/bin/env python3
"""
run_ablate_inproc.py — in-process GROUP feature-ablation sweep.

Leave-one-GROUP-out: physically related features are removed together (the two
radiation fluxes, the three temperatures, VPD max+mean, …), one group per run,
plus a baseline. Like run_sweep_inproc, it loads the RAM dataset ONCE and trains
several configs sequentially on the same GPU — pass a slice with --indices so a
job does 3 runs/GPU.

Ablation = mean-imputation: the z-scored input channels of the group are zeroed
before every forward (their mean is 0 → the channel carries no information), so
ΔR² vs the baseline measures that group's contribution. Installed as a forward
pre-hook by wrapping the two model factories in train_full_ram's namespace — no
change to phenonn/ and checkpoints stay byte-compatible (the hook adds no params).

Usage
-----
    python -m phenonn.analysis.feature_ablation.run_ablate_inproc \\
        --model aelstm --indices 0-2 \\
        <train_full_ram flags: --loss_type mse --patience 7 --pft_mixing …>

Every flag other than --model / --indices is passed to train_full_ram.parse_args.
"""

import argparse
import copy
import os

import numpy as np
import torch

from phenonn.training import train_full_ram as T
from phenonn.utils.config import ALL_FEATURES, PFT_START
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils

# Combined ablation feature set (also reused by run_ablate_combo_models.py).
COMBO_FEATURES = ["tp_sum", "ssrd_sum", "strd_sum", "VPD_max", "Tmean"]

# ── Ablation groups (name, feature list). Related meteo features go together. ──
_RAW_GROUPS = [
    ("baseline",      []),
    ("temperature",   ["Tmin", "Tmax", "Tmean"]),
    ("radiation",     ["ssrd_sum", "strd_sum"]),     # the two downward fluxes
    ("net_radiation", ["Rn_tot"]),
    ("vpd",           ["VPD_max", "VPD_mean"]),
    ("precip",        ["tp_sum"]),
    ("smi",           ["SMI"]),
    ("pet",           ["PET"]),
    ("daylength",     ["daylength"]),
    ("co2",           ["co2"]),
    # Combined ablation (removed together): precip + both radiation fluxes +
    # VPD_max + Tmean. Runs on the same GPU as co2 (last slice).
    ("combo_rad_tp_vpdmax_tmean", COMBO_FEATURES),
]
# Keep only features present in this build; drop empty non-baseline groups.
GROUPS = [(n, [f for f in fs if f in ALL_FEATURES]) for n, fs in _RAW_GROUPS]
GROUPS = [(n, fs) for n, fs in GROUPS if n == "baseline" or fs]


def _channels(feats):
    chs = []
    for f in feats:
        ch = ALL_FEATURES.index(f)
        if ch >= PFT_START:
            raise SystemExit(f"{f!r} is a PFT channel; only non-PFT features "
                             f"are ablatable.")
        chs.append(ch)
    return chs


def _install(module, channels):
    """Wrap build_model / build_model_pft in `module` so the returned model
    zeroes the given input channels before every forward. Returns the originals."""
    def hook(_mod, inputs):
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        x = x.clone()
        for ch in channels:
            x[:, ch, :] = 0.0
        return (x,) + tuple(inputs[1:])

    orig = (module.build_model, module.build_model_pft)

    def build_model(args):
        m = orig[0](args)
        m.register_forward_pre_hook(hook)
        return m

    def build_model_pft(args, norm_stats):
        m = orig[1](args, norm_stats)
        m.register_forward_pre_hook(hook)
        return m

    module.build_model, module.build_model_pft = build_model, build_model_pft
    return orig


def _restore(module, orig):
    module.build_model, module.build_model_pft = orig


def _parse_indices(spec, n):
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
    return os.path.exists(os.path.join(args_i.output_dir, args_i.experiment,
                                       "config.json"))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", default="aelstm")
    ap.add_argument("--indices", default="",
                    help="Ablation groups for THIS process: '0-2', '0,3', or "
                         "empty for all. Slice across GPUs (3/GPU).")
    known, rest = ap.parse_known_args()

    base_args = T.parse_args(rest)                 # standard train_full_ram CLI

    idxs = _parse_indices(known.indices, len(GROUPS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_dir = os.path.join(base_args.output_dir, f"ablate_{known.model}")
    FileUtils.makedir(sweep_dir)
    load_logger = Logger(console_output=True, file_output=True,
                         log_file=os.path.join(sweep_dir, "load.log"))
    load_logger.show_header(f"PhenoNN group ablation — {known.model} "
                            f"({len(idxs)}/{len(GROUPS)} groups)")

    _seed_all(base_args.seed)
    shared = T.load_shared(base_args, load_logger, device)

    for i in idxs:
        name, feats = GROUPS[i]
        channels = _channels(feats)
        args_i = copy.deepcopy(base_args)
        args_i.type = known.model
        args_i.experiment = f"ablate_{known.model}_{name}"

        if _is_done(args_i):
            load_logger.info(f"[skip] {args_i.experiment} — already completed.")
            continue

        log_dir = os.path.join(args_i.output_dir, args_i.experiment, "logs")
        FileUtils.makedir(log_dir)
        cfg_logger = Logger(console_output=True, file_output=True,
                            log_file=os.path.join(log_dir, "train.log"))
        cfg_logger.show_header(f"{known.model} ablate [{name}] "
                               f"channels={channels} feats={feats}")

        _seed_all(args_i.seed)
        orig = _install(T, channels) if channels else None
        try:
            T.train_one_config(args_i, shared, cfg_logger)
        finally:
            if orig is not None:
                _restore(T, orig)


if __name__ == "__main__":
    main()
