#!/usr/bin/env python3
"""
sweep_plan.py
=============

Per-model experiment plans for the in-process sweep (run_sweep_inproc.py):
a larger, "go-bigger" grid than tuning_grid.py, split into two LOAD GROUPS
because the RAM dataset differs by anomaly_mode (see run_sweep_inproc):

  • group "nonanomaly"  — the MAIN pft_mixing sweep + a few no-mixing baselines.
    All share ONE load (pft_mixing on/off leaves the dataset identical).
  • group "anomaly"     — a few pft_mixing configs to be run WITH --anomaly_mode
    (clim on the train years); a SEPARATE load (targets become anomalies).

Each plan entry is a dict of Namespace-attribute overrides applied on top of the
base train_full_ram args (learning_rate / hidden_size / num_layers / dropout*),
plus the model-side switches pft_mixing / pft_meteo_only. anomaly_mode itself is
NOT in the dict — it is set once per invocation on the CLI (so the whole load
group is homogeneous). Dropout wiring reuses tuning_grid.config_overrides so the
per-model rule stays single-source.

Torch-free (argparse + tuning_grid only), callable as a file from bash.

    python sweep_plan.py --model lstm --group nonanomaly --count      # -> 33
    python sweep_plan.py --model aelstm --group anomaly --list        # names
"""

import argparse

try:                                              # imported as a package module
    from .tuning_grid import MODELS, config_overrides
except ImportError:                               # run as a plain file (bash)
    from tuning_grid import MODELS, config_overrides


LR = [3e-4, 1e-3, 3e-3]
SIZE = [128, 256, 512]            # includes the larger 512
DEPTH_LSTM = [2, 3, 4]            # lstm has no dropout axis → deeper instead
DEPTH_DROP = [2, 3]
DROPOUT = [0.0, 0.2]
DROPOUT_FIX = 0.1                 # frozen dropout for the stress-dim sweep (below)
STRESS = [1, 4, 8]                # --stress_dim, only for the two models below
_STRESS_MODELS = ("bitransformer_v2", "attnlstm")

BIG_SIZE = [512, 768, 1024]       # "big model" group: 3 large hidden sizes / model

GROUPS = ["nonanomaly", "anomaly", "anom_loss", "anom_loss_nomix",
          "big", "loss", "daily", "daily_cmp"]


def _main_combos(model):
    """(lr, size, depth, drop|None, stress|None) tuples of the main mixing grid."""
    if model == "lstm":
        return [(lr, s, d, None, None)
                for lr in LR for s in SIZE for d in DEPTH_LSTM]
    if model in _STRESS_MODELS:
        # Sweep stress instead of dropout on these (larger) models: dropout is
        # frozen at DROPOUT_FIX so the stress axis doesn't multiply the count.
        return [(lr, s, d, DROPOUT_FIX, st)
                for lr in LR for s in SIZE for d in DEPTH_DROP for st in STRESS]
    return [(lr, s, d, dr, None)                 # aelstm: dropout, no stress axis
            for lr in LR for s in SIZE for d in DEPTH_DROP for dr in DROPOUT]


def _curated_combos(model):
    """Small curated set reused for the no-mixing and anomaly blocks (stress at
    the model default → not swept here)."""
    dr = None if model == "lstm" else 0.1
    return [(lr, s, 2, dr, None) for lr in LR for s in [256, 512]]


def _big_combos(model):
    """3 LARGE-model configs (hidden 512/768/1024), fixed lr/depth/dropout, stress
    at 8 for the transformers — for the exclusive-node 'big' runs."""
    dr = None if model == "lstm" else 0.1
    st = 8 if model in _STRESS_MODELS else None
    return [(1e-3, s, 3, dr, st) for s in BIG_SIZE]


def _anom_loss_plan(model, pft_mixing):
    """4 configs/GPU for the anomaly loss sweep: lr 1e-3, depth 2,
    hidden {256, 512} × loss_type {huber, mse} (dropout/stress at the per-model
    default). pft_mixing selects the mixing vs no-mixing GPU."""
    dr = None if model == "lstm" else DROPOUT_FIX
    st = 8 if model in _STRESS_MODELS else None
    tag = "anomloss" if pft_mixing else "anomlossnomix"
    do = f"_do{dr:g}" if dr is not None else ""
    ss = f"_s{st}" if st is not None else ""
    out = []
    for s in [256, 512]:
        for loss in ["huber", "mse"]:
            ov = config_overrides(model, 1e-3, s, 2, dr, st)
            ov["pft_mixing"] = pft_mixing
            ov["pft_meteo_only"] = pft_mixing
            ov["loss_type"] = loss
            ov["name"] = f"{tag}_{loss}_h{s}_d2{do}{ss}"
            out.append(ov)
    return out


def _daily9_combos(model):
    """9 SIZE-varying configs per model for the daily-target sweep: the 3×3 grid
    hidden_size {128,256,512} × depth {2,3,4}, at fixed lr 1e-3 (dropout / stress
    at the per-model default). 9 configs = 3 per GPU × 3 GPUs per model."""
    dr = None if model == "lstm" else DROPOUT_FIX
    st = 8 if model in _STRESS_MODELS else None
    return [(1e-3, s, d, dr, st) for s in SIZE for d in DEPTH_LSTM]


def loss_plan(model):
    """8 configs exploring LOSS functions from a huber baseline (+ a hidden-size
    variant), at lr 1e-3 / depth 3. Run 2 per GPU across the 4 GPUs of a node
    (run_loss.sh). Loss keys (loss_type, corr_loss_weight, amp_loss_weight,
    peak_penalty_weight) are extra Namespace overrides on top of the base config."""
    dr = None if model == "lstm" else 0.1
    st = 8 if model in _STRESS_MODELS else None

    def mk(hidden, loss_extra, tag):
        ov = config_overrides(model, 1e-3, hidden, 3, dr, st)
        ov.update(pft_mixing=True, pft_meteo_only=True)
        ov.update(loss_extra)
        ov["name"] = tag
        return ov

    return [
        mk(256, {"loss_type": "huber"}, "base_huber_h256"),                 # baseline
        mk(256, {"loss_type": "mse"}, "mse_h256"),
        mk(256, {"loss_type": "mae"}, "mae_h256"),
        mk(256, {"loss_type": "smoothl1"}, "smoothl1_h256"),
        mk(256, {"loss_type": "huber", "corr_loss_weight": 0.5},            # + shape/phase
           "huber_corr_h256"),
        mk(256, {"loss_type": "huber", "amp_loss_weight": 0.5},             # + amplitude
           "huber_amp_h256"),
        mk(256, {"loss_type": "huber", "peak_penalty_weight": 0.5},         # + peak
           "huber_peak_h256"),
        mk(512, {"loss_type": "huber"}, "huber_h512"),                      # hidden variant
    ]


def _entry(model, lr, s, d, dr, st, *, pft_mixing, tag):
    ov = config_overrides(model, lr, s, d, dr, st)
    ov["pft_mixing"] = pft_mixing
    ov["pft_meteo_only"] = pft_mixing        # meteo_only only meaningful with mixing
    do = f"_do{dr:g}" if dr is not None else ""
    ss = f"_s{st}" if st is not None else ""
    ov["name"] = f"{tag}_lr{lr:g}_h{s}_d{d}{do}{ss}"
    return ov


def plan(model, group):
    """List of override dicts for (model, group)."""
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; choose from {MODELS}")
    if group == "nonanomaly":
        out = [_entry(model, *c, pft_mixing=True, tag="mix")
               for c in _main_combos(model)]
        out += [_entry(model, *c, pft_mixing=False, tag="nomix")
                for c in _curated_combos(model)]
        return out
    if group == "anomaly":
        return [_entry(model, *c, pft_mixing=True, tag="anom")
                for c in _curated_combos(model)]
    if group == "anom_loss":
        return _anom_loss_plan(model, pft_mixing=True)
    if group == "anom_loss_nomix":
        return _anom_loss_plan(model, pft_mixing=False)
    if group == "big":
        return [_entry(model, *c, pft_mixing=True, tag="big")
                for c in _big_combos(model)]
    if group == "loss":
        return loss_plan(model)
    if group == "daily":
        return [_entry(model, *c, pft_mixing=True, tag="daily")
                for c in _daily9_combos(model)]
    if group == "daily_cmp":
        # The first 2 main-block (nonanomaly) configs, re-run in daily mode to
        # compare daily vs the dekadal main sweep at identical config. loss_type
        # forced to huber to match the main block (ONLY the target differs).
        out = []
        for e in plan(model, "nonanomaly")[:2]:
            e = dict(e)
            e["loss_type"] = "huber"
            e["name"] = "cmp_" + e["name"]
            out.append(e)
        return out
    raise SystemExit(f"unknown group {group!r}; choose from {GROUPS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--group", required=True, choices=GROUPS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--list", action="store_true")
    a = ap.parse_args()
    p = plan(a.model, a.group)
    if a.count:
        print(len(p))
    else:
        for i, e in enumerate(p):
            print(f"{i:3d}  {e['name']:28s}  {e}")


if __name__ == "__main__":
    main()
