#!/usr/bin/env python3
"""
tuning_grid.py
==============

Emit the swept hyperparameter flags for ONE grid point of a model's tuning
sweep (learning_rate × size × depth × dropout). Restricted grid, model-aware:

  • lstm             RNN_LSTM(hidden_size, num_layers) — IGNORES dropout, so its
                     grid omits the dropout axis  → 3×2×2 = 12 configs.
  • aelstm           dropout via --dropout2         → 3×2×2×2 = 24 configs.
  • bitransformer_v2 dropout on BOTH --dropout1 & --dropout2 (the two stages);
    / attnlstm       --hidden_size also drives d_model (build_model's fallback),
                     so one "size" knob scales both stages → 24 configs each.

Standalone (argparse only, torch-free) so it runs in any env, called from the
sbatch as a plain file.

    python tuning_grid.py --model bitransformer_v2 --count      # -> 24
    python tuning_grid.py --model bitransformer_v2 --index 7     # -> flag string
"""

import argparse
import itertools

LR = [3e-4, 1e-3, 3e-3]
SIZE = [128, 256]
DEPTH = [2, 3]
DROPOUT = [0.0, 0.2]

MODELS = ["lstm", "aelstm", "attnlstm", "bitransformer_v2"]


def _grid(model):
    """List of (lr, size, depth, dropout|None) tuples for the model."""
    if model == "lstm":                       # RNN_LSTM ignores dropout
        return [(lr, s, d, None)
                for lr, s, d in itertools.product(LR, SIZE, DEPTH)]
    return list(itertools.product(LR, SIZE, DEPTH, DROPOUT))


def config_overrides(model, lr, size, depth, drop, stress=None):
    """Namespace-attribute overrides for one grid point. Single source shared by
    the sbatch flag emitter (_flags) and the in-process sweep driver
    (run_sweep_inproc.py), so both apply the exact same model-aware dropout wiring.

    `stress` (--stress_dim) is only read by bitransformer_v2 / attnlstm; it is
    emitted only for those models and only when passed (sweep_plan uses it; the
    legacy _flags grid leaves it None).
    """
    ov = {"learning_rate": lr, "hidden_size": size, "num_layers": depth}
    if drop is not None:
        if model in ("bitransformer_v2", "attnlstm"):
            ov["dropout1"] = drop                 # stage-1 transformer
            ov["dropout2"] = drop                 # stage-2 / LSTM
        else:                                     # aelstm
            ov["dropout2"] = drop
    if stress is not None and model in ("bitransformer_v2", "attnlstm"):
        ov["stress_dim"] = stress
    return ov


def _flags(model, lr, size, depth, drop):
    ov = config_overrides(model, lr, size, depth, drop)
    order = ["learning_rate", "hidden_size", "num_layers", "dropout1", "dropout2"]
    return " ".join(f"--{k} {ov[k]:g}" for k in order if k in ov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="grid point → prints its flags")
    g.add_argument("--count", action="store_true", help="prints the grid size")
    a = ap.parse_args()

    grid = _grid(a.model)
    if a.count:
        print(len(grid))
        return
    if not (0 <= a.index < len(grid)):
        raise SystemExit(f"index {a.index} out of range 0..{len(grid) - 1} "
                         f"for {a.model}")
    print(_flags(a.model, *grid[a.index]))


if __name__ == "__main__":
    main()
