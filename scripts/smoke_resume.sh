#!/usr/bin/env bash
#
# scripts/smoke_resume.sh — resume the smoke run for two more epochs.
#
# Purpose
# -------
# Load a checkpoint written by scripts/smoke_train.sh, continue training
# for two more epochs (epochs 3 and 4), and overwrite the `best` and
# `final` snapshots in the same experiment folder.
#
# Everything except `--num_epochs` and `--resume` is inherited from the
# checkpoint's stored arguments. Passing a flag on the command line
# overrides the stored value.
#
# Usage
# -----
#   bash scripts/smoke_resume.sh
#
#   # resume a different checkpoint
#   CKPT=/path/to/checkpoint.pth.tar bash scripts/smoke_resume.sh
#
#   # run more (or fewer) extra epochs
#   NUM_EPOCHS=10 bash scripts/smoke_resume.sh

set -euo pipefail

# ── Absolute paths ──────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# ── Where the smoke run wrote its snapshots ─────────────────────────────────
OUT_DIR="$REPO/runs/smoke"
EXP_NAME="smoke_lstm"
CKPT="${CKPT:-$OUT_DIR/$EXP_NAME/checkpoints/best_model.pth.tar}"

# ── Number of epochs to run to, TOTAL ───────────────────────────────────────
# The smoke run stopped at epoch 2. We want two more, so 4.
NUM_EPOCHS="${NUM_EPOCHS:-8}"

# ── Sanity ──────────────────────────────────────────────────────────────────
if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    echo "run scripts/smoke_train.sh first, or set CKPT=..." >&2
    exit 1
fi

echo "Resuming from  : $CKPT"
echo "Experiment     : $OUT_DIR/$EXP_NAME"
echo "Total epochs   : $NUM_EPOCHS"

# ── The run ─────────────────────────────────────────────────────────────────
cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python"
fi

# Keep the smoke test inside a 1 GB per-process cgroup on shared login nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$PYTHON" -m phenonn.training.train_full_ram \
    --resume            "$CKPT" \
    --num_epochs        "$NUM_EPOCHS" \
    --output_dir        "$OUT_DIR" \
    --experiment        "$EXP_NAME" \
    --prefix            "" \
    --save_checkpoint_name "model"

echo
echo "──────────────────────────────────────────────────────────────"
echo "Resume run finished."
echo "Checkpoint dir : $OUT_DIR/$EXP_NAME/checkpoints"
ls -la "$OUT_DIR/$EXP_NAME/checkpoints" || true
echo "──────────────────────────────────────────────────────────────"
