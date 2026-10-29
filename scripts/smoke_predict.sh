#!/usr/bin/env bash
#
# scripts/smoke_predict.sh — inference with the smoke checkpoint.
#
# Purpose
# -------
# Load the checkpoint written by scripts/smoke_train.sh (or its resume
# follow-up), run inference on the validation sites, and write the CSV,
# metrics summary and diagnostic plots into the same experiment folder.
#
# The checkpoint is self-describing: data paths, site pools, normalisation
# statistics and mode flags are all read from the snapshot, so only the
# checkpoint path and the output CSV need to be given. Override any of the
# stored paths with the CLI flags if the layout has moved.
#
# Usage
# -----
#   bash scripts/smoke_predict.sh
#
#   # point at a different checkpoint
#   CKPT=/path/to/best_model.pth.tar bash scripts/smoke_predict.sh
#
#   # predict on more / fewer sites
#   N_SITES=200 bash scripts/smoke_predict.sh

set -euo pipefail

# ── Absolute paths ──────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# ── Checkpoint and experiment folder ────────────────────────────────────────
OUT_DIR="$REPO/runs/smoke"
EXP_NAME="smoke_lstm"
CKPT="${CKPT:-$OUT_DIR/$EXP_NAME/checkpoints/best_model.pth.tar}"

# ── Inference parameters ────────────────────────────────────────────────────
# The site pool is taken from the checkpoint's val_site_ids unless overridden.
N_SITES="${N_SITES:-8}"
N_CURVES="${N_CURVES:-100}"       # cap the per-site curve grid
OUTPUT_CSV="$OUT_DIR/$EXP_NAME/predictions.csv"

# ── Sanity ──────────────────────────────────────────────────────────────────
if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    echo "run scripts/smoke_train.sh first, or set CKPT=..." >&2
    exit 1
fi

echo "Checkpoint    : $CKPT"
echo "Output CSV    : $OUTPUT_CSV"
echo "Sites cap     : $N_SITES"
echo "Curves cap    : $N_CURVES"

# ── The run ─────────────────────────────────────────────────────────────────
# Only three flags are needed:
#   --checkpoint      what to load
#   --predict_sites   which site pool (val / train / all / grid / test)
#   --output_csv      where to write
# Everything else (data directories, parent map, seq_length, mode flags)
# comes from the checkpoint.
cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python"
fi

# Keep the smoke test inside a 1 GB per-process cgroup on shared login nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$PYTHON" -m phenonn.prediction.predict \
    --checkpoint      "$CKPT" \
    --predict_sites   val \
    --n_predict_sites "$N_SITES" \
    --n_curves        "$N_CURVES" \
    --scatter_years \
    --output_csv      "$OUTPUT_CSV"

echo
echo "──────────────────────────────────────────────────────────────"
echo "Prediction run finished."
echo "Experiment dir : $OUT_DIR/$EXP_NAME"
ls -la "$OUT_DIR/$EXP_NAME"/*.csv \
       "$OUT_DIR/$EXP_NAME"/*.txt \
       "$OUT_DIR/$EXP_NAME"/*.png 2>/dev/null || true
echo "Log            : $OUT_DIR/$EXP_NAME/checkpoints/predict.log"
echo "──────────────────────────────────────────────────────────────"
