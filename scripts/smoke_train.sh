#!/usr/bin/env bash
#
# scripts/smoke_train.sh — minimal end-to-end training run.
#
# Purpose
# -------
# Verify that the kept surface (dataset, model, trainer, wrapper,
# checkpoint writer) still works after the cleanup. Runs on CPU, two
# epochs, a few sites, a couple of years. Finishes in a few minutes.
#
# Not a replacement for the real training jobs: those are the SLURM
# scripts under analysis/*/jobs/.
#
# Usage
# -----
#   bash scripts/smoke_train.sh
#
#   # override the data directory
#   DATA="$PWD/pixelset1" bash scripts/smoke_train.sh
#
# All default paths point to ../pixelset1 next to the repo. Edit below
# if your layout changes.

set -euo pipefail

# ── Absolute paths ──────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

DATA="${DATA:-$REPO/../pixelset1}"

FEATURES_DIR="$DATA/era5_pixelset"
TARGET_DIR="$DATA/LAI_pixelset"
PFT_DIR="$DATA/PFT_pixelset"
SELECTED_PIXELS="$DATA/selected_pixels005_1.nc"
PARENT_MAP="$DATA/selected_pixels01_1.nc"
STATS_PATH="$DATA/norm_stats_1992_2019.json"
CO2_PATH="$DATA/CO2_1700_2023_TRENDYv2024.txt"

# ── Output location ─────────────────────────────────────────────────────────
OUT_DIR="$REPO/runs/smoke"
EXP_NAME="smoke_lstm"

# ── Sanity: confirm each input exists ───────────────────────────────────────
for p in "$FEATURES_DIR" "$TARGET_DIR" "$PFT_DIR" \
         "$SELECTED_PIXELS" "$PARENT_MAP" "$STATS_PATH" "$CO2_PATH"; do
    if [[ ! -e "$p" ]]; then
        echo "missing: $p" >&2
        exit 1
    fi
done

# ── The run ─────────────────────────────────────────────────────────────────
# --subset 0.001       : ~1/1000th of the pool  →  ~5–10 sites
# --n_sites_per_epoch  : cap per-epoch sample to 8
# --n_years_per_epoch  : 2 years
# --num_epochs 2       : two epochs is enough to prove the loop reaches
# --batch_size 4       : small batch keeps it CPU-friendly
# --num_workers 0      : avoid fork overhead on a tiny run
# --amp                : OFF (no GPU)
# --prefix ""          : short checkpoint names (best_model.pth.tar, …)
# --save_checkpoint_name model
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
    --features_dir      "$FEATURES_DIR" \
    --target_dir        "$TARGET_DIR" \
    --pft_dir           "$PFT_DIR" \
    --selected_pixels   "$SELECTED_PIXELS" \
    --parent_map        "$PARENT_MAP" \
    --stats_path        "$STATS_PATH" \
    --co2_path          "$CO2_PATH" \
    --train_years       "2000-2001" \
    --val_years         "2002" \
    --val_fraction_of_grid 100 \
    --subset            0.001 \
    --n_sites_per_epoch 8 \
    --n_years_per_epoch 2 \
    --n_val_sites       8 \
    --type              lstm \
    --hidden_size       32 \
    --num_layers        1 \
    --pft_mixing \
    --pft_meteo_only \
    --loss_type         huber \
    --num_epochs        2 \
    --batch_size        4 \
    --num_workers       0 \
    --max_grad_norm     1.0 \
    --prefix            "" \
    --save_checkpoint_name "model" \
    --output_dir        "$OUT_DIR" \
    --experiment        "$EXP_NAME"

echo
echo "──────────────────────────────────────────────────────────────"
echo "Smoke run finished."
echo "Checkpoint dir : $OUT_DIR/$EXP_NAME/checkpoints"
ls -la "$OUT_DIR/$EXP_NAME/checkpoints" || true
echo "──────────────────────────────────────────────────────────────"
