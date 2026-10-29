#!/usr/bin/env bash
#
# scripts/run_baseline.sh — small GPU-first baseline training run.
#
# Purpose
# -------
# Provide the W1 "baseline" checkpoint: same code path as
# scripts/smoke_train.sh (dataset, model, trainer, checkpoint writer)
# but ~10x larger, so metrics are meaningful while still finishing
# quickly on GPU (tens of minutes on CPU fallback).
#
# Relationship
# ------------
#   smoke_train.sh  : tiny repro (minutes, 2 epochs, 8 sites, CPU).
#   run_baseline.sh : small baseline (GPU-first, 10 epochs, 32 sites).
#   analysis/*/jobs/ : full SLURM / GPU runs.
#
# Usage
# -----
#   DATA=/data/ecarlin/pixelset1 bash scripts/run_baseline.sh
#   DEVICE=cpu DATA=... bash scripts/run_baseline.sh   # force CPU
#   DEVICE=cuda DATA=... bash scripts/run_baseline.sh  # force GPU
#
# All paths point to the dataset folder outside the repo
# (default ../pixelset1 for local layouts). Override with $DATA.
# Device policy: GPU first; automatic fallback to CPU when CUDA is
# unavailable or the GPU run fails (e.g. torch/CUDA kernel mismatch).

set -euo pipefail

# ── Absolute paths ──────────────────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

DATA="${DATA:-$REPO/pixelset1}"

FEATURES_DIR="$DATA/era5_pixelset"
TARGET_DIR="$DATA/LAI_pixelset"
PFT_DIR="$DATA/PFT_pixelset"
SELECTED_PIXELS="$DATA/selected_pixels005_1.nc"
PARENT_MAP="$DATA/selected_pixels01_1.nc"
STATS_PATH="$DATA/norm_stats_1992_2019.json"
CO2_PATH="$DATA/CO2_1700_2023_TRENDYv2024.txt"

# ── Output location ─────────────────────────────────────────────────────────
# runs/ is a symlink to SSD-backed storage on the cluster; override with
# RUNS_DIR only if bypassing the link.
RUNS_DIR="${RUNS_DIR:-$REPO/runs}"
OUT_DIR="$RUNS_DIR/baseline"
EXP_NAME="baseline_lstm"

# ── Sanity: confirm each input exists ───────────────────────────────────────
for p in "$FEATURES_DIR" "$TARGET_DIR" "$PFT_DIR" \
         "$SELECTED_PIXELS" "$PARENT_MAP" "$STATS_PATH" "$CO2_PATH"; do
    if [[ ! -e "$p" ]]; then
        echo "missing: $p" >&2
        exit 1
    fi
done

# ── Device policy: GPU first, CPU fallback ───────────────────────────────────
# DEVICE=auto (default) | cpu | cuda. The trainer itself picks
# cuda when torch.cuda.is_available(); here we only tune the flags
# that differ (batch size, workers, bf16 autocast) and retry on CPU
# if the GPU attempt dies (known torch/CUDA kernel mismatch).
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS=()
USE_GPU=0

if [[ "$DEVICE" == "cuda" || "$DEVICE" == "gpu" ]]; then
    USE_GPU=1
elif [[ "$DEVICE" == "cpu" ]]; then
    USE_GPU=0
else
    if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        USE_GPU=1
    else
        USE_GPU=0
    fi
fi

if [[ "$USE_GPU" == "1" ]]; then
    # GPU-first: bigger batches, parallel loading, bf16 autocast.
    # (Fused Adam stays on: trainer enables it on cuda by default.)
    BATCH_SIZE=64
    NUM_WORKERS=4
    EXTRA_ARGS=(--amp)
    echo "Device: GPU requested (fallback to CPU on failure)."
else
    BATCH_SIZE=16
    NUM_WORKERS=0
    echo "Device: CPU."
fi

# ── The run ─────────────────────────────────────────────────────────────────
# Deltas vs smoke_train.sh (same trainer, same lstm + pft_mixing setup):
# --subset 0.01          : ~10x the pool
# --n_sites_per_epoch 32 : vs 8
# --n_years_per_epoch 4  : vs 2
# --train_years 2000-2003 / --val_years 2004 : wider window than 2000-2001/2002
# --hidden_size 64 --num_epochs 10 : vs 32 / 2
# Batch/workers/amp follow the device policy above (GPU: 64/4/--amp).
run_train() {
    PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
    if [[ ! -x "$PYTHON" ]]; then
        PYTHON="python"
    fi
    "$PYTHON" -m phenonn.training.train_full_ram \
    --features_dir      "$FEATURES_DIR" \
    --target_dir        "$TARGET_DIR" \
    --pft_dir           "$PFT_DIR" \
    --selected_pixels   "$SELECTED_PIXELS" \
    --parent_map        "$PARENT_MAP" \
    --stats_path        "$STATS_PATH" \
    --co2_path          "$CO2_PATH" \
    --train_years       "2000-2003" \
    --val_years         "2004" \
    --val_fraction_of_grid 100 \
    --subset            0.01 \
    --n_sites_per_epoch 32 \
    --n_years_per_epoch 4 \
    --n_val_sites       32 \
    --type              lstm \
    --hidden_size       64 \
    --num_layers        1 \
    --pft_mixing \
    --pft_meteo_only \
    --loss_type         huber \
    --num_epochs        10 \
    --batch_size        "$BATCH_SIZE" \
    --num_workers       "$NUM_WORKERS" \
    --max_grad_norm     1.0 \
    --output_dir        "$OUT_DIR" \
    --experiment        "$EXP_NAME" \
    "${EXTRA_ARGS[@]:-}"
}

START="$SECONDS"
cd "$REPO"

if [[ "$USE_GPU" == "1" ]]; then
    if run_train; then
        :
    else
        echo "GPU run failed; retrying on CPU with safe flags." >&2
        BATCH_SIZE=16
        NUM_WORKERS=0
        EXTRA_ARGS=()
        CUDA_VISIBLE_DEVICES="" run_train
    fi
else
    run_train
fi

echo
echo "──────────────────────────────────────────────────────────────"
echo "Baseline run finished in $((SECONDS - START))s."
echo "Checkpoint dir : $OUT_DIR/$EXP_NAME/checkpoints"
ls -la "$OUT_DIR/$EXP_NAME/checkpoints" || true
echo "──────────────────────────────────────────────────────────────"
