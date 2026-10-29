#!/usr/bin/env bash
#
# scripts/smoke_feature_ablation.sh — short feature-ablation pipeline test.
#
# Runs the SMOKE profile (2 train years, 1 val year, 5 sites, 1 epoch) for the
# baseline and the selected features, then aggregates them. Data is loaded once
# by the in-process sweep. No SLURM needed.
#
# Usage:
#   bash scripts/smoke_feature_ablation.sh
#   bash scripts/smoke_feature_ablation.sh Tmean tp_sum
#   DATA="$PWD/pixelset1" bash scripts/smoke_feature_ablation.sh
#
# Output: runs/feature_ablation_smoke/{abl_*,feature_ablation.csv,png}

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
export PHENONN_ROOT="$PROJECT"
# shellcheck source=/dev/null
source "$PROJECT/analysis/_cluster.sh"
phenonn_cluster_init
phenonn_require_data \
    "$DATA/era5_pixelset" "$DATA/LAI_pixelset" "$DATA/PFT_pixelset" \
    "$DATA/selected_pixels005_1.nc" "$DATA/selected_pixels01_1.nc" \
    "$DATA/norm_stats_1992_2019.json" "$DATA/CO2_1700_2023_TRENDYv2024.txt"

FEATURES="${*:-Tmean,tp_sum}"
FEATURES="${FEATURES// /,}"

# Keep the smoke test inside a 1 GB per-process cgroup on shared login nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "$PHENONN_PYTHON" -m analysis.feature_ablation.smoke \
    --project "$PROJECT" --data "$DATA" --features "$FEATURES"
