#!/bin/bash -l
# Étude de perturbation météo sur hal : les 4 scénarios à la suite sur UN modèle,
# puis agrégation. Si le checkpoint a été entraîné avec des features ablatées,
# l'ablation est automatiquement rejouée à l'inférence (lue dans le checkpoint).
#
#   sbatch --export=ALL,CKPT=runs_daily_hal/<exp>/checkpoints/best_model.pth \
#          run_perturbation_hal.sh
#SBATCH --job-name=perturb
#SBATCH --output=perturb.o%j
#SBATCH --error=perturb.e%j
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --account=ipsl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
: "${CKPT:?CKPT non défini — passe --export=ALL,CKPT=<chemin best_model.pth>}"

source /home/sbarbu/PhenoNN/.venv/bin/activate

DATA=/data/sbarbu/dataset_building_final/pixelset1
cd /home/sbarbu/LaiNN_final

# Dossier de sortie dérivé du nom de l'expérience
EXP=$(basename "$(dirname "$(dirname "$CKPT")")")
OUT=runs_perturb/$EXP
NSITES=${NSITES:-5000}

COMMON=(
    --checkpoint   "$CKPT"
    --features_dir "$DATA/era5_pixelset"
    --target_dir   "$DATA/LAI_pixelset"
    --pft_dir      "$DATA/PFT_pixelset"
    --parent_map   "$DATA/selected_pixels01_1.nc"
    --predict_sites val --n_predict_sites "$NSITES"
    --output_dir   "$OUT"
)

echo "=== Perturbation study : $EXP  → $OUT  (n_sites=$NSITES) ==="

python -m phenonn.analysis.perturbation.drought      "${COMMON[@]}"
python -m phenonn.analysis.perturbation.heatwave     "${COMMON[@]}"
python -m phenonn.analysis.perturbation.warming_co2  "${COMMON[@]}"
python -m phenonn.analysis.perturbation.winter_frost "${COMMON[@]}"

python -m phenonn.analysis.perturbation.aggregate_perturbation --output_dir "$OUT"

echo "=== Terminé : $OUT ==="
