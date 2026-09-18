#!/bin/bash -l
# hal : évaluation complète des 2 runs daily (aelstm + bitransformer_v2).
# Pour chaque modèle : 1) prédiction sur la val  2) les 4 scénarios de
# perturbation  3) l'agrégation comparative.
# L'ablation (tp_sum, ssrd_sum, strd_sum, Tmean) et la sortie daily (365 j → 36
# dékades) sont rejouées automatiquement, lues dans le checkpoint.
#   sbatch run_eval_hal.sh
#SBATCH --job-name=eval_hal
#SBATCH --output=eval_hal.o%j
#SBATCH --error=eval_hal.e%j
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --account=ipsl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

source /home/sbarbu/PhenoNN/.venv/bin/activate

DATA=/data/sbarbu/dataset_building_final/pixelset1
cd /home/sbarbu/LaiNN_final

NSITES=${NSITES:-5000}
EXPS=(
    dailyx_hal_aelstm_h256_no_tp_ssrd_strd_Tmean
    dailyx_hal_bitr_h256_s1_no_tp_ssrd_strd_Tmean
)

DIRS=(
    --features_dir "$DATA/era5_pixelset"
    --target_dir   "$DATA/LAI_pixelset"
    --pft_dir      "$DATA/PFT_pixelset"
    --parent_map   "$DATA/selected_pixels01_1.nc"
)

for EXP in "${EXPS[@]}"; do
    CKPT=runs_daily_hal/$EXP/checkpoints/best_model.pth
    if [ ! -f "$CKPT" ]; then
        echo "!! $CKPT absent — modèle pas encore entraîné, on saute"; continue
    fi
    echo "================ $EXP ================"

    # ── 1) Prédiction sur la validation ──
    # --n_curves 100 : SANS ça (défaut 0) predict.py trace TOUS les sites dans une
    # seule figure → 5 000 sous-graphes → matplotlib dépasse sa limite de 2^16 px.
    python -m phenonn.prediction.predict \
        --checkpoint "$CKPT" "${DIRS[@]}" \
        --predict_sites val --n_predict_sites "$NSITES" \
        --n_curves 100 --scatter_years \
        --output_csv "runs_daily_hal/$EXP/pred_val.csv"

    # ── 2) Les 4 scénarios de perturbation ──
    OUT=runs_perturb/$EXP
    PCOMMON=(--checkpoint "$CKPT" "${DIRS[@]}"
             --predict_sites val --n_predict_sites "$NSITES" --output_dir "$OUT")
    python -m phenonn.analysis.perturbation.drought      "${PCOMMON[@]}"
    python -m phenonn.analysis.perturbation.heatwave     "${PCOMMON[@]}"
    python -m phenonn.analysis.perturbation.warming_co2  "${PCOMMON[@]}"
    python -m phenonn.analysis.perturbation.winter_frost "${PCOMMON[@]}"

    # ── 3) Comparaison des scénarios ──
    python -m phenonn.analysis.perturbation.aggregate_perturbation --output_dir "$OUT"
done

echo "=== Terminé ==="
