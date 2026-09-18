#!/bin/bash
# One extra GPU: lstm & bitransformer_v2 (stress_dim 1) at hidden {256,512},
# all trained with the COMBO ablation (tp_sum + ssrd_sum + strd_sum + VPD_max +
# Tmean removed). 4 configs sequentially on ONE shared load. Loss MSE, patience 7.
#   sbatch run_ablate_combo.sh
#SBATCH --job-name=abl-combo-models
#SBATCH --output=Output_ablate_%x_%j.out
#SBATCH --error=Output_ablate_%x_%j.err
#SBATCH --account=EUHPC_D36_053
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=245G
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

module purge
module load python cuda
source //leonardo/home/userexternal/sbarbu00/.venv/bin/activate

DATA=/leonardo_work/EUHPC_D36_053/pixelset1
cd /leonardo_work/EUHPC_D36_053/LaiNN_final

# NB: --type / --hidden_size / --stress_dim are set PER CONFIG by the driver.
python -m phenonn.analysis.feature_ablation.run_ablate_combo_models \
    --features_dir    "$DATA/era5_pixelset" \
    --target_dir      "$DATA/LAI_pixelset" \
    --pft_dir         "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_balanced.nc" \
    --parent_map      "$DATA/selected_pixels01_1.nc" \
    --stats_path      "$DATA/norm_stats_1992_2019.json" \
    --co2_path        "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --train_years 1992-2009 --val_years 2010-2019 --val_fraction_of_grid 100 \
    --n_years_per_epoch 30 --n_sites_per_epoch 250000 --n_val_sites 50000 \
    --subset 0.5 \
    --num_epochs 300 --patience 7 \
    --loss_type mse \
    --pft_mixing --pft_meteo_only \
    --num_layers 2 \
    --batch_size 64 --amp --num_workers 2 \
    --output_dir ./runs_ablate
