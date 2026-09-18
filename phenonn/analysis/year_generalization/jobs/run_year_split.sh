#!/bin/bash
# One GPU = one YEAR split of the temporal-generalization study. Fixed model
# (aelstm h256/d2, huber, pft_mixing meteo_only); only train/val YEARS change
# (set by the driver from $SPLIT). Do NOT sbatch directly: use submit_year_split.sh.
#SBATCH --output=Output_yeargen_%x_%j.out     # %x = job-name (yg-<split>)
#SBATCH --error=Output_yeargen_%x_%j.err
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
: "${SPLIT:?SPLIT non défini (chrono|rand1..4|coldhot) — lance via submit_year_split.sh}"

module purge
module load python cuda
source //leonardo/home/userexternal/sbarbu00/.venv/bin/activate

DATA=/leonardo_work/EUHPC_D36_053/pixelset1
cd /leonardo_work/EUHPC_D36_053/LaiNN_final

# Fixed model config — identical for every split so ΔR² isolates the split effect.
python -m phenonn.analysis.year_generalization.run_year_split \
    --split "$SPLIT" --temp_rank_json ./year_temp_rank.json \
    --features_dir    "$DATA/era5_pixelset" \
    --target_dir      "$DATA/LAI_pixelset" \
    --pft_dir         "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_balanced.nc" \
    --parent_map      "$DATA/selected_pixels01_1.nc" \
    --stats_path      "$DATA/norm_stats_1992_2019.json" \
    --co2_path        "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --val_fraction_of_grid 100 \
    --n_years_per_epoch 30 --n_sites_per_epoch 250000 --n_val_sites 50000 \
    --subset 0.5 \
    --type aelstm --hidden_size 256 --num_layers 2 \
    --learning_rate 1e-3 --weight_decay 1e-5 \
    --loss_type huber \
    --pft_mixing --pft_meteo_only \
    --num_epochs 300 --patience 15 \
    --batch_size 64 --amp --num_workers 2 \
    --output_dir ./runs_year_gen
