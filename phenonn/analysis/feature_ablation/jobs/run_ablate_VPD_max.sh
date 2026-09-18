#!/usr/bin/env bash
#SBATCH --job-name=abl_VPD_max
#SBATCH --output=/net/nfs/ssd2/sbarbu/PhenoNN/study_ablate_VPD_max_%j.out
#SBATCH --error=/net/nfs/ssd2/sbarbu/PhenoNN/study_ablate_VPD_max_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module purge
module load python
source ~/PhenoNN/.venv/bin/activate
cd /net/nfs/ssd2/sbarbu/PhenoNN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/ablation_logs

python -m phenonn.analysis.feature_ablation.train_ablate --ablate_feature VPD_max --features_dir /data/sbarbu/PhenoNN/data/era5_10% --target_dir /data/sbarbu/PhenoNN/data/pixelset_10%/LAI_pixelset --pft_dir /data/sbarbu/PhenoNN/data/pixelset_10%/PFT_pixelset --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_10%.nc --co2_path /net/nfs/ssd1/sbarbu/PhenoNN/data/CO2_1700_2023_TRENDYv2024.txt --stats_path /net/nfs/ssd2/sbarbu/PhenoNN/data/norm_stats_10%.json --train_years 2001-2004 --val_years 2005 --val_fraction_of_grid 100 --n_sites_per_epoch 50000 --n_years_per_epoch 4 --n_val_sites 50000 --num_epochs 200 --pft_mixing --type lstm --hidden_size 256 --num_layers 2 --batch_size 1024 --output_dir runs --experiment abl_VPD_max > runs/ablation_logs/ablate_VPD_max.log 2>&1
