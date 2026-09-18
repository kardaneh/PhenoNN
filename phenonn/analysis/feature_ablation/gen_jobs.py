#!/usr/bin/env python3
"""
gen_jobs.py — write one SLURM script per feature-ablation run.

Produces, into `study/feature_ablation/jobs/`:
    run_baseline.sh                 (nothing masked, --experiment abl_baseline)
    run_ablate_<feature>.sh   × N   (one non-PFT feature masked each)

N = number of non-PFT features (currently 11 = 10 meteo + 1 CO2). Every run uses
identical hyperparameters/splits/seed so ΔR² isolates the ablated feature.

The SBATCH header + the common training command below are seeded from
jobs/overnight_sweep.sh (the Base1 run). EDIT the two blocks marked `# EDIT`
once to match your paths, then:

    python -m phenonn.analysis.feature_ablation.gen_jobs
    for f in study/feature_ablation/jobs/run_*.sh; do sbatch "$f"; done
"""

from pathlib import Path

from phenonn.analysis.feature_ablation._common import NON_PFT_FEATURES, BASELINE_EXP, exp_name

# ── EDIT (cluster) ───────────────────────────────────────────────────────────
VENV       = "~/PhenoNN/.venv/bin/activate"
PROJECT    = "/net/nfs/ssd2/sbarbu/PhenoNN"      # dir containing phenon/ and study/
OUTPUT_DIR = "runs"                              # --output_dir (runs/<exp>/…)
LOG_SUBDIR = "runs/ablation_logs"               # python stdout goes here

# ── EDIT (training command) — the Base1 baseline from overnight_sweep.sh, ─────
#    minus --experiment (set per run). One space-joined line.
COMMON_ARGS = " ".join([
    "--features_dir /data/sbarbu/PhenoNN/data/era5_10%",
    "--target_dir /data/sbarbu/PhenoNN/data/pixelset_10%/LAI_pixelset",
    "--pft_dir /data/sbarbu/PhenoNN/data/pixelset_10%/PFT_pixelset",
    "--selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_10%.nc",
    "--co2_path /net/nfs/ssd1/sbarbu/PhenoNN/data/CO2_1700_2023_TRENDYv2024.txt",
    "--stats_path /net/nfs/ssd2/sbarbu/PhenoNN/data/norm_stats_10%.json",
    "--train_years 2001-2004 --val_years 2005",
    "--val_fraction_of_grid 100",
    "--n_sites_per_epoch 50000 --n_years_per_epoch 4 --n_val_sites 50000",
    "--num_epochs 200 --pft_mixing",
    "--type lstm --hidden_size 256 --num_layers 2 --batch_size 1024",
])
# ─────────────────────────────────────────────────────────────────────────────

SBATCH_HEADER = """#!/usr/bin/env bash
#SBATCH --job-name={job}
#SBATCH --output={proj}/study_{tag}_%j.out
#SBATCH --error={proj}/study_{tag}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module purge
module load python
source {venv}
cd {proj}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p {logdir}
"""


def _script(tag: str, job: str, ablate: str, experiment: str) -> str:
    header = SBATCH_HEADER.format(job=job, proj=PROJECT, tag=tag,
                                  venv=VENV, logdir=LOG_SUBDIR)
    cmd = (f"python -m phenonn.analysis.feature_ablation.train_ablate "
           f"--ablate_feature {ablate} {COMMON_ARGS} "
           f"--output_dir {OUTPUT_DIR} --experiment {experiment} "
           f"> {LOG_SUBDIR}/{tag}.log 2>&1\n")
    return header + "\n" + cmd


def main():
    out_dir = Path(__file__).resolve().parent / "jobs"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_baseline.sh").write_text(
        _script("baseline", "abl_baseline", "none", BASELINE_EXP))

    for feat in NON_PFT_FEATURES:
        tag = f"ablate_{feat}"
        (out_dir / f"run_{tag}.sh").write_text(
            _script(tag, f"abl_{feat}", feat, exp_name(feat)))

    n = len(NON_PFT_FEATURES)
    print(f"Wrote {n + 1} scripts to {out_dir}/  (1 baseline + {n} ablations)")
    print("Ablated features:", ", ".join(NON_PFT_FEATURES))
    print(f'\nSubmit all:\n  for f in {out_dir}/run_*.sh; do sbatch "$f"; done')


if __name__ == "__main__":
    main()
