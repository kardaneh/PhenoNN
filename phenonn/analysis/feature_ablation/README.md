# Feature ablation (leave-one-feature-out)

Measures how much each **non-PFT input feature** contributes to the model, by
retraining once per feature with that feature masked out and comparing
validation R² to a baseline trained with all features.

- **Mechanism**: mean-imputation. Features are z-scored, so a channel's mean is
  0; the ablated channel is zeroed before every forward pass (train + val),
  making it constant → the model cannot use it. Architecture and the 26-channel
  input are unchanged, so ΔR² is directly comparable. Implemented as a forward
  pre-hook (`train_ablate.py`) — `phenon/` is not modified, checkpoints stay
  compatible.
- **What gets ablated**: `phenonn.utils.config.ALL_FEATURES[:PFT_START]`, currently the
  10 meteo features + CO2 = **11 runs**. PFT fractions are kept in every run.
- **Baseline**: trained here as its own run (`abl_baseline`) with identical
  seed / splits / hyperparameters, so only the ablated feature differs.

## Workflow

1. **Edit** the two `# EDIT` blocks in `gen_jobs.py` (paths + the common
   training command; seeded from `jobs/overnight_sweep.sh`).
2. **Generate** the SLURM scripts:
   ```bash
   python -m phenonn.analysis.feature_ablation.gen_jobs
   ```
   → `study/feature_ablation/jobs/run_baseline.sh` + `run_ablate_<feature>.sh`.
3. **Submit** (baseline + 11 ablations run in parallel):
   ```bash
   for f in study/feature_ablation/jobs/run_*.sh; do sbatch "$f"; done
   ```
4. **Aggregate** once they finish:
   ```bash
   python -m phenonn.analysis.feature_ablation.aggregate --runs_dir runs
   ```
   → printed table + `feature_ablation.csv` + `feature_ablation.png`
   (features sorted by R² lost when removed).

## Reading the result

`delta_r2 = R2(ablated) − R2(baseline)`. A **negative** ΔR² means removing the
feature hurt → the feature was useful; the more negative, the more important.
The plot shows `-ΔR²` (R² lost when removed), largest first.

## Run a single ablation by hand

```bash
python -m phenonn.analysis.feature_ablation.train_ablate --ablate_feature Tmean \
    <the usual phenonn.training.train_full_ram args> --experiment abl_Tmean
# baseline: --ablate_feature none --experiment abl_baseline
```
