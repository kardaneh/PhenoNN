# Feature ablation

Feature ablation measures how much each non-PFT input feature contributes to
the model. It trains one model per feature with that feature removed, then
compares validation R2 against a baseline trained with every feature.

## What it does

- Masking uses mean imputation. Features are z-scored, so the mean of each
  channel is zero. The ablated channel is set to zero before every forward
  pass in training and validation. The model cannot use that channel.
- The architecture and the remaining inputs are unchanged, so validation R2
  is directly comparable with the baseline.
- The current configuration has 13 non-PFT features: 10 meteorological
  features, daylength, SMI, and CO2. The per-feature study therefore runs 13
  ablations plus one baseline.
- PFT fractions remain active in every run.
- Each checkpoint stores `ablate_features`. Prediction and aggregation use
  this mask to apply or validate the same ablation automatically.
- The baseline is a separate run, `abl_baseline`, with the same seed, splits,
  and hyperparameters.

## Before you run

Run every command from the repository root. The shell wrappers set up the
environment, but the `python -m` commands below expect the virtual environment
to be active.

```bash
cd /path/to/PhenoNN
source .venv/bin/activate

export DATA="${DATA:-$PWD/pixelset1}"
export RUNS="${RUNS:-$PWD/runs}"
export PHENONN_ROOT="$PWD"
export PHENONN_SITE=local
export FA_ROOT="$RUNS/feature_ablation"
export FA_SMOKE="$RUNS/feature_ablation_smoke"

export DATA
export RUNS
export FA_ROOT
export FA_SMOKE
```

Use `PHENONN_SITE=leonardo` for Leonardo or `PHENONN_SITE=local` for a local
run. The presets live in `analysis/_cluster.sh`. The paper profile defaults to
`SUBSET=0.25` and `PHENONN_MEM=120G`; override both for larger nodes.

## Runnable scripts

### `scripts/smoke_feature_ablation.sh`

Runs the short end-to-end test. It checks the seven inputs, trains a baseline
and the selected features on 2008-2009, validates on 2010, and aggregates the
results. It needs no SLURM and loads the data once.

```bash
bash scripts/smoke_feature_ablation.sh
```

Pass feature names as arguments:

```bash
bash scripts/smoke_feature_ablation.sh Tmean tp_sum
```

Set the data path explicitly when `$DATA` is not exported:

```bash
DATA="$PWD/pixelset1" bash scripts/smoke_feature_ablation.sh
```

Outputs go to `$FA_SMOKE`: the ablation checkpoints, `feature_ablation.csv`,
and a plot.

### `analysis.feature_ablation.smoke`

This is the Python driver behind the smoke shell script. It runs the same
one-epoch, five-site sweep in one process.

```bash
.venv/bin/python -m analysis.feature_ablation.smoke --features Tmean,tp_sum
```

Useful options:

```bash
.venv/bin/python -m analysis.feature_ablation.smoke \
    --features Tmean,tp_sum \
    --data "$DATA" \
    --output_dir "$FA_SMOKE"
```

The default features are `Tmean,tp_sum`. The default data path is `$DATA`, then
`<repo>/pixelset1`, then `<repo>/../pixelset1`.

### `analysis.feature_ablation.gen_jobs`

Writes site-agnostic job scripts. The scripts contain no partition, account,
QoS, or memory directives; `analysis/_cluster.sh` adds those at submission
time.

```bash
.venv/bin/python -m analysis.feature_ablation.gen_jobs \
    --profile paper --study features --output_dir "$FA_ROOT"
bash "$FA_ROOT/jobs/submit_all.sh"
```

Choose another study with `--study`:

```bash
python -m analysis.feature_ablation.gen_jobs --profile paper --study groups
python -m analysis.feature_ablation.gen_jobs --profile paper --study combo
python -m analysis.feature_ablation.gen_jobs --profile paper --study all
```

Generate a small test set without submitting it:

```bash
python -m analysis.feature_ablation.gen_jobs \
    --profile smoke --study features \
    --output_dir /tmp/fa_check
```

Use `--features` to restrict the per-feature sweep, `--subset` to override the
profile site subset, `--common_args` to replace training hyperparameters, and
`--skip-baseline` for combo runs that reuse existing baselines. Dataset flags
are always added automatically. `--subset` cannot be combined with
`--common_args`; include the subset in the replacement arguments instead.

### `analysis/feature_ablation/jobs/submit_ablate.sh`

Generates the per-feature study and submits it. This is the normal entry point
for a production feature or group sweep.

```bash
OUTPUT_DIR="$FA_ROOT" PHENONN_SITE=local \
    bash analysis/feature_ablation/jobs/submit_ablate.sh --study features
```

The default profile is `paper`: AELSTM, 18 training years, 10 validation years,
and 300 epochs. It queues the baseline and 13 ablations. The wrapper
uses `SUBSET=0.25` and `PHENONN_MEM=120G` by default.

Run the grouped sweep:

```bash
OUTPUT_DIR="$FA_ROOT" PHENONN_SITE=local \
    bash analysis/feature_ablation/jobs/submit_ablate.sh --study groups
```

Run a local test:

```bash
PROFILE=smoke PHENONN_LOCAL=1 PHENONN_SITE=local \
  bash analysis/feature_ablation/jobs/submit_ablate.sh
```

Write the jobs without submitting them:

```bash
OUTPUT_DIR="$FA_ROOT" PHENONN_DRY_RUN=1 PHENONN_SITE=local \
  bash analysis/feature_ablation/jobs/submit_ablate.sh --study features
```

Extra arguments are forwarded to `gen_jobs.py`.

### `analysis/feature_ablation/jobs/submit_combo.sh`

Generates and submits the fixed model and size sweep. Each configuration gets a
baseline and one ablation.

```bash
OUTPUT_DIR="$FA_ROOT" PHENONN_SITE=local \
    bash analysis/feature_ablation/jobs/submit_combo.sh
```

Reuse existing baselines:

```bash
OUTPUT_DIR="$FA_ROOT" PHENONN_SITE=local \
    bash analysis/feature_ablation/jobs/submit_combo.sh --skip-baseline
```

Submit combo only after the feature or group jobs finish. On a single-GPU
quota, a later combo job can otherwise start first. To chain it explicitly:

```bash
export PHENONN_DEPENDENCY=afterok:<jobid>
```

### `analysis.feature_ablation.aggregate`

Reads checkpoints and writes a CSV table and a bar plot. It validates the
stored ablation mask before including a run.

Per-feature study:

```bash
.venv/bin/python -m analysis.feature_ablation.aggregate \
    --kind features --runs_dir "$FA_ROOT"
```

Grouped study:

```bash
.venv/bin/python -m analysis.feature_ablation.aggregate \
    --kind groups --model aelstm --runs_dir "$FA_ROOT"
```

Combo study:

```bash
.venv/bin/python -m analysis.feature_ablation.aggregate \
    --kind combo --runs_dir "$FA_ROOT"
```

Use `--output` to choose the plot path. The CSV is written next to it. Use
`--strict-mask` to skip checkpoints without a stored mask, or
`--allow-mask-mismatch` to include mismatched runs.

### `analysis.feature_ablation.train_ablate`

Runs one ablation by hand. This is the same trainer used by generated jobs, with
the ablation hooks added.

```bash
python -m analysis.feature_ablation.train_ablate \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --train_years 2008-2009 --val_years 2010 \
    --type aelstm --pft_mixing --num_epochs 1 \
    --n_sites_per_epoch 5 --n_val_sites 5 \
    --ablate_feature Tmean --experiment abl_Tmean \
    --output_dir "$FA_ROOT"
```

This command is a one-epoch test. Increase the years, sites, and epochs for a
real run. Use `--ablate_feature none` for the baseline and a different
`--experiment` name for each run.

### `analysis.feature_ablation.run_ablate_inproc`

This is the in-process driver used by group and feature jobs. It loads the
dataset once, then trains the selected runs one after another. Index 0 is the
baseline in features mode.

```bash
python -m analysis.feature_ablation.run_ablate_inproc \
    --mode features --model lstm \
    --features Tmean,tp_sum --indices 0-2 \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --train_years 2008-2009 --val_years 2010 \
    --type lstm --pft_mixing --num_epochs 1 \
    --n_sites_per_epoch 5 --n_val_sites 5 \
    --prefix "" --output_dir "$FA_ROOT"
```

For a grouped run, replace the mode and features:

```bash
python -m analysis.feature_ablation.run_ablate_inproc \
    --mode groups --model aelstm --indices 0-2 \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --train_years 2008-2009 --val_years 2010 \
    --type aelstm --pft_mixing --num_epochs 1 \
    --n_sites_per_epoch 5 --n_val_sites 5 \
    --prefix "" --output_dir "$FA_ROOT"
```

### `analysis.feature_ablation.run_ablate_combo_models`

This driver trains the fixed combo configuration list. Each configuration gets
a baseline and one ablation unless `--skip-baseline` is given.

```bash
python -m analysis.feature_ablation.run_ablate_combo_models \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --train_years 2008-2009 --val_years 2010 \
    --type lstm --pft_mixing --num_epochs 1 \
    --n_sites_per_epoch 5 --n_val_sites 5 \
    --prefix "" --output_dir "$FA_ROOT"
```

Use the job wrappers for real runs. The commands above are short smoke
invocations with one epoch and five sites.

## Inputs

`$DATA` must contain these seven inputs:

```text
era5_pixelset/
LAI_pixelset/
PFT_pixelset/
selected_pixels005_1.nc
selected_pixels01_1.nc
norm_stats_1992_2019.json
CO2_1700_2023_TRENDYv2024.txt
```

`gen_jobs.py` and the smoke driver add all dataset flags automatically.
`--common_args` only replaces training hyperparameters; dataset flags are
always added.

Before `sbatch`, generated `submit_all.sh` checks that all seven inputs exist.
If a path is missing, it prints the path instead of submitting a job. Set
`PHENONN_DRY_RUN=1` to write the scripts without submitting them.

A healthy run logs the normalisation statistics and parent map:

```text
[INFO] Norm stats      : /data/.../norm_stats_1992_2019.json
[INFO] Parent map      : /data/.../selected_pixels01_1.nc
```

Check them before waiting for a long job:

```bash
grep -E "Norm stats|Parent map" "$FA_ROOT"/logs/baseline_*.out
```

## Read the results

The aggregation uses:

```text
delta_r2 = R2(ablated) - R2(baseline)
```

A negative `delta_r2` means removing the feature hurt the model, so the feature
was useful. The more negative the value, the more important the feature. The
plot shows `-delta_r2`, with the largest loss first.

The CSV writes both `label` and `group`, so
`analysis.figures.make_all --ablation_csv ...` consumes it directly. The
aggregation does not compute bootstrap CIs; add `ci_low`/`ci_high` columns if a
paired bootstrap is run separately.

The aggregation also validates the stored mask:

- A checkpoint mask that does not match the expected run is skipped.
- `--allow-mask-mismatch` includes mismatched runs.
- Legacy checkpoints without a mask are included with a warning.
- `--strict-mask` skips checkpoints without a mask.
- Checkpoints without a finite `val_r2` are skipped with a warning.

## Generated jobs

`gen_jobs.py` writes scripts under `runs/feature_ablation/jobs/`. Do not edit
them by hand. Edit `_profiles.py` or pass `--common_args` instead.

The generated jobs pass `--prefix ""`, so checkpoints are named
`best_model.pth.tar`. The aggregation also accepts the trainer default
`run_best_model.pth.tar` and legacy `.pth` files.

`gen_jobs.py` creates `<output_dir>/logs/` and writes absolute `#SBATCH` output
and error paths. `submit_all.sh` runs `mkdir -p` on this directory before
submission.
