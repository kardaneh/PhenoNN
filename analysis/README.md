# Post-training analysis

This directory contains scripts that run after a model has been trained.
Nothing here is imported by the `phenonn` package, and none of it is installed
with `pip`.

The scripts read pixelset files and checkpoints, and write outputs under
`runs/`. A full study can take minutes to hours. CI runs the full unit suite,
including the analysis import and command-line tests in
`tests/test_analysis.py` and the feature-ablation tests in
`tests/test_feature_ablation.py`.

## Before you start

Run every command from the repository root. Set `DATA` to the pixelset folder
for your machine.

```bash
cd /path/to/PhenoNN
source .venv/bin/activate

export DATA="${DATA:-$PWD/pixelset1}"
export RUNS="${RUNS:-$PWD/runs}"

export DATA
export RUNS
```

`$DATA` must contain the pixelset files described in the main README. The
analysis modules resolve from the repository root because `phenonn` is
installed in editable mode. Keep `runs/` as a symlink to SSD-backed storage;
the repository ignores the link itself.

## Quick start: evaluate a checkpoint

This example uses the checkpoint from `scripts/smoke_train.sh`. Replace the
checkpoint path with your own.

1. Generate a predictions CSV:

   ```bash
   python -m phenonn.prediction.predict \
       --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth.tar \
       --features_dir "$DATA/era5_pixelset" \
       --target_dir "$DATA/LAI_pixelset" \
       --pft_dir "$DATA/PFT_pixelset" \
       --parent_map "$DATA/selected_pixels01_1.nc" \
       --predict_sites val --predict_years 2000-2002 \
       --output_csv runs/smoke_pred.csv
   ```

   Most diagnostics use this CSV. Its columns include `site_id`, `year`,
   `doy`, `lai_pred`, and `lai_obs`.

2. Compare the model with a climatology baseline:

   ```bash
   python analysis/diagnostics/climatology_baseline.py \
       --predictions runs/smoke_pred.csv --eval_years 2002
   ```

3. Plot annual trends and seasonal metrics:

   ```bash
   python analysis/diagnostics/annual_trend_study.py \
       --predictions runs/smoke_pred.csv \
       --years 2000-2002 --output_dir runs/trend_out
   ```

The generated files are described in `diagnostics/README.md`.

## Choose a task

| Goal | Start with |
|---|---|
| Check input data or site alignment | [Data checks](diagnostics/README.md#data-checks) |
| Compare predictions with climatology or trends | [Prediction comparisons](diagnostics/README.md#prediction-comparisons) |
| Plot per-PFT curves or skill | [PFT and model checks](diagnostics/README.md#pft-and-model-checks) |
| Measure feature importance | [Feature ablation](feature_ablation/README.md) |
| Measure climate sensitivity | [Climate perturbation](perturbation/README.md) |
| Test PFT linearity | [PFT linearity](pft_linearity/README.md) |
| Compare year splits | [Year generalization](year_generalization/README.md) |
| Run the memory kernel (M1) | [Memory analysis](memory_analysis/README.md) |
| Build paper figures F1–F13 | [Figures](figures/README.md) |
| Submit cluster jobs | [Feature-ablation jobs](feature_ablation/jobs/README.md), [perturbation jobs](perturbation/jobs/README.md), [year-split jobs](year_generalization/jobs/README.md) |

## Common data notes

### Checkpoint names

The trainer writes `runs/<experiment>/checkpoints/run_best_model.pth.tar` by
default. Smoke scripts pass `--prefix ""`, so they write
`best_model.pth.tar`. Analysis scripts accept both names, plus legacy `.pth`
checkpoints.

### Two pixel resolutions

LAI and PFT cubes use 0.05 degree pixels with `pix_...` IDs. ERA5 cubes use
deduplicated 0.1 degree pixels with `E...` IDs.

`selected_pixels01_1.nc` maps the two grids. Any script that reads ERA5 data
for a `pix_...` site pool needs:

```bash
--parent_map "$DATA/selected_pixels01_1.nc"
```

Scripts that read only LAI or PFT data do not need the parent map.

## Feature ablation

Feature ablation retrains the model once per non-PFT feature with that feature
masked out. It then compares each run with a baseline that uses all features.

Run the local smoke test first:

```bash
bash scripts/smoke_feature_ablation.sh
```

For the full study:

```bash
export FA_ROOT="$RUNS/feature_ablation"

OUTPUT_DIR="$FA_ROOT" PHENONN_SITE=local \
    bash analysis/feature_ablation/jobs/submit_ablate.sh --study features

.venv/bin/python -m analysis.feature_ablation.aggregate \
    --kind features --runs_dir "$FA_ROOT"
```

The paper profile defaults to `SUBSET=0.25` and `PHENONN_MEM=120G`.
Override with `SUBSET=...` or `PHENONN_MEM=...` for larger nodes.

See `feature_ablation/README.md` for the full workflow, grouped and combo
studies, and result interpretation.

## Climate perturbation

Climate perturbation re-runs a trained checkpoint with selected weather
channels changed and reports the difference from the baseline prediction. Four
scenarios are provided: drought, heatwave, warming plus CO2, and winter frost.

```bash
export EXP=baseline_lstm
export CKPT="$RUNS/baseline/baseline_lstm/checkpoints/run_best_model.pth.tar"
export PERT_OUT="$RUNS/perturbation/$EXP"

.venv/bin/python -m analysis.perturbation.drought \
    --checkpoint "$CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --precip_factor 0.5 --season growing \
    --output_dir "$PERT_OUT"
```

Aggregate all four scenarios after they finish:

```bash
.venv/bin/python -m analysis.perturbation.aggregate_perturbation \
    --output_dir "$PERT_OUT"
```

See [perturbation/README.md](perturbation/README.md) for every scenario
command, parameter defaults, output files, and interpretation. Cluster
submission is documented in
[perturbation/jobs/README.md](perturbation/jobs/README.md).

## PFT linearity

This check compares a no-mixing model's direct prediction with the weighted sum
of pure-PFT predictions. It tests whether the model behaves linearly in the 15
PFT fractions.

```bash
export NOMIX_EXP=nomix_aelstm
export NOMIX_CKPT="$RUNS/$NOMIX_EXP/checkpoints/run_best_model.pth.tar"
export PFT_OUT="$RUNS/pft_linearity/$NOMIX_EXP"

.venv/bin/python -m analysis.pft_linearity.compare_mixing \
    --checkpoint "$NOMIX_CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --purity 0.9 \
    --output_dir "$PFT_OUT"
```

See [pft_linearity/README.md](pft_linearity/README.md) for the optional
mixing-model comparison and the output summary.

## Year generalization

Year generalization trains the same model on different train and validation
year splits. It covers a chronological split, four random splits, and a
cold-to-hot split.

```bash
export YEAR_ROOT="$RUNS/year_generalization"
export RANK_JSON="$YEAR_ROOT/year_temp_rank.json"

.venv/bin/python -m analysis.year_generalization.rank_years_by_temp \
    --features_dir "$DATA/era5_pixelset" \
    --year_start 1992 --year_end 2019 \
    --out "$RANK_JSON"

export SPLIT=chrono
export YEAR_SUBSET=0.25

.venv/bin/python -m analysis.year_generalization.run_year_split \
    --split "$SPLIT" --all_years 1992-2019 \
    --temp_rank_json "$RANK_JSON" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_balanced.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --stats_path "$DATA/norm_stats_1992_2019.json" \
    --co2_path "$DATA/CO2_1700_2023_TRENDYv2024.txt" \
    --val_fraction_of_grid 100 \
    --n_years_per_epoch 30 --n_sites_per_epoch 250000 --n_val_sites 50000 \
    --subset "$YEAR_SUBSET" \
    --type aelstm --hidden_size 256 --num_layers 2 \
    --learning_rate 1e-3 --weight_decay 1e-5 \
    --loss_type huber \
    --pft_mixing --pft_meteo_only \
    --num_epochs 300 --patience 15 \
    --batch_size 64 --amp --num_workers 2 \
    --prefix "" --output_dir "$YEAR_ROOT"
```

The paper profile keeps `YEAR_SUBSET=0.25`. Use `YEAR_SUBSET=0.5` on a
large-memory system. See
[year_generalization/README.md](year_generalization/README.md) for the full
fixed model configuration, split definitions, and job submission. Job details
are in
[year_generalization/jobs/README.md](year_generalization/jobs/README.md).

## Memory analysis

The memory study is design-only. It contains `PROTOCOL.md` and no runnable
scripts yet. See [memory_analysis/README.md](memory_analysis/README.md) for the
status and scope.

## Diagnostics

Every diagnostic script is listed below with a ready-to-paste command and its
main outputs. Run each command from the repository root after the setup steps
above. Use `--help` for the full option list.

### Data checks

#### `inspect_era5_pixelset.py`

Checks ERA5 files for time-length mismatches, site counts, and missing values.

```bash
python analysis/diagnostics/inspect_era5_pixelset.py \
    --era5_dir "$DATA/era5_pixelset" \
    --year_start 2000 --year_end 2001 \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --output_csv runs/era5_missing.csv
```

Omitting `--selected_pixels` and `--parent_map` inspects every site. Omitting
`--output_csv` prints the report to the terminal.

#### `diagnose_grid_alignment.py`

Compares the stored ERA5 coordinates with the LAI/PFT coordinates for the same
site IDs.

```bash
python analysis/diagnostics/diagnose_grid_alignment.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --era5_dir "$DATA/era5_pixelset" --year 2000
```

This script prints to the terminal. Add a raw ERA5 file with
`--era5_grid /path/to/raw/ERA5-Land/file.nc` to inspect the grid geometry.

#### `spatial_plot_selected_pixels.py`

Plots every site in a selected-pixel file on a world map.

```bash
python analysis/diagnostics/spatial_plot_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --output runs/spatial_selected_pixels.png
```

The script uses cartopy when available and falls back to a plain lon/lat plot
otherwise.

#### `print_lai_pft_sites.py`

Prints PFT fractions and all 36 dekadal LAI values for selected sites.

```bash
python analysis/diagnostics/print_lai_pft_sites.py \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --years 2000-2001 \
    --sites pix_0351_01105,pix_0351_01120
```

This script writes no files. Replace the example site IDs with IDs from your
site pool.

#### `pft_diagnostics.py`

Checks one PFT-dominated pool and writes four figures for LAI and meteorological
features.

```bash
python analysis/diagnostics/pft_diagnostics.py \
    --selected_pixels "$DATA/greedy_PFT_10000_0.1/selected_pixels_PFT10.nc" \
    --era5_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --year_start 2000 --year_end 2001 \
    --output_dir runs/pft_diag_out
```

Outputs are `lai_seasonal.png`, `features_hist.png`,
`features_dist_by_year.png`, and `features_seasonal.png`.

#### `diagnose_selected_pixels.py`

Plots the dominant PFT, purity, and spatial density of a selected-pixel file.

```bash
python analysis/diagnostics/diagnose_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_diag.png
```

The script starts one subprocess per year, so keep the year range small.

### Prediction comparisons

#### `climatology_skill.py`

Measures the skill of climatology alone. This is the baseline that a trained
model should beat.

```bash
python analysis/diagnostics/climatology_skill.py \
    --target_dir "$DATA/LAI_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --clim_years 1992-2009 --eval_years 2010-2019 \
    --n_sites 200 --output_dir runs/clim_skill
```

Outputs are `clim_predictions.csv`, `summary.txt`, and `clim_skill.png`.
`--n_sites 0` uses every site.

#### `annual_trend_study.py`

Computes per-site seasonal metrics and plots their yearly medians. The metrics
include total LAI, SOS and EOS, and season length.

```bash
python analysis/diagnostics/annual_trend_study.py \
    --predictions runs/smoke_pred.csv \
    --years 2000-2002 --output_dir runs/trend_out
```

Outputs include `metrics_obs.csv`, `metrics_pred.csv`, and one PNG per metric.
Use `--target_dir "$DATA/LAI_pixelset"` instead of `--predictions` to work from
raw LAI targets.

#### `climatology_baseline.py`

Compares the model with a climatology built from the same predictions CSV.

```bash
python analysis/diagnostics/climatology_baseline.py \
    --predictions runs/smoke_pred.csv --eval_years 2002
```

It reports `SS = 1 - MSE_model / MSE_clim`; a positive value means the model
beats climatology. The default is leave-one-year-out. Use `--pooled` for a
faster but slightly optimistic baseline.

#### `pft_amplitude_correlation.py`

Separates amplitude damping from low predictability for each dominant PFT.

```bash
python analysis/diagnostics/pft_amplitude_correlation.py \
    --predictions runs/smoke_pred.csv \
    --pft_dir "$DATA/PFT_pixelset" \
    --output_dir runs/pft_amp_out
```

Outputs are `pft_diagnostic_summary.csv`, `mean_curves_per_pft.png`, and
`skill_amplitude_per_pft.png`.

### PFT and model checks

#### `pft_curves.py`

Rebuilds a checkpoint trained with `--pft_mixing` and plots the 15 pure PFT
curves, the mixed prediction, and the observations.

```bash
python analysis/diagnostics/pft_curves.py \
    --checkpoint runs/smoke/smoke_lstm/checkpoints/best_model.pth.tar \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --n_sites 2 --year 2002 --output runs/pft_curves.png
```

The script exits with an error if the checkpoint was not trained with
`--pft_mixing`.

## Shared helpers

`_checkpoint_utils.py` contains the checkpoint helpers used by the analysis
scripts:

- `load_state_dict()` accepts current and legacy state-dict keys.
- `find_best_checkpoint()` finds current and legacy checkpoint names.
- `has_completed_run()` checks whether a run has a final checkpoint.

`_cluster.sh` handles HAL, Leonardo, and local runs. It loads site modules when
needed, activates the virtual environment, and adds site-specific `sbatch`
options. HAL and local runs need no modules. Leonardo defaults to
`profile/deeplrn cineca-ai`. Override this with `PHENONN_MODULES`; set it to an
empty value to disable module loading. HAL nodes have at most 120G per job,
which is why the year-split and paper feature-ablation wrappers default to a
0.25 site subset on HAL.
