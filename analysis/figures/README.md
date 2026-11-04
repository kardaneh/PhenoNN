# Paper figures and tables (F1–F13)

Builds the paper figures and LaTeX tables from prediction bundles. Nothing
here trains a model; it only reads CSV/JSON artefacts under `runs/`. The
package lives under `analysis/` because it is a post-training study, not part
of the installed `phenonn` package.

## Bundle contract

`predict.py` writes most of what the figures need. A run directory is any
subdirectory of `--runs_dir` containing `metrics.json` **or** `history.csv`:

```text
run_<name>/
├── metrics.json              prediction marker (F1–F7); optional for F12-only
├── predictions.csv           F1–F4 (site_id, year, doy, lai_obs, lai_pred, ...)
├── rmse_by_doy.csv           F4
├── history.csv               F12 (epoch, train_loss, val_loss, is_best)
├── climatology_metrics.json  F6 climatology row
├── run_metadata.json         training protocol, seed, git commit
├── predict_metadata.json     prediction split and years
└── config.json               model type and hyper-parameters
```

Training-only runs without `metrics.json` are discovered through
`history.csv`, so F12 can be produced before prediction. F1–F4 and F5–F7
still skip runs that have no prediction bundle.

Old runs without `run_metadata.json` or `predict_metadata.json` still load;
their protocol/split is reported as unknown. F5–F7 collapse multiple seeds of
one architecture into one row and show mean ± std.

## Build everything

Run from the repository root. Protocol B (the unbiased paper split) lives under
`runs/protocol_B`:

```bash
python -m analysis.figures.make_all \
    --runs_dir runs/protocol_B \
    --out_dir paper/figures \
    --run aelstm
```

For the Protocol A report reproduction, point at that tree instead:

```bash
python -m analysis.figures.make_all \
    --runs_dir runs/report_reproduction \
    --out_dir runs/report_reproduction/pilot_figures \
    --run pilot_aelstm_A
```

After R1 finishes, replace `pilot_aelstm_A` with `aelstm_A_s42` and predict
first (see `doc/source/prediction.rst`).

`--run` selects the primary run for F1–F4 (name or substring). F5–F7 use every
run in `--runs_dir` by default; pass `--table_runs` with an explicit comma list
to keep old pilot or preliminary runs out of the paper tables:

```bash
python -m analysis.figures.make_all \
    --runs_dir runs/report_reproduction \
    --out_dir runs/report_reproduction/figures_protocol_A \
    --run aelstm_A_s42_subset05 \
    --table_runs aelstm_A_s42_subset05,lstm_A_s42_subset05,bitransformer_A_s42_subset05 \
    --figures F1,F2,F3,F4,F5,F6,F7,F12
```

F12 uses one run per architecture by default; override it with `--loss_runs`:

```bash
python -m analysis.figures.make_all \
    --runs_dir runs/protocol_B --out_dir paper/figures \
    --figures F12 \
    --loss_runs aelstm_B_s42,bitransformer_B_s42,lstm_B_s42
```

Outputs are PDF + 300 dpi PNG (`.tex` for tables). F12 writes linear and log
loss curves. F2 reads the PFT composition from the run's `config.json`
`pft_dir`; override with `--pft_dir` or a `--pft_csv` from `pft_curves.py`.

F2 and F11 accept deterministic site-year choices from a JSON file:

```bash
cp analysis/figures/figure_choices.example.json \
   analysis/figures/figure_choices.json
# edit the site IDs and years
python -m analysis.figures.make_all \
    --runs_dir runs/protocol_B --out_dir paper/figures \
    --choices analysis/figures/figure_choices.json
```

The file has two optional keys, `F2` (a list of `[site_id, year]` pairs) and
`F11` (`{"site_id": ..., "year": ...}`). Without it, F2 picks the best/worst
RMSE site-years and F11 picks the site-year with the most PFTs above 5 %.

## Optional inputs

F8–F11 and F13 need artefacts produced by the analysis scripts:

```bash
python -m analysis.figures.make_all \
    --runs_dir runs/protocol_B --out_dir paper/figures \
    --ablation_csv runs/feature_ablation/feature_ablation.csv \
    --perturbation_dir runs/perturbation \
    --warming_csv path/to/warming_co2.csv \
    --pft_csv runs/pft_curves/pft_curves.csv \
    --memory_dir runs/memory/memory_kernel
```

`--warming_csv` is produced by the warming/CO₂ sweep:

```bash
python -m analysis.perturbation.warming_co2_sweep \
    --checkpoint runs/protocol_B/aelstm_B_s42/checkpoints/best_model.pth.tar \
    --predict_sites test --predict_years 2010-2019 \
    --delta_t_values 0,1,2,3,4 \
    --co2_modes absolute,detrend \
    --output_dir runs/perturbation
# -> runs/perturbation/warming_co2.csv
```

F8 accepts the aggregate's `label`/`delta_r2` columns directly; it also reads
`group`/`dR2` for older tables. Bootstrap CIs are drawn when `ci_low` and
`ci_high` are present. Missing optional inputs are skipped with a `[skip]`
line, never a crash.

## Figure map

| Figure | Source | Script |
|---|---|---|
| F1–F4 | prediction bundle | `predict.py`, `train_full_ram.py` |
| F5–F7 | bundle metrics + climatology | `analysis/figures/tables.py` |
| F8 | feature ablation | `analysis/feature_ablation` |
| F9–F10 | perturbation + warming sweep | `analysis/perturbation` |
| F11 | per-PFT curves | `analysis/diagnostics/pft_curves.py` |
| F12 | three architecture histories | `train_full_ram.py` → `history.csv` |
| F13 | memory kernel | `analysis/memory_analysis/run_memory_kernel.py` |

Plot implementations live in `analysis/figures/figures.py` as `plot_paper_*`.
The figure code is tested with synthetic bundles in `tests/test_figures.py`.

## Raw and anomaly runs

F1, F2, F4, F5, F6, F11 and F12 use the raw-LAI runs. F3 and F7 need a
dedicated anomaly-mode model (no PFT mixing); the raw model's anomaly skill is
not the paper result:

```bash
python -m analysis.figures.make_all \
    --runs_dir runs/report_reproduction \
    --run aelstm_A_s42_anomaly_subset05 \
    --table_runs aelstm_A_s42_anomaly_subset05 \
    --figures F3,F7
```

## Site-density map (D1)

The dataset overview map is separate from `make_all`: it needs the
selected-pixels NetCDF and cartopy.

```bash
uv pip install -e ".[full,figures]"
# or: uv pip install cartopy

export CARTOPY_DATA_DIR="${CARTOPY_DATA_DIR:-/path/to/cartopy_data}"  # optional

python -m analysis.diagnostics.spatial_plot_selected_pixels \
    --selected_pixels "$PWD/pixelset1/selected_pixels005_balanced.nc" \
    --density \
    --highlight_site pix_2077_04163 \
    --output runs/report_reproduction/D1_site_density.png
```

This writes `D1_site_density.pdf`/`.png` with land, coastlines and borders.
Without cartopy it falls back to a plain lon/lat density panel.

## Legend and line-style rules

Helpers live in `analysis/figures/ipsl_aid.py`.

- Legends sit outside the axes (`loc="outside lower center"` or
  `"outside upper center"`) with `ncol >= 2`; never `loc="best"`.
- Figures use `layout="constrained"`; do not call `tight_layout()` as well.
- Line styles come from `mpltex` via `linestyle_cycle()`/`next_style()`;
  palette colours stay semantic.
- Annotation text (metrics, PFT composition, control ratios) goes in the
  footnote (`figure_note`) or the panel title, never inside the axes.
