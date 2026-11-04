# Diagnostics

The scripts here inspect input data, compare predictions, and plot per-PFT
results. Run them from the repository root after following
[the setup steps](../README.md#before-you-start).

Every script accepts `--help`. The commands below show the common path; use
`--help` for every flag.

## Data checks

### `inspect_era5_pixelset.py`

Checks one or more `ERA5_daily_pixelset_{Y}.nc` files. It reports time lengths,
site counts, missing values, and per-feature NaN statistics.

```bash
python analysis/diagnostics/inspect_era5_pixelset.py \
    --era5_dir "$DATA/era5_pixelset" \
    --year_start 2000 --year_end 2001 \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --output_csv runs/era5_missing.csv
```

Omit `--selected_pixels` and `--parent_map` to inspect every site in the files.
Omit `--output_csv` for console output only.

### `diagnose_grid_alignment.py`

Compares the stored ERA5 coordinates with the LAI/PFT coordinates for the same
site IDs.

```bash
python analysis/diagnostics/diagnose_grid_alignment.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --era5_dir "$DATA/era5_pixelset" --year 2000
```

Add `--era5_grid /path/to/raw/ERA5-Land/file.nc` to inspect the raw ERA5 grid.
Use `--max_report N` to change the number of reported sites.

### `spatial_plot_selected_pixels.py`

Plots every site in a `selected_pixels*.nc` file on a world map.

```bash
python -m analysis.diagnostics.spatial_plot_selected_pixels \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --density \
    --output runs/spatial_selected_pixels.png
```

Writes PDF + PNG. `--density` draws a hexbin count map; without cartopy it
falls back to a plain lon/lat grid.

### `print_lai_pft_sites.py`

Prints PFT fractions and all 36 dekadal LAI values for selected sites.

```bash
python analysis/diagnostics/print_lai_pft_sites.py \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --years 2000-2001 \
    --sites pix_0351_01105,pix_0351_01120
```

This script prints to the terminal and writes no files. `--years` accepts a
range such as `2000-2001` or a comma-separated list.

### `pft_diagnostics.py`

Checks one PFT-dominated pixel pool. It produces four figures:

- `lai_seasonal.png`: mean dekadal LAI and its spread across site-years.
- `features_hist.png`: one histogram per meteorological feature.
- `features_dist_by_year.png`: one density curve per year and feature.
- `features_seasonal.png`: mean daily feature cycle on a fixed 366-day grid.

```bash
python analysis/diagnostics/pft_diagnostics.py \
    --selected_pixels "$DATA/greedy_PFT_10000_0.1/selected_pixels_PFT10.nc" \
    --era5_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --year_start 2000 --year_end 2001 \
    --output_dir runs/pft_diag_out
```

`--features` limits the meteorological variables. ERA5 is matched through
`--parent_map`; LAI is read directly with the subset site IDs.

### `diagnose_selected_pixels.py`

Plots the dominant PFT, purity, and spatial density of a selected-pixel file.

```bash
python analysis/diagnostics/diagnose_selected_pixels.py \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --pft_dir "$DATA/PFT_pixelset" \
    --year_start 2000 --year_end 2001 \
    --output runs/selected_pixels_diag.png
```

Output names use the `--output` stem. The script starts one subprocess per year,
so keep the year range small. Use `--pure_threshold`, `--block_deg`, and
`--raster_deg` to change the classification and binning.

## Prediction comparisons

### `climatology_skill.py`

Measures the skill of climatology alone. This is the reference that a trained
model should beat.

```bash
python analysis/diagnostics/climatology_skill.py \
    --target_dir "$DATA/LAI_pixelset" \
    --selected_pixels "$DATA/selected_pixels005_1.nc" \
    --clim_years 1992-2009 --eval_years 2010-2019 \
    --n_sites 200 --output_dir runs/clim_skill
```

Outputs are `clim_predictions.csv`, `summary.txt`, and `clim_skill.png`.
`--n_sites 0` uses every site. The default is 20000 sites and can be slow, so
start with 200.

### `annual_trend_study.py`

Computes per-site seasonal metrics and plots their yearly medians. The metrics
include total LAI, SOS and EOS at 10 percent and 25 percent amplitude, and
season length.

```bash
python analysis/diagnostics/annual_trend_study.py \
    --predictions runs/smoke_pred.csv \
    --years 2000-2002 --output_dir runs/trend_out
```

Outputs include `metrics_obs.csv`, `metrics_pred.csv` when predictions are
present, and one PNG per metric. To work from raw LAI targets instead:

```bash
python analysis/diagnostics/annual_trend_study.py \
    --target_dir "$DATA/LAI_pixelset" \
    --sites_mode random --n_sites 500 \
    --output_dir runs/baseline_trends
```

### `climatology_baseline.py`

Compares model predictions with a climatology built from the same CSV. It
reports `SS = 1 - MSE_model / MSE_clim`; a positive value means the model beats
climatology.

```bash
python analysis/diagnostics/climatology_baseline.py \
    --predictions runs/smoke_pred.csv --eval_years 2002
```

The default is leave-one-year-out. Use `--pooled` for a faster but slightly
optimistic baseline, or `--clim_years 1992-2009` for a fixed reference period.

### `pft_amplitude_correlation.py`

Separates amplitude damping from low predictability for each dominant PFT. It
writes `pft_diagnostic_summary.csv`, `mean_curves_per_pft.png`, and
`skill_amplitude_per_pft.png`.

```bash
python analysis/diagnostics/pft_amplitude_correlation.py \
    --predictions runs/smoke_pred.csv \
    --pft_dir "$DATA/PFT_pixelset" \
    --output_dir runs/pft_amp_out
```

Use `--column norm` for normalized predictions. Use `--pft_year` to select the
PFT map year.

## PFT and model checks

### `pft_curves.py`

Rebuilds a checkpoint trained with `--pft_mixing` and plots the 15 pure PFT
curves, the area-weighted prediction, and the observations for each selected
site-year.

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
`--pft_mixing`. Site selection follows this order: `--sites`, then
`--selected_pixels`, then the checkpoint validation sites.
