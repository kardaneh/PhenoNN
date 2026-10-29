# Climate perturbation

This study measures how a trained PhenoNN checkpoint responds to a controlled
change in one or more weather inputs. It runs the same cells twice: once with
the observed inputs and once with a perturbation installed as a forward
pre-hook. The reported quantity is

```text
Delta LAI = LAI(perturbed) - LAI(baseline)
```

Only the requested channels change. All other inputs are held fixed, so the
result is a ceteris-paribus sensitivity rather than a forecast of a future
climate scenario.

## Before you run

Run every command from the repository root after following
[the analysis setup](../README.md#before-you-start).

```bash
cd /path/to/PhenoNN
source .venv/bin/activate

export DATA="${DATA:-$PWD/pixelset1}"
export RUNS="${RUNS:-$PWD/runs}"

export DATA
export RUNS
```

The checkpoint normally stores the training data paths. Pass the data flags
explicitly when those paths are not available on the machine running the
analysis.

## Run all four scenarios

The existing baseline checkpoint is usable:

```bash
export EXP=baseline_lstm
export CKPT="$RUNS/baseline/baseline_lstm/checkpoints/run_best_model.pth.tar"
export PERT_OUT="$RUNS/perturbation/$EXP"

test -f "$CKPT"
```

```bash
.venv/bin/python -m analysis.perturbation.drought \
    --checkpoint "$CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --precip_factor 0.5 --season growing \
    --output_dir "$PERT_OUT"

.venv/bin/python -m analysis.perturbation.heatwave \
    --checkpoint "$CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --delta_t 5 --season summer \
    --output_dir "$PERT_OUT"

.venv/bin/python -m analysis.perturbation.warming_co2 \
    --checkpoint "$CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --delta_t 2 --delta_co2 100 \
    --output_dir "$PERT_OUT"

.venv/bin/python -m analysis.perturbation.winter_frost \
    --checkpoint "$CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --delta_t 8 --season winter \
    --output_dir "$PERT_OUT"
```

## Scenarios

| Script | Default change | Channels | Default season |
|---|---|---|---|
| `drought.py` | `tp_sum` and `SMI` times `0.5` | precipitation and its 30-day proxy | growing |
| `heatwave.py` | `+5 C` | `Tmin`, `Tmax`, `Tmean` | summer |
| `warming_co2.py` | `+2 C` and `+100 ppm` | temperature and `co2` | year |
| `winter_frost.py` | `-8 C` | `Tmin`, `Tmax`, `Tmean` | winter |

Season windows use a non-leap 365-day calendar:

| Name | Day-of-year range |
|---|---|
| `year` | whole window |
| `spring` | 60 to 151 |
| `summer` | 152 to 243 |
| `autumn` | 244 to 334 |
| `winter` | 335 to 59, wrapping the new year |
| `growing` | 91 to 304 |

Set `--season year` to apply the perturbation to the whole 720-day input
window. Set `--delta_co2 0` for a pure warming run, or `--delta_t 0` for a pure
CO2 run.

## Common options

- `--predict_sites val|train|all|grid|test`: site pool. The default is `val`.
- `--predict_years 2015-2018`: prediction years. The default is the
  checkpoint validation years.
- `--n_predict_sites N`: random site subsample. `0` uses every site.
- `--selected_pixels PATH`: restrict prediction to a selected-pixel file.
- `--sites ID1,ID2`: explicit site IDs.
- `--parent_map PATH`: map 0.05 degree LAI sites to 0.1 degree ERA5 pixels.
- `--batch_size 64`: inference batch size.
- `--seed 42`: sampling seed.

If the checkpoint was trained with ablated features, the same channels are
zeroed during perturbation inference. Daily-output checkpoints are sampled at
the 36 dekad positions so every scenario writes the same table layout.

## Warming + CO2 sweep (F10)

`warming_co2.py` runs one scenario.  For the F10 response table, sweep a
temperature grid and CO2 modes in one process:

```bash
.venv/bin/python -m analysis.perturbation.warming_co2_sweep \
    --checkpoint "$CKPT" \
    --predict_sites test --predict_years 2010-2019 \
    --delta_t_values 0,1,2,3,4 \
    --co2_modes absolute,detrend \
    --output_dir "$PERT_OUT"
```

It writes `$PERT_OUT/warming_co2.csv`, with one row per
`(delta_t, co2_mode)`: mean ΔLAI, median ΔSOS/ΔEOS/ΔLOS at the 10 % threshold,
the number of site-years, and an `out-of-range` flag when the perturbed CO2
falls outside the checkpoint's training-year CO2 range.  Pass this CSV to
`analysis.figures.make_all --warming_csv ...`.

## Aggregate the scenarios

After all four scenarios have written their `deltas.csv` files:

```bash
.venv/bin/python -m analysis.perturbation.aggregate_perturbation \
    --output_dir "$PERT_OUT"
```

The aggregation writes `comparison.csv`, `comparison.txt`, and
`comparison.png`. The table reports the mean, median, spread, summer mean, and
relative change for each scenario.

## Outputs

Each scenario writes under `<output_dir>/<scenario>/`:

```text
deltas.csv          per-site, per-year, per-dekad baseline and perturbed LAI
summary.txt         aggregate Delta LAI statistics
delta_seasonal.png  mean seasonal Delta LAI curve
delta_hist.png      distribution of per-site-year mean Delta LAI
```

The aggregation writes next to the scenario folders:

```text
comparison.csv
comparison.txt
comparison.png
```

## Cluster jobs

The job wrappers are documented in [jobs/README.md](jobs/README.md). They run
all four scenarios for one checkpoint, or evaluate several checkpoints and then
aggregate the results. They use `analysis/_cluster.sh` for the environment,
site presets, dry-run mode, and local execution.

## Notes

- The perturbation hook edits normalized inputs after converting them back to
  physical units. Log-transformed features are handled with `log1p` and
  `expm1`.
- The perturbation does not change VPD, PET, radiation, daylength, or other
  channels unless that feature is listed explicitly.
- The four scenarios are designed as sensitivity baselines. They are not
  coupled climate-model projections.
