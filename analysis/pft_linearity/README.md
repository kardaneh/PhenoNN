# PFT linearity

This check asks whether a model without PFT mixing behaves linearly in the 15
PFT fractions. For every evaluation cell the script compares two predictions:

```text
direct = model(x)
mix    = sum_i p_i * model(x with PFT forced to pure PFT i)
```

Here `p_i` is the cell's observed fraction for PFT `i`. The pure-PFT inputs are
one-hot vectors renormalized to sum to one. If `direct` and `mix` agree, the
model's response is close to linear in the PFT fractions. That supports the
additive assumption behind `PFTMixingWrapper` and the greedy unmixing check.

The script also reports per-PFT skill on nearly pure cells, defined by
`--purity` (default `0.9`).

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

Use a checkpoint trained without `--pft_mixing`. The model must ingest the PFT
fractions as ordinary input channels. A mixing checkpoint is accepted only as
an optional comparison model, and the script warns that `mix == direct` by
construction for that model. No non-mixing checkpoint exists in the current
runs tree, so train one before running this study.

## Command

```bash
export NOMIX_EXP=nomix_aelstm
export NOMIX_CKPT="$RUNS/$NOMIX_EXP/checkpoints/run_best_model.pth.tar"
export PFT_OUT="$RUNS/pft_linearity/$NOMIX_EXP"

test -f "$NOMIX_CKPT"
grep -q '"pft_mixing": false' "$RUNS/$NOMIX_EXP/config.json"

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

The check is expensive because each site-year needs 15 extra forward passes.
Use `--n_predict_sites` to keep the first run small.

Add a PFT-mixing checkpoint on the same cells:

```bash
export MIX_EXP=mix_aelstm
export MIX_CKPT="$RUNS/$MIX_EXP/checkpoints/run_best_model.pth.tar"

.venv/bin/python -m analysis.pft_linearity.compare_mixing \
    --checkpoint "$NOMIX_CKPT" \
    --checkpoint_mixing "$MIX_CKPT" \
    --features_dir "$DATA/era5_pixelset" \
    --target_dir "$DATA/LAI_pixelset" \
    --pft_dir "$DATA/PFT_pixelset" \
    --parent_map "$DATA/selected_pixels01_1.nc" \
    --predict_sites val --n_predict_sites 0 \
    --purity 0.9 \
    --output_dir "$PFT_OUT"
```

Both checkpoints must use the same `seq_length` and the same feature
normalization. The script stops with a clear error when they do not.

## Options

- `--checkpoint`: no-mixing checkpoint. Required.
- `--checkpoint_mixing`: optional PFT-mixing checkpoint evaluated on the same
  cells.
- `--predict_sites val|train|all|grid|test`: site pool. The default is `val`.
- `--predict_years 2015-2018`: prediction years. The default is the
  checkpoint validation years.
- `--n_predict_sites N`: random site subsample. `0` uses every site.
- `--purity F`: dominant fraction used to define a nearly pure cell.
- `--parent_map PATH`: map 0.05 degree LAI sites to 0.1 degree ERA5 pixels.

## Outputs

Under `--output_dir`:

```text
mixing_compare.csv   per-site, per-year, per-dekad direct, mix, and observed LAI
summary.txt          agreement statistics and per-PFT skill
mixing_compare.png   direct-vs-mix scatter plot and mean seasonal curves
```

In `summary.txt`, the `direct vs mix` row is the linearity test. An `R2` close
to one means the model behaves almost linearly in the PFT fractions. The
`direct vs obs` and `mix vs obs` rows show whether that linear reconstruction
still has skill against observations. When `--checkpoint_mixing` is supplied,
the summary also compares the mixing model with the direct and linear-mix
predictions.

## Notes

- The PFT fractions are read from the last timestep of each input window.
- Fractions are clamped at zero and renormalized before the weighted sum.
- Plotting uses the non-interactive `Agg` backend, so the command is safe on a
  headless login node.
