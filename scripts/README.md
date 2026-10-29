# scripts: reproducible entry points

These scripts are run from the repository root. They are not part of the
installed `phenonn` package.

Set the data directory before running them:

```bash
cd /path/to/PhenoNN
source .venv/bin/activate

export DATA="${DATA:-$PWD/pixelset1}"
export RUNS="${RUNS:-$PWD/runs}"
```

`$DATA` must contain the pixelset files described in the main README. The
scripts fall back to `<repo>/pixelset1` and then `<repo>/../pixelset1` when
`$DATA` is unset. `runs/` may be a symlink to faster storage; the repository
ignores the link itself.

## `smoke_train.sh`

Runs a tiny end-to-end training job on CPU. It uses two epochs, a few sites,
and a couple of years. It checks the dataset, model, trainer, wrapper, and
checkpoint writer.

```bash
bash scripts/smoke_train.sh
```

Override the data path:

```bash
DATA="$PWD/pixelset1" bash scripts/smoke_train.sh
```

Outputs go to `runs/smoke/smoke_lstm/`. The checkpoint is
`checkpoints/best_model.pth.tar`.

## `run_baseline.sh`

Runs a small baseline with the same trainer as the smoke test, but with more
sites and epochs. It uses a GPU when available and falls back to CPU.

```bash
bash scripts/run_baseline.sh
```

Force a device:

```bash
DEVICE=cpu DATA="$PWD/pixelset1" bash scripts/run_baseline.sh
DEVICE=cuda DATA="$PWD/pixelset1" bash scripts/run_baseline.sh
```

Outputs go to `runs/baseline/baseline_lstm/`.

## `smoke_feature_ablation.sh`

Runs a short feature-ablation study. It checks the seven inputs, trains a
baseline and the selected features on 2008-2009, validates on 2010, and
aggregates the results. It needs no SLURM and loads the data once.

```bash
bash scripts/smoke_feature_ablation.sh
```

Pass feature names as arguments:

```bash
bash scripts/smoke_feature_ablation.sh Tmean tp_sum
```

Outputs go to `runs/feature_ablation_smoke/`.

## `smoke_predict.sh`

Runs inference with the checkpoint written by `smoke_train.sh`. It reads the
data paths and site pool from the checkpoint.

```bash
bash scripts/smoke_predict.sh
```

Use another checkpoint or change the site limit:

```bash
CKPT=/path/to/best_model.pth.tar N_SITES=200 \
  bash scripts/smoke_predict.sh
```

Outputs go to `runs/smoke/smoke_lstm/`. They include `predictions.csv`,
metrics, and diagnostic plots.

## `smoke_resume.sh`

Resumes the smoke run from its latest checkpoint. It inherits the stored
training arguments and only changes the total epoch count.

```bash
bash scripts/smoke_resume.sh
```

Use another checkpoint or run more epochs:

```bash
CKPT=/path/to/checkpoint.pth.tar NUM_EPOCHS=10 \
  bash scripts/smoke_resume.sh
```

Outputs overwrite the checkpoints in `runs/smoke/smoke_lstm/`.

## `phenonn-predict`

Legacy inference helper for pre-cleanup checkpoints. It imports modules that
are no longer part of the current package, so it is not runnable with the
current `phenonn` installation. Use `phenonn predict` instead.
