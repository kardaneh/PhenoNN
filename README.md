# PhenoNN: Phenology Neural Networks

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Docs](https://img.shields.io/badge/docs-sphinx-brightgreen.svg)](https://phenonn.readthedocs.io/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

PhenoNN is a deep learning framework for predicting **Leaf Area Index
(LAI)** from climate data. It reads a two-year window of daily ERA5-Land
meteorology, CO2 concentration, and 15 Plant Functional Type (PFT)
fractions, and predicts the 36 dekadal LAI observations of a target year.

PhenoNN was developed in the context of the **AI4PEX project** (Research
Focus LAND). AI4PEX aims to improve the representation of terrestrial
ecosystems in climate models, with four research directions:

- **Hybrid modelling and history matching** — better predict
  instantaneous vegetation responses to water and heat stress.
- **Deep learning for phenology** — use sequence models to simulate
  phenology and explore plant carbon dynamics and mortality drivers.
- **Temperature sensitivity of decomposition** — understand how
  temperature affects soil decomposition and land–atmosphere carbon
  exchange.
- **Land–atmosphere feedbacks** — improve the representation of the
  processes that control energy and carbon feedbacks, reducing
  uncertainties in projected warming.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data layout](#data-layout)
- [Command line interface](#command-line-interface)
- [Python API](#python-api)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Features

- **Four architectures** — `RNN_LSTM`, `AELSTM` (LSTM + transformer
  self-attention), `BiTransformerV2` (two-stage transformer with an
  explicit weather → stress → LAI factorisation),
  `CausalSparseTransformer` (Informer-style causal encoder-decoder).
- **Pixelset pipeline** — reads per-year NetCDF cubes keyed by
  `site_id`; no per-site CSVs required.
- **RAM-resident trainer** — loads a working set of sites and years
  once, then sub-selects sample indices each epoch. No per-epoch disk
  I/O.
- **PFT decomposition** — the model can output 15 pure-LAI curves that
  a wrapper mixes into a single cell-level LAI using the local PFT
  fractions.
- **NaN-safe loss** — missing dekads are masked out of every reduction.
  Optional peak, correlation and amplitude penalties complement the
  base loss.
- **Causal masking** — transformer and LSTM variants can be made
  strictly causal so that no future weather leaks into earlier
  predictions.
- **Post-training analysis** — feature ablation, climate perturbation, PFT
  linearity, year generalization, and climatology/diagnostic baselines.
- **GPU acceleration** — CUDA support, optional bfloat16 autocast.

---

## Installation

### Prerequisites

- Python **3.10** or higher
- PyTorch 2.0 or higher (installed by the `full` extra)
- CUDA-capable GPU (optional)
- Git

### Quick install

1. **Install the `uv` package manager**:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Clone and install PhenoNN**:

   ```bash
   git clone https://github.com/kardaneh/PhenoNN.git
   cd PhenoNN
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[full]"       # models, training, prediction, figures
   uv pip install -e ".[full,dev]"   # + pre-commit and ruff
   uv pip install -e ".[full,docs]"  # + Sphinx
   ```

   For a CPU-only machine, install the CPU torch wheel first:

   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install -e ".[full]"
   ```

3. **Verify the installation**:

   ```bash
   python tests/test_phenonn_installation.py
   # ends with: "All checks passed — PhenoNN is ready to use."
   ```

For detailed instructions, see the
[Installation Guide](https://phenonn.readthedocs.io/en/latest/installation.html).

---

## Quick start

### Train a model

The training entry point reads three per-year pixelset NetCDF files
plus a few optional auxiliary files:

```bash
python -m phenonn.training.train_full_ram \
    --features_dir /path/to/era5_pixelset \
    --target_dir   /path/to/LAI_pixelset \
    --pft_dir      /path/to/PFT_pixelset \
    --selected_pixels /path/to/selected_pixels005_balanced.nc \
    --parent_map      /path/to/selected_pixels01_1.nc \
    --stats_path      /path/to/norm_stats.json \
    --co2_path        /path/to/CO2.txt \
    --train_years 1992-2006 --val_years 2007-2009 --test_years 2010-2019 \
    --norm_years 1992-2006 \
    --protocol B \
    --type lstm --hidden_size 256 --num_layers 2 \
    --pft_mixing --pft_meteo_only \
    --num_epochs 100 \
    --output_dir runs --experiment my_run
```

Protocol B is the unbiased paper split. For the report reproduction
(Protocol A), use `--train_years 1992-2009 --val_years 2010-2019
--test_years 2010-2019 --protocol A` instead. Pass `--norm_years` with the
years used to build `--stats_path`, and keep `--clim_years` inside the
training years: the trainer refuses either window if it includes validation or
test years. See [Evaluation protocols](#evaluation-protocols).

Checkpoints and logs land in `runs/my_run/checkpoints/` and
`runs/my_run/logs/`. Keep `runs/` as a symlink to SSD-backed storage on the
cluster; the repository ignores the link itself.

### Predict with a trained checkpoint

```bash
python -m phenonn.prediction.predict \
    --checkpoint runs/my_run/checkpoints/run_best_model.pth.tar \
    --predict_sites test \
    --protocol B --test_years 2010-2019 \
    --output_csv runs/my_run/predictions.csv
```

The predictor reads the data directories from the checkpoint's saved
arguments. It writes a per-`(site, year, dekad)` CSV plus diagnostic
plots (`*_pred_vs_obs.png`, `*_lai_curves.png`,
`*_lai_curves_all.png`) and a metrics summary text file.

### Evaluation protocols

`--protocol` selects one of three splits. It is case-insensitive and applies
to both training and prediction.

| Protocol | Train | Validation | Test | Why |
|---|---|---|---|---|
| `A` | 1992–2009 | 2010–2019 | 2010–2019 | **Reproduction.** Matches the report, which used one decade for early stopping and reporting. Validation and test overlap by design, so these test metrics are optimistic and not a headline claim. |
| `B` | 1992–2006 | 2007–2009 | 2010–2019 | **Unbiased paper split.** Train, validation and test are disjoint, so model selection never sees the test decade. Use this for headline claims. |
| `none` | user-defined | user-defined | optional | Ad-hoc runs. Leakage checks still run, but no protocol-specific overlap rule is applied. |

Protocol B needs `--test_years`. Protocol A needs `--test_years` equal to
`--val_years`. Pass `--norm_years` with the years used to build `--stats_path`,
and keep `--clim_years` inside the training years; otherwise the run stops
with a `LeakageError` (override with `--allow_leakage` only for a documented
diagnostic).

### Run a smoke test

A minimal end-to-end run on a handful of sites for two epochs, suitable
as a first check on a new machine:

```bash
bash scripts/smoke_train.sh
```

If your data is not in the default location:

```bash
DATA="$PWD/pixelset1" bash scripts/smoke_train.sh
```

---

## Data layout

Three per-year NetCDF files, one per variable family. LAI/PFT targets use
native `pix_LLLL_OOOOO` site IDs. On the report route ERA5 features use
deduplicated 0.1° `E...` cell IDs and `selected_pixels01_1.nc` maps every
target site to its parent cell; the legacy identity route stores ERA5 on the
same `pix_...` sites.

| File | Variable | Dimensions |
|---|---|---|
| `ERA5_daily_pixelset_{YYYY}.nc` | 12 dynamic variables: 10 base meteorology + `daylength` + `SMI` | `(time, site)` |
| `LAI_dekadal_{YYYY}.nc` | `LAI` | `(dekad, site)` |
| `PFTmap_{YYYY}.nc` | `pft_frac` (15 PFTs, ORCHIDEE order) | `(pft, site)` |

Plus auxiliary files that apply to all years:

- `selected_pixels005_balanced.nc` - the 0.05° training site pool.
- `selected_pixels01_1.nc` - optional parent map for the 0.1° ERA5 route.
- `norm_stats.json` - per-feature `{"mean": ..., "std": ...}` statistics.
- a CO2 lookup file with one `Annee_YYYY=VALUE` line per year.

On the report route, pass both `--selected_pixels selected_pixels005_balanced.nc`
and `--parent_map selected_pixels01_1.nc` to training and prediction.

For a target year `Y`, the dataset concatenates the `Y-1` and `Y` feature
files and keeps the last `seq_length` days. **The `Y-1` file must be
present** for any sample of year `Y` to be built.

See the
[Data Preparation Guide](https://phenonn.readthedocs.io/en/latest/data_preparation.html)
for the full layout.

---

## Command line interface

Two sub-commands are supported:

```bash
phenonn train      # Train a model (RAM working-set trainer)
phenonn predict    # Run inference with a trained checkpoint
phenonn --version
phenonn --help
```

Both are thin wrappers around the `python -m` entry points:

- `phenonn train` → `python -m phenonn.training.train_full_ram`
- `phenonn predict` → `python -m phenonn.prediction.predict`

For the full list of flags, run `phenonn train --help` or
`phenonn predict --help`. The SLURM scripts under `analysis/*/jobs/` use
the same two entry points.

---

## Python API

```python
import torch

from phenonn.data import RamLAIDataset
from phenonn.utils import build_model_pft

# 1. Dataset — loads the requested sites and years into RAM once.
ds = RamLAIDataset(
    features_dir="/path/to/era5_pixelset",
    target_dir="/path/to/LAI_pixelset",
    pft_dir="/path/to/PFT_pixelset",
    years=[2010, 2011],
    site_ids=["pix_0123_04567", "pix_0124_04568"],
    seq_length=720,
    norm_stats=None,            # or a dict with mean/std per feature
    normalize_lai=True,
    # parent_map=load_parent01_map("/path/to/selected_pixels01_1.nc"),
    # ^ required for the deduplicated 0.1° ERA5 route
)
features, targets = ds[0]      # (n_features, 720)  and  (1, 36)

# 2. Model — pick an architecture via the args namespace.
from types import SimpleNamespace
args = SimpleNamespace(
    type="lstm",
    seq_length=720,
    hidden_size=128,
    num_layers=2,
    pft_mixing=True,
    pft_meteo_only=True,
    normalize_lai=True,
    # architecture-specific fields
    d_model=None, nhead=4, forward_expansion=4,
    dropout1=0.0, dropout2=0.0, dropout_att=0.0,
    dropout_trans=0.0, dropout_encoder=0.0,
    feed_forward_trans1=4, feed_forward_trans2=4,
    feed_forward_encoder=4,
    n_attn_blocks=2, stress_dim=8, num_layers1=2, d_layers=2,
    embed_size=64, daily_lai=False,
)
model = build_model_pft(args, norm_stats=None)

# 3. Forward pass
preds = model(features.unsqueeze(0))    # (1, 1, 36)
```

Public architectures exposed by `phenonn.models`:

| Class | Role |
|---|---|
| `RNN_LSTM` | LSTM encoder + 1×1 conv head |
| `AELSTM` | LSTM encoder + transformer self-attention |
| `BiTransformerV2` | two-stage transformer: weather → stress → LAI |
| `CausalSparseTransformer` | Informer-style causal encoder + decoder |

Plus `EncoderTorch` (plain transformer encoder) and
`SinusoidalPositionalEncoding`.

---

## Project structure

```text
PhenoNN/
├── phenonn/                          # importable package
│   ├── __init__.py                   # public surface re-exports
│   ├── __main__.py                   # `python -m phenonn`
│   ├── cli.py                        # console script entry point
│   ├── version.py
│   ├── data/
│   │   └── lai_dataset.py            # LAIDataset, RamLAIDataset
│   ├── models/
│   │   ├── rnn.py                    # RNN_LSTM, RNN_GRU, BaseRNN
│   │   ├── aelstm.py                 # AELSTM + SelfAttentionBlock
│   │   ├── bitransformer.py          # BiTransformerV2
│   │   ├── transformer.py            # EncoderTorch, CausalSparseTransformer
│   │   ├── attn_lstm.py              # AttnLSTM
│   │   └── transformerbis.py         # legacy BiTransformer (not exported)
│   ├── training/
│   │   └── train_full_ram.py         # main, load_shared, train_one_config
│   ├── prediction/
│   │   └── predict.py                # main, _parse_year_spec, _resolve_sites
│   ├── utils/
│   │   ├── config.py                 # FEATURES, PFT, dekad constants
│   │   ├── wrappers.py               # DailyWrapper, Every10DaysWrapper,
│   │   │                             #   PFTMixingWrapper, permuteWrapper
│   │   ├── model_loader.py           # build_model, build_model_pft
│   │   ├── evaluater.py              # nan_safe_loss, make_loss_fn
│   │   ├── diagnostics.py            # plot_* helpers
│   │   ├── logger.py                 # rich-based Logger
│   │   ├── model_utils.py            # ModelUtils (checkpoint helpers)
│   │   └── utils.py                  # EasyDict, FileUtils
├── analysis/                         # post-training studies (not packaged)
│   ├── _checkpoint_utils.py          # state_dict / best-checkpoint helpers
│   ├── _cluster.sh                   # shared cluster and environment helpers
│   ├── diagnostics/                  # one-shot data/model diagnostics
│   ├── figures/                      # paper figures F1–F13 and tables
│   ├── feature_ablation/             # leave-one-feature-out ablation
│   ├── perturbation/                 # weather-perturbation scenarios
│   ├── pft_linearity/                # PFT-fraction linearity check
│   ├── year_generalization/          # train/validation year splits
│   └── memory_analysis/              # memory kernel (M1) + protocol
├── tests/                            # unit tests (unittest)
├── scripts/                          # hand-run diagnostics and smoke tests
├── legacy/                           # archived modules (removed 2027-01)
├── doc/                              # Sphinx documentation
├── pyproject.toml
└── README.md
```

---

## Testing

The test suite uses the standard library's `unittest` — no `pytest`
needed. All tests are CPU-only and use no external data files.

```bash
python tests/test_phenonn_installation.py
python -m unittest discover -s tests
```

The full workflow:

```bash
python tests/test_phenonn_installation.py
python -m unittest discover -s tests

uvx ruff@0.5.0 check . && uvx ruff@0.5.0 format --check .

cd doc && make clean && make html; cd ..
uv build
```

The pipeline smoke test (`bash scripts/smoke_train.sh`) goes further and
exercises the training loop end to end on a handful of sites.

The package uses lazy imports: `phenonn --version` and `phenonn --help` do not
import torch, and `phenonn train --help` / `phenonn predict --help` use the
lightweight parser modules. Heavy training, prediction and plotting modules are
imported only when their names are used.

---

## Documentation

Full documentation is at
[https://phenonn.readthedocs.io/](https://phenonn.readthedocs.io/).

- [Installation Guide](https://phenonn.readthedocs.io/en/latest/installation.html)
- [Quickstart](https://phenonn.readthedocs.io/en/latest/quickstart.html)
- [Data Preparation](https://phenonn.readthedocs.io/en/latest/data_preparation.html)
- [Training Guide](https://phenonn.readthedocs.io/en/latest/training.html)
- [Prediction Guide](https://phenonn.readthedocs.io/en/latest/prediction.html)
- [API Reference](https://phenonn.readthedocs.io/en/latest/api/modules.html)

Build the documentation locally:

```bash
cd doc
uv pip install -e ".[full,docs]"
make clean
make html
firefox build/html/index.html
```

---

## Contributing

Contributions are welcome. Before opening a pull request:

```bash
uv pip install -e ".[full,dev,docs]"
pre-commit install
pre-commit run --all-files
python -m unittest discover -s tests
cd doc && make clean && make html
```

See the
[Contributing Guide](https://phenonn.readthedocs.io/en/latest/contributing.html)
for the conventions (branch names, docstring style, the public-API
rule, and the pull-request checklist).

---

## License

This work is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to:

- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

Under the following terms:

- **Attribution** — give appropriate credit, provide a link to the
  license, and indicate if changes were made.
- **NonCommercial** — no commercial use.
- **ShareAlike** — distribute your contributions under the same license.

![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)

For more details, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.

---

## Citation

If you use PhenoNN in your research, please cite:

```bibtex
@software{ardaneh_phenonn_2024,
  author = {Barbu, Stefan and Ardaneh, Kazem and Carlin, Esteban},
  title  = {PhenoNN: Deep Learning for Phenology Prediction},
  year   = {2024},
  url    = {https://github.com/kardaneh/PhenoNN},
  doi    = {10.5281/zenodo.xxxxxxx}
}
```

---

## Acknowledgments

- CNRS for institutional support
- Max-Planck-Institute for Biogeochemistry Jena
- All contributors and users of PhenoNN

---

## Contact

- **Authors**: Kazem Ardaneh, Stefan Barbu, Esteban Carlin
- **Contact**: Kazem Ardaneh
- **Email**: kardaneh@ipsl.fr
- **GitHub**: https://github.com/kardaneh
- **Institution**: CNRS

---

**PhenoNN** — deep learning for phenology prediction.
