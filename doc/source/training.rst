Training
========

Entry point
-----------

The training entry point is the RAM-resident trainer. It loads the raw
features, targets and PFT vectors for a working set of sites and years
into memory once, then sub-selects sample indices at every epoch. There
is no per-epoch disk I/O.

Command-line invocation::

    python -m phenonn.training.train_full_ram [options]

or, once the package is installed::

    phenonn train [options]

Both forms are equivalent -- ``phenonn train`` forwards its arguments to
the module's :func:`main`.

The equivalent Python API is described below.

Data arguments
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--features_dir``
     - Folder of ``ERA5_daily_pixelset_{Y}.nc``.
   * - ``--target_dir``
     - Folder of ``LAI_dekadal_{Y}.nc``.
   * - ``--pft_dir``
     - Folder of ``PFTmap_{Y}.nc``.
   * - ``--selected_pixels``
     - ``selected_pixels005_1.nc`` (or the balanced variant). Prefer this
       over ``--valid_dir`` on the pixel-set pipeline.
   * - ``--valid_dir``
     - Folder of ``valid_pixels_{Y}.nc``. Used when
       ``--selected_pixels`` is not given.
   * - ``--parent_map``
     - Optional ``selected_pixels01_1.nc`` (0.05 deg to 0.1 deg parent
       map). Reads ERA5 features from the deduplicated 0.1 deg file.
   * - ``--stats_path``
     - ``norm_stats.json`` (optional). Enables log1p and z-scoring.
   * - ``--co2_path``
     - CO2 lookup (optional).

Year splits
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--train_years``
     - e.g. ``1992-2009`` or ``1992,2000,2005``.
   * - ``--val_years``
     - e.g. ``2010-2019``. Must be disjoint from train years when
       ``--val_fraction_of_grid`` is at least 100 (overlap mode).
   * - ``--test_years``
     - Held-out decade, e.g. ``2010-2019``. Required by Protocol B and
       used only for leakage checks, never for early stopping.
   * - ``--protocol``
     - ``A`` (reproduction), ``B`` (unbiased paper split) or ``none``
       (ad-hoc). Case-insensitive.
   * - ``--norm_years``
     - Years used to build ``--stats_path``. Must be a subset of
       ``--train_years``.
   * - ``--clim_years``
     - Years used for the anomaly climatology. Defaults to
       ``--train_years``; may not include validation or test years.
   * - ``--allow_leakage``
     - Downgrade the leakage errors to warnings. Diagnostic runs only.
   * - ``--selection_rule``
     - ``val_loss`` (paper default) or ``val_r2`` for the diagnostic
       best-R2 checkpoint.

Protocol A (reproduction) trains on 1992--2009 and reports on 2010--2019,
so validation and test share a decade and the reported scores are
optimistic. Protocol B (unbiased) trains on 1992--2006, validates on
2007--2009 and tests on 2010--2019; the three sets are disjoint. Use
Protocol B for headline claims and Protocol A only to compare with the
original report.

Site pool and sampling
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--subset``
     - Keep this fraction of the site pool, in ``(0, 1]``. Applied
       before the train/val split, so RAM stays bounded.
   * - ``--n_sites_per_epoch``
     - How many sites the sampler draws per epoch.
   * - ``--n_years_per_epoch``
     - How many years the sampler draws per epoch.
   * - ``--n_val_sites``
     - Size of the validation pool.
   * - ``--val_fraction_of_grid``
     - Fraction of the pool reserved for validation. Value >= 100
       selects *overlap mode* (train and val share the pool, years must
       be disjoint).
   * - ``--row_min``, ``--row_max``
     - Sub-grid bbox on the latitude index, for the ``--valid_dir``
       pool.
   * - ``--col_min``, ``--col_max``
     - Sub-grid bbox on the longitude index, for the ``--valid_dir``
       pool.

Model selection
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--type``
     - One of ``lstm``, ``aelstm``, ``bitransformer_v2``,
       ``transformer``.
   * - ``--seq_length``
     - Input window length in days (default 730).
   * - ``--hidden_size``
     - LSTM/GRU hidden size, or stage-2 transformer width.
   * - ``--d_model``
     - Stage-1 transformer width for ``bitransformer_v2`` and
       ``attnlstm``. Defaults to ``--hidden_size``.
   * - ``--num_layers``
     - Number of blocks in the encoder.
   * - ``--num_layers1``
     - Stage-1 transformer blocks (``bitransformer_v2``,
       ``attnlstm``).
   * - ``--nhead``
     - Attention heads.
   * - ``--forward_expansion``
     - FFN multiplier for ``aelstm``.
   * - ``--dropout1``, ``--dropout2``
     - Dropout for stages 1 and 2.
   * - ``--stress_dim``
     - Stress-vector size (``bitransformer_v2``, ``attnlstm``).

PFT mixing
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--pft_mixing``
     - The model outputs 15 pure-LAI curves; ``PFTMixingWrapper`` mixes
       them into a single cell-level LAI using the local PFT fractions.
   * - ``--pft_meteo_only``
     - With ``--pft_mixing``, the base model receives only the
       meteorology / cyclic / CO2 channels. The PFT fractions are used
       only as mixing weights.
   * - ``--pft_nonneg``
     - With ``--pft_mixing``, soft-floor each per-PFT pure LAI at
       physical 0.

Target
------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--no_normalize_lai``
     - Skip LAI z-scoring (features stay z-scored).
   * - ``--anomaly_mode``
     - Subtract each site's per-dekad climatology from the target.
   * - ``--clim_years``
     - Years used to build the climatology (default ``1992-2010``).
   * - ``--clim_target_dir``
     - Folder of climatology source files. Defaults to ``--target_dir``.
   * - ``--daily_lai``
     - Output 365 daily values instead of 36 dekads; the loss covers
       all days.
   * - ``--daily_target_dir``
     - Folder of ``LAI_daily_{Y}.nc`` files. Required with
       ``--daily_lai``.

Loss
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--loss_type``
     - One of ``mse``, ``mae``, ``huber``, ``smoothl1``, ``nmse``,
       ``nmae``, ``gradient``.
   * - ``--huber_beta``
     - Huber transition point.
   * - ``--gradient_loss_weight``
     - Weight of the temporal-gradient term (with
       ``loss_type=gradient``).
   * - ``--peak_penalty_weight``
     - Weight of the yearly-peak L1 penalty.
   * - ``--corr_loss_weight``
     - Weight of the ``1 - Pearson`` shape penalty. Scale-invariant
       complement to MSE / Huber.
   * - ``--amp_loss_weight``
     - Weight of the ``|std pred - std target|`` amplitude penalty.

Optimisation
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--learning_rate``
     - Initial learning rate (default 1e-3).
   * - ``--weight_decay``
     - Adam weight decay.
   * - ``--batch_size``
     - Samples per batch.
   * - ``--num_epochs``
     - Maximum number of epochs.
   * - ``--patience``
     - Early-stopping patience.
   * - ``--max_grad_norm``
     - Gradient-clipping threshold.
   * - ``--amp``
     - Enable bfloat16 autocast.
   * - ``--compile``
     - Wrap the model in ``torch.compile``.
   * - ``--no_fused_adam``
     - Disable the fused CUDA Adam kernel.
   * - ``--num_workers``
     - DataLoader workers.

Output and resume
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``--output_dir``
     - Root output directory (default ``runs``).
   * - ``--experiment``
     - Experiment name; the checkpoint goes to
       ``<output_dir>/<experiment>/``.
   * - ``--seed``
     - Reproducibility seed.
   * - ``--resume``
     - Path to a ``.pth.tar`` checkpoint from a previous run.

Loss penalties
--------------

The three optional penalties can be combined with any base loss:

- ``corr_loss_weight`` penalises ``1 - Pearson(pred, target)`` per
  sample over the 36 dekads. It is scale-invariant, so a flat prediction
  scores ``r = 0`` and is penalised. Use it when the model damps the
  seasonal cycle but tracks the phase.
- ``amp_loss_weight`` penalises ``|std_t pred - std_t target|`` per
  sample. Use it when the seasonal amplitude is too small.
- ``peak_penalty_weight`` penalises ``|max_t pred - max_t target|`` per
  sample. Use it when the yearly maximum is systematically off.

All three are NaN-safe and reduce only over the finite target positions.

Outputs
-------

For an experiment ``my_run``, the trainer writes under
``<output_dir>/my_run/``:

- ``checkpoints/run_best_model.pth.tar`` -- best validation checkpoint
  (the ``run`` prefix is the default ``--prefix``; pass ``--prefix ""`` for
  ``best_model.pth.tar``)
- ``checkpoints/run_final_model_epoch<N>.pth.tar`` -- final checkpoint
- ``logs/train.log`` -- per-epoch metrics
- ``loss_history.png``, ``metric_history.png`` -- training curves
- ``config.json`` -- the exact arguments used

The checkpoint is a dictionary with ``state_dict`` (legacy files may use
``model_state_dict``), ``optimizer_state_dict``, ``scheduler_state_dict``,
the training
arguments, the feature normalisation statistics, the CO2 lookup, the
site pools, and the current validation metrics. The predictor loads
everything it needs from it.

Examples
--------

A minimal run on CPU (useful as a smoke test)::

    python -m phenonn.training.train_full_ram \
        --features_dir /path/to/era5_pixelset \
        --target_dir   /path/to/LAI_pixelset \
        --pft_dir      /path/to/PFT_pixelset \
        --selected_pixels /path/to/selected_pixels005_1.nc \
        --parent_map   /path/to/selected_pixels01_1.nc \
        --train_years 2000-2001 --val_years 2002 \
        --val_fraction_of_grid 100 \
        --subset 0.001 \
        --n_sites_per_epoch 8 --n_years_per_epoch 2 --n_val_sites 8 \
        --type lstm --hidden_size 32 --num_layers 1 \
        --pft_mixing --pft_meteo_only \
        --loss_type huber --num_epochs 2 --batch_size 4 \
        --output_dir runs --experiment smoke_lstm

A realistic cluster run::

    python -m phenonn.training.train_full_ram \
        --features_dir /path/to/era5_pixelset \
        --target_dir   /path/to/LAI_pixelset \
        --pft_dir      /path/to/PFT_pixelset \
        --selected_pixels /path/to/selected_pixels005_balanced.nc \
        --parent_map   /path/to/selected_pixels01_1.nc \
        --stats_path   /path/to/norm_stats_1992_2019.json \
        --co2_path     /path/to/CO2_1700_2023_TRENDYv2024.txt \
        --train_years 1992-2009 --val_years 2010-2019 \
        --val_fraction_of_grid 100 \
        --n_sites_per_epoch 250000 --n_years_per_epoch 30 --n_val_sites 50000 \
        --subset 0.5 \
        --type aelstm --hidden_size 256 --num_layers 2 \
        --pft_mixing --pft_meteo_only \
        --loss_type huber --num_epochs 300 --patience 15 \
        --batch_size 64 --amp \
        --output_dir runs_year_gen --experiment my_aelstm_run

Python API
----------

The trainer is composable. ``load_shared`` does the expensive disk load
once and returns a container; ``train_one_config`` trains one
configuration on it. This is what the in-process sweeps use to avoid
re-reading the same data for each configuration.

.. code-block:: python

    from phenonn.training.train_full_ram import (
        load_shared, parse_args, setup_device_and_seed,
        setup_directories_and_logging, train_one_config,
    )

    # 1. Prepare directories, logger, device, then load the working set once.
    cli_args = parse_args([
        "--features_dir", ".../era5_pixelset",
        "--target_dir",   ".../LAI_pixelset",
        "--pft_dir",      ".../PFT_pixelset",
        "--selected_pixels", ".../selected_pixels005_1.nc",
        "--parent_map", ".../selected_pixels01_1.nc",
        "--train_years", "2010-2015", "--val_years", "2016",
        "--type", "lstm", "--hidden_size", "128",
        "--pft_mixing", "--pft_meteo_only",
        "--num_epochs", "30",
        "--output_dir", "runs", "--experiment", "api_demo",
    ])
    paths, logger = setup_directories_and_logging(cli_args)
    device = setup_device_and_seed(cli_args, logger)
    shared = load_shared(cli_args, logger, device)

    # 2. Train one config -- cheap.
    train_one_config(cli_args, shared, logger, paths)

    # 3. Train another config, same resident data (new experiment/log dirs).
    cli_args.hidden_size = 256
    cli_args.experiment = "api_demo_h256"
    paths, logger = setup_directories_and_logging(cli_args)
    train_one_config(cli_args, shared, logger, paths)

Next steps
----------

- :doc:`prediction` -- inference on the checkpoint you just produced.
- :doc:`api/modules` -- full API reference for ``train_full_ram``.
- ``analysis/feature_ablation`` -- measure which input channels
  the trained model actually relies on.
