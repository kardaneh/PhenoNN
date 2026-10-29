Prediction
==========

Entry point
-----------

Inference uses a saved checkpoint. The predictor loads the model, the
wrapper stack, the feature normalisation statistics, the CO2 lookup and
the site pools from the checkpoint, so it does not need to be told the
training configuration again.

Command-line invocation::

    python -m phenonn.prediction.predict [options]

or, once the package is installed::

    phenonn predict [options]

Both forms are equivalent.

Site selection
--------------

The site pool is chosen by the first of these flags that is set:

``--selected_pixels PATH``
    Sites listed in a ``selected_pixels*.nc`` file. Overrides ``--sites``
    and ``--predict_sites``.

``--sites A,B,C``
    Explicit comma-separated site IDs. Overrides ``--predict_sites``.

``--predict_sites val``
    The checkpoint's ``val_site_ids``. This is the default.

``--predict_sites train``
    The checkpoint's ``train_site_ids``.

``--predict_sites all``
    Union of ``train_site_ids`` and ``val_site_ids``.

``--predict_sites grid``
    Every pixel ID in the bbox ``--row_min/--row_max/--col_min/--col_max``.
    When a bound is ``-1``, the value stored in the checkpoint's training
    arguments is used. Sites not present in the pixelset are dropped
    silently by the dataset.

``--predict_sites test``
    ``grid`` minus ``val_site_ids`` — held out for the overlap mode.

To cap the number of sites, use ``--n_predict_sites N`` — the predictor
draws a random subsample of size ``N`` from the pool selected above.

Year selection
--------------

``--predict_years 2015-2018``
    Range or comma-separated list of years to predict.

``--predict_years all``
    Every year present in the dataset.

If not given, the checkpoint's ``val_years`` are used.

Data arguments
--------------

``--features_dir``, ``--target_dir``, ``--pft_dir``
    Override the paths stored in the checkpoint. Usually not needed: the
    checkpoint is self-describing.

``--parent_map``
    Override the 0.05°→0.1° parent map path from the checkpoint.

Leakage checks
--------------

``--protocol {A,B,none}`` (case-insensitive) and ``--test_years`` enable the
same split checks as training. Under Protocol B the predictor refuses an
anomaly climatology that overlaps the test years. A checkpoint written before
the protocol support may store a climatology without ``clim_years``; that
window cannot be verified, so it raises under Protocol A/B unless
``--allow_leakage`` is set. Use ``--allow_leakage`` only for a documented
diagnostic, never for reported test metrics.

Ablation replay
---------------

If the model was trained with
``analysis/feature_ablation``, the ablated channels are read
from the checkpoint and zeroed again at inference, so the model sees
exactly the input it saw during training. Use ``--ablate_features`` to
force a different list when the checkpoint predates that bookkeeping.

Outputs
-------

The predictor writes several files next to ``--output_csv``:

``<output_csv>``
    One row per ``(site, year, dekad)``::

        site_id, year, month, day, doy, lai_pred, lai_obs,
        lai_pred_norm, lai_obs_norm, error

    ``lai_pred`` and ``lai_obs`` are in physical LAI units.
    ``lai_pred_norm`` and ``lai_obs_norm`` are the values actually seen
    by the loss (z-scored, or anomaly). ``error = lai_pred − lai_obs``.

``<output_csv stem>_metrics.txt``
    Pooled metrics, per-site R² distribution, and the R² decomposition
    (overall / site-year / interannual).

``<output_csv stem>_pred_vs_obs.png``
    Predicted-vs-observed hexbin with R², RMSE, MAE, bias.

``<output_csv stem>_pred_vs_obs_norm.png``
    Same, on the normalised values.

``<output_csv stem>_pred_vs_obs_anomaly.png``
    Only when the checkpoint was trained in anomaly mode.

``<output_csv stem>_lai_curves.png``
    Three representative sites (low / medium / high R²) with all
    validation years overlaid.

``<output_csv stem>_lai_curves_all.png``
    Grid of per-site curves, sorted by R². Each subplot lists the
    dominant PFTs of the site when the PFT file is available. By
    default 100 sites are shown; pass ``--n_curves 0`` to plot every
    site, or another value to change the cap.

Metrics
-------

The predictor reports four R² definitions, because they answer different
questions:

- **Global R² (NSE)** — one NSE over all ``(site, year, dekad)`` points.
  Includes the mean difference between sites. Generous.
- **Centered R² (NSE on dynamics)** — remove each site's own mean from
  observed and predicted before computing the NSE. Isolates the
  intra-site temporal variation. This is the fair test of dynamics.
- **Pearson² R²** — the square of the linear correlation. Ignores bias
  and scale; matches how some papers report their metrics.
- **Per-site R² (NSE)** — one NSE per site, and the distribution (mean,
  median, 5th and 95th percentile, fraction of sites with R² > 0).

The R² decomposition reports:

- **Overall** — one value, the same as global R².
- **Site-year** — NSE within each site-year (36 dekads), averaged over
  site-years.
- **Interannual** — NSE across site-years at a fixed dekad, averaged over
  dekads. This is the hardest of the three and the one that distinguishes
  a model that captures inter-annual variability from a climatology.

Examples
--------

Predict on validation sites and years::

    python -m phenonn.prediction.predict \
        --checkpoint runs/my_run/checkpoints/run_best_model.pth.tar \
        --predict_sites val \
        --output_csv runs/my_run/predictions.csv

Predict on a specific site pool, specific years::

    python -m phenonn.prediction.predict \
        --checkpoint runs/my_run/checkpoints/run_best_model.pth.tar \
        --selected_pixels /path/to/selected_pixels_PFT9.nc \
        --predict_years 2016-2018 \
        --n_predict_sites 5000 \
        --output_csv runs/my_run/predictions_pft9.csv

Predict with per-year scatter plots and a small grid of curves::

    python -m phenonn.prediction.predict \
        --checkpoint runs/my_run/checkpoints/run_best_model.pth.tar \
        --predict_sites val \
        --scatter_years \
        --n_curves 100 \
        --output_csv runs/my_run/predictions_val.csv

Python API
----------

The predictor is a script, but its pieces are importable. The two
functions used by the analysis modules are the year parser and the
site resolver:

.. code-block:: python

    from phenonn.prediction.predict import (
        _parse_year_spec as parse_year_spec, _resolve_sites,
    )

    years = parse_year_spec("2015-2018")   # [2015, 2016, 2017, 2018]

    # `args` is any object exposing the relevant CLI attributes
    # (predict_sites, selected_pixels, sites, n_predict_sites, ...).
    sites = _resolve_sites(args, ckpt, train_args)

For a full custom inference loop, load the checkpoint and use the
resident dataset directly:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader

    from phenonn.data import RamLAIDataset
    from phenonn.utils.model_loader import build_model, build_model_pft

    ckpt = torch.load("runs/my_run/checkpoints/run_best_model.pth.tar",
                      map_location="cpu", weights_only=False)
    train_args = ckpt["args"]

    model = (build_model_pft if ckpt.get("pft_mixing") else build_model)(
        type("A", (), train_args), ckpt.get("norm_stats")
    )
    state = ckpt.get("state_dict")
    if state is None:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    ds = RamLAIDataset(
        features_dir="/path/to/era5_pixelset",
        target_dir="/path/to/LAI_pixelset",
        pft_dir="/path/to/PFT_pixelset",
        years=[2016],
        site_ids=ckpt["val_site_ids"][:100],
        seq_length=train_args["seq_length"],
        norm_stats=ckpt.get("norm_stats"),
        co2_lut=ckpt.get("co2_lut"),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    with torch.no_grad():
        for features, targets in loader:
            preds = model(features)      # (B, 1, 36)

Next steps
----------

- :doc:`training` — how to produce the checkpoint the predictor reads.
- ``analysis/diagnostics/climatology_skill.py`` — the baseline every model
  must beat.
- :doc:`api/modules` — full API reference.
