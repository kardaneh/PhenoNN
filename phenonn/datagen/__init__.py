# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
phenonn.datagen
===============

Data-generation pipeline for the ``pixelset1`` dataset: site selection,
rebalancing, ERA5 daily pixelsets, LAI/PFT pixelsets, daily LAI interpolation,
optional derived features, and normalisation statistics.

Submodules are CLI scripts and are meant to be run as modules, e.g.::

    python -m phenonn.datagen.select_pixels --help
    python -m phenonn.datagen.build_daily_dataset_pixelset --help
    python -m phenonn.datagen.create_stats_file --help

They are NOT imported here on purpose: importing them would drag in
``xarray`` / ``matplotlib`` / ``dask`` at package-import time. Import the
shared constants from :mod:`phenonn.datagen.config` instead.

Pipeline order
--------------
The report route uses 0.05 degree LAI/PFT targets and 0.1 degree ERA5
features, joined by a parent map. The 0.05 degree ERA5 identity route remains
available as a legacy alternative.

1. ``compute_valid_pixels`` -> ``valid_pixels_{YYYY}.nc`` (0.05 degree masks)
2. ``select_pixels`` -> ``selected_pixels005_1.nc`` (and optionally
   ``all_pixels_selected.nc``)
3. ``rebalance_pixels`` (PFT route) / ``rebalance_spatial`` (spatial route) ->
   ``selected_pixels005_balanced.nc``
4. ``make_selected_pixels_01`` -> ``selected_pixels01_1.nc`` (deduplicated
   0.1 degree ERA5 cells plus the 0.05 -> 0.1 parent map)
5. ``build_daily_dataset_era5idx`` (0.1 degree report route) /
   ``build_daily_dataset_pixelset`` (0.05 degree legacy route) ->
   ``era5_pixelset/ERA5_daily_pixelset_{Y}.nc``
6. ``build_pixelset_targets`` -> ``LAI_pixelset/``, ``PFT_pixelset/``
7. ``interpolate_lai_daily_linear`` -> ``LAI_daily_linear/``
8. ``add_daylength`` and ``add_soil_moisture_proxy`` (required for the report
   contract); ``add_pheno_thermal`` is optional
9. ``create_stats_file`` -> ``norm_stats_{Y0}_{Y1}.json``

Diagnostics (not part of the data build):
``diagnose_selected_pixels``, ``diagnose_selected_pixels_bis``,
``pft_diagnostics``.
"""

from phenonn.datagen.config import (
    N_PFT,
    N_DEKAD_YEAR,
    METEO_BASE,
    DYNAMIC_FEATURES,
    PFT_COLS,
    PFT_NAMES,
    FEATURES_FNAME,
    TARGETS_FNAME,
    PFT_FNAME,
    VALID_FNAME,
    SELECTED_PIXELS_FNAME,
)

__all__ = [
    "N_PFT",
    "N_DEKAD_YEAR",
    "METEO_BASE",
    "DYNAMIC_FEATURES",
    "PFT_COLS",
    "PFT_NAMES",
    "FEATURES_FNAME",
    "TARGETS_FNAME",
    "PFT_FNAME",
    "VALID_FNAME",
    "SELECTED_PIXELS_FNAME",
]
