# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN data package.

Public surface: the pixelset-based LAI datasets and the site-id / lookup
helpers that the training, prediction and analysis layers consume.

The legacy CSV datasets (``PhenoCamDataset``, ``LAIDataset`` flat-CSV,
``BigLAIDataset``) and the pandas feature-engineering helpers have been
moved to ``legacy/data/`` and are no longer part of this package.

Public API
----------
Datasets
    LAIDataset, RamLAIDataset
Climatology / pools
    compute_climatology_lookup, list_valid_sites
Reference tables
    load_co2_lut, load_parent01_map, load_selected_pixels
Site-id encoding
    encode_site_id, decode_site_id, generate_site_ids_from_range
Daily-LAI constants
    LAI_DAILY_FNAME, N_DAYS_YEAR
"""

import importlib

__all__ = [
    # Datasets
    "RamLAIDataset",
    # Climatology / pools
    "compute_climatology_lookup",
    "list_valid_sites",
    # Reference tables
    "load_co2_lut",
    "load_parent01_map",
    "load_selected_pixels",
    # Site-id encoding
    "encode_site_id",
    "decode_site_id",
    "generate_site_ids_from_range",
    # Daily-LAI constants
    "LAI_DAILY_FNAME",
    "N_DAYS_YEAR",
]

_LAZY_ATTRS = {name: ("phenonn.data.lai_dataset", name) for name in __all__}


def __getattr__(name):
    """PEP 562 lazy access for the dataset helpers."""
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
