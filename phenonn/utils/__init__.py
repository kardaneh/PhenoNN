# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN utility package.

Public API
----------
Configuration
    ALL_FEATURES, DYNAMIC_FEATURES, CYCLIC_FEATURES, CO2_FEATURES,
    METEO_BASE, PFT_COLS, PFT_NAMES, PFT_START, FEATURE_CHANNELS,
    N_PFT, N_DEKAD_YEAR, DEKAD_DAYS, SEQ_LENGTH_DEFAULT,
    LOG_TRANSFORM_FEATURES, FEATURES_FNAME, TARGETS_FNAME, PFT_FNAME,
    VALID_FNAME, SELECTED_PIXELS_FNAME
Wrappers
    DailyWrapper, Every10DaysWrapper, PFTMixingWrapper, permuteWrapper
Model loader
    build_model, build_model_pft
Loss
    nan_safe_loss, nan_safe_mse, make_loss_fn
Diagnostics
    plot_loss_histories, plot_metric_histories, plot_pred_vs_obs,
    plot_gcc_curves, plot_gcc_curves_all, make_history_dicts
Logger
    Logger
General helpers
    FileUtils, EasyDict

Heavy names are imported lazily, so ``from phenonn.utils.logger import Logger``
does not import torch or matplotlib.
"""

import importlib

from .config import (
    ALL_FEATURES,
    DYNAMIC_FEATURES,
    CYCLIC_FEATURES,
    CO2_FEATURES,
    METEO_BASE,
    PFT_COLS,
    PFT_NAMES,
    PFT_START,
    FEATURE_CHANNELS,
    N_PFT,
    N_DEKAD_YEAR,
    DEKAD_DAYS,
    SEQ_LENGTH_DEFAULT,
    LOG_TRANSFORM_FEATURES,
    FEATURES_FNAME,
    TARGETS_FNAME,
    PFT_FNAME,
    VALID_FNAME,
    SELECTED_PIXELS_FNAME,
)

__all__ = [
    "ALL_FEATURES",
    "DYNAMIC_FEATURES",
    "CYCLIC_FEATURES",
    "CO2_FEATURES",
    "METEO_BASE",
    "PFT_COLS",
    "PFT_NAMES",
    "PFT_START",
    "FEATURE_CHANNELS",
    "N_PFT",
    "N_DEKAD_YEAR",
    "DEKAD_DAYS",
    "SEQ_LENGTH_DEFAULT",
    "LOG_TRANSFORM_FEATURES",
    "FEATURES_FNAME",
    "TARGETS_FNAME",
    "PFT_FNAME",
    "VALID_FNAME",
    "SELECTED_PIXELS_FNAME",
    "DailyWrapper",
    "Every10DaysWrapper",
    "PFTMixingWrapper",
    "permuteWrapper",
    "build_model",
    "build_model_pft",
    "nan_safe_loss",
    "nan_safe_mse",
    "make_loss_fn",
    "plot_loss_histories",
    "plot_metric_histories",
    "plot_pred_vs_obs",
    "plot_gcc_curves",
    "plot_gcc_curves_all",
    "make_history_dicts",
    "Logger",
    "FileUtils",
    "EasyDict",
]

_LAZY_ATTRS = {
    "DailyWrapper": ("phenonn.utils.wrappers", "DailyWrapper"),
    "Every10DaysWrapper": ("phenonn.utils.wrappers", "Every10DaysWrapper"),
    "PFTMixingWrapper": ("phenonn.utils.wrappers", "PFTMixingWrapper"),
    "permuteWrapper": ("phenonn.utils.wrappers", "permuteWrapper"),
    "build_model": ("phenonn.utils.model_loader", "build_model"),
    "build_model_pft": ("phenonn.utils.model_loader", "build_model_pft"),
    "nan_safe_loss": ("phenonn.utils.evaluater", "nan_safe_loss"),
    "nan_safe_mse": ("phenonn.utils.evaluater", "nan_safe_mse"),
    "make_loss_fn": ("phenonn.utils.evaluater", "make_loss_fn"),
    "plot_loss_histories": (
        "phenonn.utils.diagnostics",
        "plot_loss_histories",
    ),
    "plot_metric_histories": (
        "phenonn.utils.diagnostics",
        "plot_metric_histories",
    ),
    "plot_pred_vs_obs": ("phenonn.utils.diagnostics", "plot_pred_vs_obs"),
    "plot_gcc_curves": ("phenonn.utils.diagnostics", "plot_gcc_curves"),
    "plot_gcc_curves_all": (
        "phenonn.utils.diagnostics",
        "plot_gcc_curves_all",
    ),
    "make_history_dicts": (
        "phenonn.utils.diagnostics",
        "make_history_dicts",
    ),
    "Logger": ("phenonn.utils.logger", "Logger"),
    "FileUtils": ("phenonn.utils.utils", "FileUtils"),
    "EasyDict": ("phenonn.utils.utils", "EasyDict"),
}


def __getattr__(name):
    """PEP 562 lazy access for the heavy utility names."""
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
