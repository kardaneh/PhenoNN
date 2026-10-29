# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN training package.

Public API
----------
The training pipeline is exposed as three composable functions plus a
container, so that both the standalone CLI and the in-process sweeps
(``run_ablate_inproc``, ``run_ablate_combo_models``, ``run_year_split``)
can share the same RAM-resident datasets.

- :class:`SharedData`          — container holding the resident datasets and
                                  pools for one working-set.
- :func:`load_shared`          — load the working-set into RAM once.
- :func:`train_one_config`     — train one hyperparameter config on a
                                  ``SharedData``.
- :func:`main`                 — CLI entry point;
                                  ``python -m phenonn.training.train_full_ram``.
- :func:`parse_args`           — CLI parser (accepts a custom ``argv``).
- :func:`parse_year_list`      — parse ``"1992-2010"`` / ``"1992,2000"`` into a
                                  list of years.
- :func:`train_one_epoch`      — inner training loop (used by the composite
                                  train functions).
- :func:`validate`             — inner validation loop.

The legacy CSV trainers (``run_training``, ``run_training_flat``,
``run_training_big``) have been moved to ``legacy/training/``.
"""

import importlib

__all__ = [
    "SharedData",
    "load_shared",
    "train_one_config",
    "main",
    "parse_args",
    "parse_year_list",
    "train_one_epoch",
    "validate",
]

_LAZY_ATTRS = {name: ("phenonn.training.train_full_ram", name) for name in __all__}


def __getattr__(name):
    """PEP 562 lazy access for the training pipeline."""
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
