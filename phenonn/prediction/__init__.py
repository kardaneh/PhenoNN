# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN prediction package.

Public API
----------
- :func:`main`            — CLI entry point; invoked as
                             ``python -m phenonn.prediction.predict`` or via
                             ``phenonn predict``.
- :func:`_parse_year_spec` — parse ``"2010-2019"`` / ``"2010,2015"`` / ``"all"``
                             into a list of years.
- :func:`_resolve_sites`  — resolve the ``--predict_sites`` / ``--sites`` /
                             ``--selected_pixels`` CLI flags against a checkpoint
                             to a concrete list of site IDs. Public-facing
                             despite the leading underscore, because the
                             analysis modules (perturbation, pft_linearity)
                             reuse it.

The legacy flat-CSV predictor (``predict_flat``) has been moved to
``legacy/prediction/``.
"""

import importlib

__all__ = [
    "main",
    "_parse_year_spec",
    "_resolve_sites",
]

_LAZY_ATTRS = {name: ("phenonn.prediction.predict", name) for name in __all__}


def __getattr__(name):
    """PEP 562 lazy access for the prediction pipeline."""
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
