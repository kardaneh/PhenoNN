"""Evaluation, leakage checks and reproducibility helpers for PhenoNN.

Submodules are loaded lazily so importing ``phenonn.evaluation.leakage`` does
not pull in the pandas-based metrics/bundle modules.
"""

from __future__ import annotations

import importlib

__all__ = ["bundle", "leakage", "metrics", "reproducibility"]

_LAZY_ATTRS = {name: (f"phenonn.evaluation.{name}", None) for name in __all__}


def __getattr__(name):
    """PEP 562 lazy access for the evaluation submodules."""
    try:
        module_name, _ = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = importlib.import_module(module_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
