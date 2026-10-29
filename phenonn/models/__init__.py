# Copyright 2026 CNRS
# Authors: Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures for RTnn."""

import importlib

__all__ = [
    "RNN_LSTM",
    "RNN_GRU",
    "AttnLSTM",
    "AELSTM",
    "EncoderTorch",
    "CombinedModel",
    "BiTransformer",
]

_LAZY_ATTRS = {
    "RNN_LSTM": ("phenonn.models.rnn", "RNN_LSTM"),
    "RNN_GRU": ("phenonn.models.rnn", "RNN_GRU"),
    "AttnLSTM": ("phenonn.models.attn_lstm", "AttnLSTM"),
    "AELSTM": ("phenonn.models.aelstm", "AELSTM"),
    "EncoderTorch": ("phenonn.models.transformer", "EncoderTorch"),
    "CombinedModel": ("phenonn.models.transformerbis", "CombinedModel"),
    "BiTransformer": ("phenonn.models.transformerbis", "BiTransformer"),
}


def __getattr__(name):
    """PEP 562 lazy access for the model architectures."""
    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
