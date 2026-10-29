# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN - Deep learning for phenology prediction using climate data.

This package provides LSTM, GRU, and Transformer models to predict
Green Chromatic Coordinate (GCC) or Leaf Area Index (LAI) from climate data.

The heavy public names are loaded lazily.  Importing :mod:`phenonn` is cheap
and does not import torch, the training stack or the prediction stack; those
are imported on first access (for example ``phenonn.train`` or
``phenonn.RNN_LSTM``).  This keeps the CLI ``--help`` and ``--version`` fast on
login nodes with slow shared filesystems.
"""

import importlib
import logging
import os

# Version (will be updated by setuptools)
try:
    from .version import __version__, get_version, get_versions
except ImportError:
    __version__ = "0.1.0.dev0"

__author__ = "Kazem Ardaneh, Stefan Barbu, Esteban Carlin"
__email__ = "kardaneh@ipsl.fr"
__license__ = "CC BY-NC-SA 4.0"

__description__ = "LSTM-based phenology prediction from climate data"
__url__ = "https://github.com/kardaneh/PhenoNN"

# Public names that need a module import on first access.
_LAZY_ATTRS = {
    "RamLAIDataset": ("phenonn.data", "RamLAIDataset"),
    "RNN_LSTM": ("phenonn.models.rnn", "RNN_LSTM"),
    "RNN_GRU": ("phenonn.models.rnn", "RNN_GRU"),
    "AELSTM": ("phenonn.models.aelstm", "AELSTM"),
    "BiTransformerV2": ("phenonn.models.bitransformer", "BiTransformerV2"),
    "EncoderTorch": ("phenonn.models.transformer", "EncoderTorch"),
    "CausalSparseTransformer": (
        "phenonn.models.transformer",
        "CausalSparseTransformer",
    ),
    "SinusoidalPositionalEncoding": (
        "phenonn.models.transformer",
        "SinusoidalPositionalEncoding",
    ),
    "train": ("phenonn.training.train_full_ram", "main"),
    "load_shared": ("phenonn.training.train_full_ram", "load_shared"),
    "train_one_config": (
        "phenonn.training.train_full_ram",
        "train_one_config",
    ),
    "predict": ("phenonn.prediction.predict", "main"),
}

__all__ = [
    # Data
    "RamLAIDataset",
    # Models
    "RNN_LSTM",
    "RNN_GRU",
    "AELSTM",
    "BiTransformerV2",
    "EncoderTorch",
    "CausalSparseTransformer",
    "SinusoidalPositionalEncoding",
    # Training
    "train",
    "load_shared",
    "train_one_config",
    # Prediction
    "predict",
    # Version / environment
    "__version__",
    "get_version",
    "get_versions",
    "DEVICE",
]

_DEVICE_PRINTED = False


def _resolve_device():
    """Import torch lazily and resolve the default device."""
    global _DEVICE_PRINTED

    import torch

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _DEVICE_PRINTED:
        _DEVICE_PRINTED = True
        print(f"PhenoNN using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    return device


def __getattr__(name):
    """PEP 562 lazy access for the heavy public names."""
    if name == "DEVICE":
        device = _resolve_device()
        globals()["DEVICE"] = device
        return device

    try:
        module_name, attribute = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    """Return the stable public names plus the module globals."""
    return sorted(set(globals()) | set(__all__))


# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)
