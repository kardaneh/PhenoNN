# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Shared fixtures for the PhenoNN unit-test suite.

Kept-surface only: `legacy/` modules are intentionally not imported here.
Heavy data files under `/data` and `runs/` are never touched; dataset
tests build tiny synthetic NetCDF files in temporary directories.
"""

TORCH_SEED = 0


def set_seed(seed: int = TORCH_SEED) -> None:
    """Deterministic CPU (and CUDA, when present) RNG for tiny test models."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
