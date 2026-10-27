"""Tests for the per-PFT wrapper and the hemisphere-aware perturbation."""

import numpy as np
import torch

from phenonn.utils.wrappers import PFTMixingWrapper
from analysis.perturbation._perturb import (
    ChannelPerturbation,
    Perturbation,
    season_mask,
)


class _FakePFT(torch.nn.Module):
    """Return a fixed (B, 15, L) tensor, ignoring the input."""

    def __init__(self, n_pft=15, length=36):
        super().__init__()
        self.n_pft = n_pft
        self.length = length

    def forward(self, x):
        b = x.shape[0]
        base = torch.arange(self.n_pft, dtype=x.dtype, device=x.device)
        return base.view(1, self.n_pft, 1).expand(b, self.n_pft, self.length)


def _norm_stats():
    stats = {"LAI": {"mean": 2.0, "std": 0.5}}
    for k in range(1, 16):
        stats[f"pft{k}_frac"] = {"mean": 1.0 / 15.0, "std": 0.1}
    return stats


def test_forward_with_pure_shapes_and_consistency():
    stats = _norm_stats()
    model = PFTMixingWrapper(
        _FakePFT(),
        norm_stats=stats,
        sparse_output=True,
        lai_normalized=True,
        return_pure=False,
    )
    x = torch.zeros(4, model.pft_start + 15, 36)
    mixed = model(x)
    assert mixed.shape == (4, 1, 36)

    mixed2, pure, frac = model.forward_with_pure(x)
    assert pure.shape == (4, 15, 36)
    assert frac.shape == (4, 15, 1)
    assert torch.allclose(mixed, mixed2, atol=1e-6)
    assert torch.allclose(mixed2, (pure * frac).sum(dim=1, keepdim=True), atol=1e-6)


def test_return_pure_flag():
    model = PFTMixingWrapper(
        _FakePFT(),
        norm_stats=_norm_stats(),
        sparse_output=True,
        return_pure=True,
    )
    out = model(torch.zeros(2, model.pft_start + 15, 36))
    assert isinstance(out, tuple) and len(out) == 3


def test_southern_hemisphere_season_shift():
    north = season_mask(365, [(152, 243)], hemisphere="north")
    south = season_mask(365, [(152, 243)], hemisphere="south")
    assert north.sum() > 0 and south.sum() > 0
    # The two masks must be (approximately) disjoint seasonal windows.
    assert np.mean(north & south) < 0.05


def test_position_mask_and_set_to():
    x = torch.zeros(2, 3, 10)
    cp = ChannelPerturbation("Tmean", set_to=5.0)
    pert = Perturbation(
        [cp],
        norm_stats=None,
        position_mask=np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
    )
    out = pert.apply(x)
    assert torch.allclose(out[:, cp.ch, 3:5], torch.full((2, 2), 5.0))
    assert torch.allclose(out[:, cp.ch, :3], torch.zeros(2, 3))


def test_hemisphere_auto_per_sample():
    x = torch.zeros(2, 3, 365)
    cp = ChannelPerturbation("Tmean", add=1.0)
    pert = Perturbation(
        [cp],
        norm_stats=None,
        season_ranges=[(152, 243)],
        hemisphere="auto",
    )
    out = pert.apply(x, lats=[45.0, -45.0])
    # Northern sample is perturbed in JJA; southern in DJF.
    assert out[0, cp.ch, 180] > 0
    assert out[0, cp.ch, 20] == 0
    assert out[1, cp.ch, 180] == 0
    assert out[1, cp.ch, 20] > 0


def test_hemisphere_south_without_latitudes():
    x = torch.zeros(1, 3, 365)
    cp = ChannelPerturbation("Tmean", add=1.0)
    pert = Perturbation(
        [cp],
        norm_stats=None,
        season_ranges=[(152, 243)],
        hemisphere="south",
    )
    out = pert.apply(x)
    # A fixed southern-hemisphere window shifts JJA to DJF.
    assert out[0, cp.ch, 20] > 0
    assert out[0, cp.ch, 180] == 0
