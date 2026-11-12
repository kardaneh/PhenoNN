# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Unit tests for the per-channel normalisation specs."""

import unittest

import numpy as np

from phenonn.data.normalisation import (
    describe_specs,
    finalise_specs,
    forward_numpy,
    get_spec,
    inverse_numpy,
    physical_zero,
    validate_specs,
)
from phenonn.datagen.norm_profiles import resolve_profile

LEGACY = {
    "Tmin": {"mean": 8.0, "std": 14.0},
    "ssrd_sum": {"mean": 2.7, "std": 0.7},
    "tp_sum": {"mean": 0.4, "std": 0.6},
    "daylength": {"mean": 12.0, "std": 2.8},
    "pft1_frac": {"mean": 0.01, "std": 0.04},
    "LAI": {"mean": 2.2, "std": 1.6},
}


class TestSpecs(unittest.TestCase):
    """Spec resolution, legacy fallback and profile expansion."""

    def test_legacy_kinds(self):
        self.assertEqual(get_spec(LEGACY, "Tmin")["kind"], "zscore")
        self.assertEqual(get_spec(LEGACY, "tp_sum")["kind"], "log1p_zscore")
        self.assertEqual(get_spec(LEGACY, "pft1_frac")["kind"], "zscore")

    def test_no_stats_is_identity(self):
        self.assertEqual(get_spec(None, "Tmin")["kind"], "none")
        self.assertEqual(get_spec({}, "Tmin")["kind"], "none")

    def test_override_wins(self):
        stats = dict(LEGACY, _transforms={"pft1_frac": {"kind": "none"}})
        self.assertEqual(get_spec(stats, "pft1_frac")["kind"], "none")

    def test_n1_expands_pft_and_touches_only_18_channels(self):
        overrides = resolve_profile("n1")
        self.assertEqual(len(overrides), 18)
        self.assertEqual(overrides["ssrd_sum"], "zscore")
        self.assertEqual(overrides["pft15_frac"], "none")
        self.assertNotIn("Tmin", overrides)

    def test_n2_adds_vpd_mean(self):
        self.assertEqual(len(resolve_profile("n2")), 19)
        self.assertEqual(resolve_profile("n2")["VPD_mean"], "sqrt_zscore")

    def test_precipitation_profiles_are_single_channel(self):
        for name in ("p_sqrt", "p_cbrt", "p_asinh"):
            self.assertEqual(list(resolve_profile(name)), ["tp_sum"])

    def test_n3_is_all_channel_minmax(self):
        overrides = resolve_profile("n3")
        self.assertEqual(len(overrides), 28)
        self.assertEqual(overrides["Tmin"]["bounds"], (-50.0, 50.0))
        self.assertEqual(overrides["co2"]["bounds"], (300.0, 450.0))
        self.assertEqual(overrides["daylength"]["bounds"], (0.0, 24.0))
        self.assertEqual(
            overrides["pft15_frac"], {"kind": "minmax", "bounds": (0.0, 1.0)}
        )
        self.assertIn("percentiles", overrides["tp_sum"])
        self.assertIn("percentiles", overrides["ssrd_sum"])

    def test_n3_finalises_with_percentiles(self):
        from phenonn.utils.config import DYNAMIC_FEATURES

        pct = {name: {"p1": 0.0, "p99": 1.0} for name in DYNAMIC_FEATURES}
        specs = finalise_specs(resolve_profile("n3"), pct)
        self.assertEqual(len(specs), 28)
        self.assertEqual(specs["tp_sum"]["lo"], 0.0)
        self.assertEqual(specs["tp_sum"]["hi"], 1.0)
        self.assertEqual(specs["Tmin"]["lo"], -50.0)

    def test_asinh_scale_must_be_one(self):
        with self.assertRaises(ValueError):
            finalise_specs(
                {"tp_sum": {"kind": "asinh_zscore", "scale": 2.0}},
                moments={"tp_sum": {"asinh_zscore": {"mean": 0.0, "std": 1.0}}},
            )

    def test_validate_rejects_log1p_on_signed(self):
        with self.assertRaises(ValueError):
            validate_specs({"Tmin": "log1p_zscore"})
        with self.assertRaises(ValueError):
            validate_specs({"Rn_tot": "sqrt_zscore"})

    def test_finalise_bounds_and_percentiles(self):
        specs = finalise_specs(
            {"daylength": {"kind": "minmax", "bounds": (0.0, 24.0)}},
        )
        self.assertEqual(specs["daylength"]["hi"], 24.0)
        specs = finalise_specs(
            {"tp_sum": {"kind": "minmax", "percentiles": (1, 99)}},
            {"tp_sum": {"p1": 0.0, "p99": 26.3}},
        )
        self.assertEqual(specs["tp_sum"]["lo"], 0.0)
        self.assertEqual(specs["tp_sum"]["hi"], 26.3)

    def test_finalise_uses_kind_specific_moments(self):
        """A zscore override must carry raw moments, not the flat log1p ones."""
        specs = finalise_specs(
            {"ssrd_sum": "zscore"},
            moments={"ssrd_sum": {"zscore": {"mean": 15.4, "std": 7.6}}},
        )
        self.assertAlmostEqual(specs["ssrd_sum"]["mean"], 15.4)
        self.assertAlmostEqual(specs["ssrd_sum"]["std"], 7.6)

    def test_finalise_requires_moments_for_zscore(self):
        with self.assertRaises(ValueError):
            finalise_specs({"ssrd_sum": "zscore"})

    def test_describe_specs(self):
        text = describe_specs(LEGACY)
        self.assertIn("zscore", text)
        self.assertIn("log1p_zscore", text)


class TestTransforms(unittest.TestCase):
    """Forward/inverse behaviour of every kind."""

    def test_round_trip(self):
        x = np.array([0.0, 0.5, 2.0, 10.0], dtype=np.float64)
        cases = [
            {"kind": "zscore", "mean": 2.0, "std": 1.5},
            {"kind": "log1p_zscore", "mean": 0.4, "std": 0.6},
            {"kind": "sqrt_zscore", "mean": 0.7, "std": 0.5},
            {"kind": "cbrt_zscore", "mean": 1.2, "std": 0.6},
            {"kind": "asinh_zscore", "mean": 0.9, "std": 0.8, "scale": 1.0},
            {"kind": "minmax", "lo": 0.0, "hi": 10.0, "clip": True},
            {"kind": "none"},
        ]
        for spec in cases:
            z = forward_numpy(x, spec)
            back = inverse_numpy(z, spec)
            np.testing.assert_allclose(back, x, rtol=1e-10, atol=1e-10)

    def test_legacy_equivalence(self):
        """The new path must equal the historical formula."""
        x = np.array([0.0, 1.0, 5.0, 20.0], dtype=np.float32)
        spec = get_spec(LEGACY, "tp_sum")
        expected = (np.log1p(np.clip(x, 0.0, None)) - LEGACY["tp_sum"]["mean"]) / max(
            LEGACY["tp_sum"]["std"], 1e-8
        )
        np.testing.assert_array_equal(forward_numpy(x, spec), expected)

        spec = get_spec(LEGACY, "Tmin")
        expected = (x - LEGACY["Tmin"]["mean"]) / max(LEGACY["Tmin"]["std"], 1e-8)
        np.testing.assert_array_equal(forward_numpy(x, spec), expected)

    def test_minmax_constant_span(self):
        spec = {"kind": "minmax", "lo": 3.0, "hi": 3.0, "clip": True}
        np.testing.assert_array_equal(
            forward_numpy(np.array([1.0, 3.0, 5.0]), spec), np.zeros(3)
        )

    def test_physical_zero(self):
        spec = {"kind": "zscore", "mean": 2.0, "std": 4.0}
        self.assertAlmostEqual(physical_zero(spec), -0.5)
        spec = {"kind": "log1p_zscore", "mean": 0.4, "std": 0.6}
        self.assertAlmostEqual(physical_zero(spec), -0.4 / 0.6)
        spec = {"kind": "minmax", "lo": -2.0, "hi": 2.0, "clip": True}
        self.assertAlmostEqual(physical_zero(spec), 0.5)


class TestWrapperInverse(unittest.TestCase):
    """The PFT mixing wrapper must invert the spec, not always z-score."""

    def _wrapper(self, stats, renormalize=False):
        import torch
        from phenonn.utils.wrappers import PFTMixingWrapper

        class Zero(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 15, 365)

        return PFTMixingWrapper(
            Zero(), norm_stats=stats, renormalize=renormalize, zero_pft1_lai=False
        )

    def test_legacy_zscore_inverse(self):
        import torch

        stats = {f"pft{k}_frac": {"mean": 0.1, "std": 0.2} for k in range(1, 16)}
        wrapper = self._wrapper(stats)
        x = torch.zeros(1, 28, 730)
        x[:, -15:, -1] = 0.0  # z = (0.1 - 0.1) / 0.2
        _, _, pft_real = wrapper.forward_with_pure(x)
        self.assertTrue(torch.allclose(pft_real, torch.full((1, 15, 1), 0.1)))

    def test_none_returns_raw_fractions(self):
        import torch

        stats = {
            "_transforms": {f"pft{k}_frac": {"kind": "none"} for k in range(1, 16)}
        }
        wrapper = self._wrapper(stats)
        x = torch.zeros(1, 28, 730)
        x[:, -15:, -1] = 0.2
        _, _, pft_real = wrapper.forward_with_pure(x)
        self.assertTrue(torch.allclose(pft_real, torch.full((1, 15, 1), 0.2)))


class TestConfigConsistency(unittest.TestCase):
    """The datagen and training feature contracts must not drift."""

    def test_log_transform_sets_match(self):
        from phenonn.datagen import config as datagen
        from phenonn.utils import config as training

        self.assertEqual(
            set(datagen.LOG_TRANSFORM_FEATURES), set(training.LOG_TRANSFORM_FEATURES)
        )
        self.assertEqual(datagen.DYNAMIC_FEATURES, training.DYNAMIC_FEATURES)


if __name__ == "__main__":
    unittest.main()
