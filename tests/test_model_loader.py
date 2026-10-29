# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
from types import SimpleNamespace

import torch

from phenonn.utils.config import FEATURE_CHANNELS
from phenonn.utils.model_loader import build_model, build_model_pft


def _ns(**overrides):
    """Tiny CLI-args namespace mirroring the .sh entry points."""
    args = dict(
        type="lstm",
        seq_length=365,
        hidden_size=16,
        num_layers=1,
        d_model=None,
        nhead=4,
        forward_expansion=2,
        dropout1=0.0,
        dropout2=0.0,
        dropout_att=0.0,
        dropout_trans=0.0,
        dropout_encoder=0.0,
        feed_forward_trans1=2,
        feed_forward_trans2=2,
        feed_forward_encoder=2,
        n_attn_blocks=1,
        stress_dim=4,
        num_layers1=1,
        d_layers=1,
        embed_size=32,
        daily_lai=False,
        pft_mixing=False,
        pft_meteo_only=False,
        pft_nonneg=False,
        normalize_lai=True,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class TestBuildModel(unittest.TestCase):
    """Unit tests for build_model (single-output factory)."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def _forward(self, typ):
        model = build_model(_ns(type=typ))
        x = torch.zeros(1, FEATURE_CHANNELS, 365)
        return model(x)

    def test_lstm(self):
        """lstm builds and outputs (B, 1, 36)."""
        self.assertEqual(self._forward("lstm").shape, (1, 1, 36))

    def test_gru(self):
        """gru builds and outputs (B, 1, 36)."""
        self.assertEqual(self._forward("gru").shape, (1, 1, 36))

    def test_transformer(self):
        """transformer builds and outputs (B, 1, 36)."""
        self.assertEqual(self._forward("transformer").shape, (1, 1, 36))

    def test_transformer_dec(self):
        """transformer_dec outputs sparse (B, 1, 36) with no wrapper."""
        self.assertEqual(self._forward("transformer_dec").shape, (1, 1, 36))

    def test_aelstm(self):
        """aelstm builds and outputs (B, 1, 36)."""
        self.assertEqual(self._forward("aelstm").shape, (1, 1, 36))

    def test_bitransformer_v2(self):
        """bitransformer_v2 builds and outputs (B, 1, 36)."""
        self.assertEqual(self._forward("bitransformer_v2").shape, (1, 1, 36))

    def test_unknown_type_rejected(self):
        """Unknown --type values raise ValueError."""
        with self.assertRaises(ValueError):
            build_model(_ns(type="does_not_exist"))

    def test_retired_types_rejected(self):
        """Retired --type values (fcn, linear family) raise ValueError."""
        for typ in ("fcn", "fullyconnected", "linear", "linear_perday"):
            with self.assertRaises(ValueError):
                build_model(_ns(type=typ))


class TestBuildModelPft(unittest.TestCase):
    """Unit tests for build_model_pft (PFT-mixing factory)."""

    def test_lstm_pft_shape(self):
        """PFT-mixed lstm outputs (B, 1, 36)."""
        model = build_model_pft(_ns(type="lstm"), norm_stats=None)
        x = torch.zeros(1, FEATURE_CHANNELS, 365)
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 36))
        self.assertTrue(torch.isfinite(y).all())

    def test_meteo_only_pft_shape(self):
        """pft_meteo_only variant keeps the (B, 1, 36) contract."""
        model = build_model_pft(_ns(type="lstm", pft_meteo_only=True), norm_stats=None)
        x = torch.zeros(1, FEATURE_CHANNELS, 365)
        self.assertEqual(model(x).shape, (1, 1, 36))

    def test_daily_pft_shape(self):
        """daily_lai variant outputs (B, 1, 365)."""
        model = build_model_pft(_ns(type="lstm", daily_lai=True), norm_stats=None)
        x = torch.zeros(1, FEATURE_CHANNELS, 365)
        self.assertEqual(model(x).shape, (1, 1, 365))

    def test_retired_types_rejected(self):
        """Retired --type values (fcn, linear family) raise ValueError."""
        for typ in ("fcn", "fullyconnected", "linear", "linear_perday"):
            with self.assertRaises(ValueError):
                build_model(_ns(type=typ))

    def test_v1_types_still_build(self):
        """bitransformer (v1) and attnlstm stay supported."""
        for typ in ("bitransformer", "attnlstm"):
            self.assertIsInstance(build_model(_ns(type=typ)), torch.nn.Module)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestBuildModel))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildModelPft))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
