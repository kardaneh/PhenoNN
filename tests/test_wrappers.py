# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest

import torch

from phenonn.models.bitransformer import BiTransformerV2
from phenonn.models.rnn import RNN_LSTM
from phenonn.utils.config import FEATURE_CHANNELS, N_PFT
from phenonn.utils.wrappers import (
    DailyWrapper,
    Every10DaysWrapper,
    PFTMixingWrapper,
    permuteWrapper,
)
from tests.conftest import set_seed


class TestEvery10DaysWrapper(unittest.TestCase):
    """Unit tests for Every10DaysWrapper."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        set_seed()
        self.base = RNN_LSTM(
            feature_channel=6, output_channel=1, hidden_size=16, num_layers=1
        )

    def test_forward_shape(self):
        """(B, C, L>=365) -> (B, C_out, 36)."""
        model = Every10DaysWrapper(self.base)
        x = torch.randn(2, 6, 370)
        y = model(x)
        self.assertEqual(y.shape, (2, 1, 36))

    def test_matches_manual_gather(self):
        """Wrapper output equals the base output gathered at obs days."""
        model = Every10DaysWrapper(self.base)
        model.eval()
        x = torch.randn(2, 6, 370)
        with torch.no_grad():
            expected = self.base(x)[:, :, -365:][:, :, model.obs_positions]
            torch.testing.assert_close(model(x), expected)

    def test_backward(self):
        """Gradients flow through to the base model."""
        model = Every10DaysWrapper(self.base)
        x = torch.randn(2, 6, 370)
        model(x).sum().backward()
        for p in self.base.parameters():
            self.assertIsNotNone(p.grad)


class TestDailyWrapper(unittest.TestCase):
    """Unit tests for DailyWrapper."""

    def test_forward_shape(self):
        """(B, C, L>=365) -> (B, C_out, 365)."""
        set_seed()
        base = RNN_LSTM(
            feature_channel=6, output_channel=1, hidden_size=16, num_layers=1
        )
        model = DailyWrapper(base)
        x = torch.randn(2, 6, 370)
        y = model(x)
        self.assertEqual(y.shape, (2, 1, 365))


class TestPermuteWrapper(unittest.TestCase):
    """Unit tests for permuteWrapper around a (B, L, C) model."""

    def test_forward_shape(self):
        """(B, C, L) -> permute -> BiTransformerV2 -> (B, out, L)."""
        set_seed()
        inner = BiTransformerV2(
            input_dim=12,
            output_dim=1,
            d_model=32,
            n_pft=5,
            stress_dim=4,
            nr_blocks_stage1=1,
            nr_blocks_stage2=1,
            nhead=4,
            seq_length=36,
        )
        model = permuteWrapper(inner)
        x = torch.randn(2, 12, 36)
        y = model(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (2, 1, 36))

    def test_backward(self):
        """Gradients flow through the double permute."""
        set_seed()
        inner = BiTransformerV2(
            input_dim=12,
            output_dim=1,
            d_model=32,
            n_pft=5,
            stress_dim=4,
            nr_blocks_stage1=1,
            nr_blocks_stage2=1,
            nhead=4,
            seq_length=36,
        )
        model = permuteWrapper(inner)
        x = torch.randn(2, 12, 36)
        model(x).sum().backward()
        for p in inner.parameters():
            self.assertIsNotNone(p.grad)


class TestPFTMixingWrapper(unittest.TestCase):
    """Unit tests for PFTMixingWrapper."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        set_seed()
        self.length = 370
        self.base = RNN_LSTM(
            feature_channel=FEATURE_CHANNELS,
            output_channel=N_PFT,
            hidden_size=16,
            num_layers=1,
        )
        self.x = torch.randn(2, FEATURE_CHANNELS, self.length)

    def test_forward_mixed_shape(self):
        """15-channel pure LAI mixes down to (B, 1, 36)."""
        model = PFTMixingWrapper(self.base, norm_stats=None)
        y = model(self.x)
        self.assertEqual(y.shape, (2, 1, 36))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_daily_shape(self):
        """daily=True returns (B, 1, 365)."""
        model = PFTMixingWrapper(self.base, norm_stats=None, daily=True)
        y = model(self.x)
        self.assertEqual(y.shape, (2, 1, 365))

    def test_forward_no_renormalize(self):
        """renormalize=False keeps the contract shape."""
        model = PFTMixingWrapper(self.base, norm_stats=None, renormalize=False)
        y = model(self.x)
        self.assertEqual(y.shape, (2, 1, 36))
        self.assertTrue(torch.isfinite(y).all())

    def test_backward(self):
        """Gradients flow back to the base model."""
        model = PFTMixingWrapper(self.base, norm_stats=None)
        model(self.x).sum().backward()
        for p in self.base.parameters():
            self.assertIsNotNone(p.grad)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEvery10DaysWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestDailyWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestPermuteWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestPFTMixingWrapper))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
