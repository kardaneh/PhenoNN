# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import tempfile
import unittest

import torch

from phenonn.models.transformer import (
    CausalSparseTransformer,
    SinusoidalPositionalEncoding,
)
from tests.conftest import set_seed


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    """Unit tests for SinusoidalPositionalEncoding."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        set_seed()

    def test_forward_adds_encoding(self):
        """Encoding is added elementwise, shape preserved."""
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
        x = torch.zeros(2, 10, 32)
        y = pe(x)
        self.assertEqual(y.shape, (2, 10, 32))
        # Position 0 of a zero input exposes the raw encoding (sin/cos of 0).
        self.assertAlmostEqual(y[0, 0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(y[0, 0, 1].item(), 1.0, places=5)

    def test_odd_d_model_raises(self):
        """Odd d_model raises ValueError."""
        with self.assertRaises(ValueError):
            SinusoidalPositionalEncoding(d_model=31)

    def test_positions_differ(self):
        """Different positions get different encodings."""
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
        x = torch.zeros(1, 5, 32)
        y = pe(x)
        self.assertFalse(torch.equal(y[0, 0], y[0, 1]))


class TestCausalSparseTransformer(unittest.TestCase):
    """Unit tests for CausalSparseTransformer (kept model)."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures (tiny but L >= 365 for the obs window)."""
        set_seed()
        self.batch_size = 2
        self.feature_channel = 6
        self.output_channel = 1
        self.seq_length = 370
        self.kwargs = dict(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=64,
            seq_length=self.seq_length,
            dropout=0.0,
        )

        if self.logger:
            self.logger.info("Test setup - CausalSparseTransformer tests")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_init(self):
        """Test CausalSparseTransformer initialization."""
        model = CausalSparseTransformer(**self.kwargs)
        self.assertIsInstance(model, CausalSparseTransformer)
        self.assertEqual(model.output_channel, self.output_channel)
        self.assertEqual(model.num_obs_days, 36)

    def test_init_invalid_heads(self):
        """d_model not divisible by n_heads raises ValueError."""
        bad = dict(self.kwargs, d_model=30, n_heads=4)
        with self.assertRaises(ValueError):
            CausalSparseTransformer(**bad)

    def test_init_odd_d_model(self):
        """Odd d_model raises ValueError."""
        bad = dict(self.kwargs, d_model=31, n_heads=1)
        with self.assertRaises(ValueError):
            CausalSparseTransformer(**bad)

    # ------------------------------------------------------------------------
    # Forward Pass Tests
    # ------------------------------------------------------------------------

    def test_forward_sparse_shape(self):
        """Forward maps (B, C, L) -> (B, C_out, 36)."""
        model = CausalSparseTransformer(**self.kwargs)
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (self.batch_size, self.output_channel, 36))
        self.assertTrue(torch.isfinite(y).all())

    def test_eval_deterministic(self):
        """Eval-mode forward is deterministic."""
        model = CausalSparseTransformer(**self.kwargs)
        model.eval()
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x))

    def test_backward(self):
        """Gradients flow back to the input projection."""
        model = CausalSparseTransformer(**self.kwargs)
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        model(x).sum().backward()
        for p in model.input_conv.parameters():
            self.assertIsNotNone(p.grad)

    def test_save_load_consistency(self):
        """Save/load round-trip preserves eval output."""
        model1 = CausalSparseTransformer(**self.kwargs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = CausalSparseTransformer(**self.kwargs)
        model2.load_state_dict(torch.load(temp_file))

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        model1.eval()
        model2.eval()
        with torch.no_grad():
            torch.testing.assert_close(model1(x), model2(x))

        os.unlink(temp_file)

    def test_device_transfer(self):
        """Test CUDA transfer."""
        if torch.cuda.is_available():
            model = CausalSparseTransformer(**self.kwargs).cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSinusoidalPositionalEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalSparseTransformer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
