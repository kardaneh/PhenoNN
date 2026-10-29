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

from phenonn.models.bitransformer import BiTransformerV2
from tests.conftest import set_seed


def _tiny_model(**overrides):
    kwargs = dict(
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
    kwargs.update(overrides)
    return BiTransformerV2(**kwargs)


class TestBiTransformerV2(unittest.TestCase):
    """Unit tests for BiTransformerV2 (kept model)."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        set_seed()
        self.batch = 2
        self.length = 36
        self.input_dim = 12

        if self.logger:
            self.logger.info("Test setup - BiTransformerV2 tests")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_init(self):
        """Test BiTransformerV2 initialization."""
        model = _tiny_model()
        self.assertIsInstance(model, BiTransformerV2)
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.output_dim, 1)
        self.assertEqual(model.n_pft, 5)

    def test_init_invalid_nhead(self):
        """d_model not divisible by nhead raises ValueError."""
        with self.assertRaises(ValueError):
            _tiny_model(d_model=30, nhead=4)

    def test_init_invalid_pft(self):
        """input_dim <= n_pft raises ValueError."""
        with self.assertRaises(ValueError):
            _tiny_model(input_dim=5, n_pft=5)

    def test_init_odd_d_model(self):
        """Odd d_model raises ValueError (sinusoidal PE)."""
        with self.assertRaises(ValueError):
            _tiny_model(d_model=31, d_model2=32, nhead=1)

    # ------------------------------------------------------------------------
    # Forward Pass Tests
    # ------------------------------------------------------------------------

    def test_forward_tensor_shape(self):
        """Default forward returns a single tensor (B, L, out)."""
        model = _tiny_model()
        x = torch.randn(self.batch, self.length, self.input_dim)
        out = model(x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (self.batch, self.length, 1))

    def test_forward_return_stress_tuple(self):
        """return_stress=True returns (out, stress) with locked shapes."""
        model = _tiny_model()
        x = torch.randn(self.batch, self.length, self.input_dim)
        out, stress = model(x, return_stress=True)
        self.assertEqual(out.shape, (self.batch, self.length, 1))
        self.assertEqual(stress.shape, (self.batch, self.length, 4))

    def test_forward_wrong_channels(self):
        """Wrong trailing channel count raises ValueError."""
        model = _tiny_model()
        x = torch.randn(self.batch, self.length, self.input_dim + 1)
        with self.assertRaises(ValueError):
            model(x)

    def test_forward_shorter_sequence(self):
        """Shorter-than-buffer sequences are cropped, not rejected."""
        model = _tiny_model(causal=True)
        x = torch.randn(self.batch, 20, self.input_dim)
        out = model(x)
        self.assertEqual(out.shape, (self.batch, 20, 1))

    def test_causal_mask_buffer(self):
        """causal=True stores a mask buffer; False stores None."""
        self.assertIsNotNone(_tiny_model(causal=True).causal_mask)
        self.assertIsNone(_tiny_model(causal=False).causal_mask)

    def test_eval_deterministic(self):
        """Eval-mode forward is deterministic."""
        model = _tiny_model()
        model.eval()
        x = torch.randn(self.batch, self.length, self.input_dim)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x))

    def test_backward(self):
        """Gradients flow back to stage 1."""
        model = _tiny_model()
        x = torch.randn(self.batch, self.length, self.input_dim)
        model(x).sum().backward()
        for p in model.stage1.parameters():
            self.assertIsNotNone(p.grad)

    def test_save_load_consistency(self):
        """Save/load round-trip preserves eval output."""
        model1 = _tiny_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = _tiny_model()
        model2.load_state_dict(torch.load(temp_file))

        x = torch.randn(self.batch, self.length, self.input_dim)
        model1.eval()
        model2.eval()
        with torch.no_grad():
            torch.testing.assert_close(model1(x), model2(x))

        os.unlink(temp_file)

    def test_device_transfer(self):
        """Test CUDA transfer."""
        if torch.cuda.is_available():
            model = _tiny_model().cuda()
            x = torch.randn(self.batch, self.length, self.input_dim).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)


class TestBiTransformerV2Integration(unittest.TestCase):
    """Integration tests."""

    def test_pft_output_channels(self):
        """output_dim=15 (PFT-decomposed setup) keeps (B, L, 15)."""
        set_seed()
        model = _tiny_model(output_dim=15)
        x = torch.randn(2, 36, 12)
        self.assertEqual(model(x).shape, (2, 36, 15))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestBiTransformerV2))
    suite.addTests(loader.loadTestsFromTestCase(TestBiTransformerV2Integration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
