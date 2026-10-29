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

from phenonn.models.aelstm import AELSTM, SelfAttentionBlock
from tests.conftest import set_seed


class TestSelfAttentionBlock(unittest.TestCase):
    """Unit tests for SelfAttentionBlock."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        set_seed()
        self.d_model = 32
        self.nhead = 4

    def test_forward_shape(self):
        """Block preserves (B, L, d_model)."""
        block = SelfAttentionBlock(d_model=self.d_model, nhead=self.nhead)
        x = torch.randn(4, 10, self.d_model)
        y = block(x)
        self.assertEqual(y.shape, (4, 10, self.d_model))


class TestAELSTM(unittest.TestCase):
    """Unit tests for AELSTM (kept model)."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        set_seed()
        self.batch_size = 4
        self.feature_channel = 6
        self.output_channel = 2
        self.seq_length = 10
        self.hidden_size = 32
        self.num_layers = 2

        if self.logger:
            self.logger.info("Test setup - AELSTM tests")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_init(self):
        """Test AELSTM initialization."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.assertIsInstance(model, AELSTM)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.num_layers, self.num_layers)
        self.assertEqual(model.output_channel, self.output_channel)
        self.assertIsInstance(model.lstm, torch.nn.LSTM)
        self.assertIsInstance(model.final, torch.nn.Conv1d)

    def test_init_hidden_shape(self):
        """init_hidden returns (h, c) of shape (layers, B, H)."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        h, c = model.init_hidden(self.batch_size, torch.device("cpu"))
        self.assertEqual(h.shape, (self.num_layers, self.batch_size, self.hidden_size))
        self.assertEqual(c.shape, (self.num_layers, self.batch_size, self.hidden_size))

    # ------------------------------------------------------------------------
    # Forward Pass Tests
    # ------------------------------------------------------------------------

    def test_forward_shape(self):
        """Forward maps (B, C, L) -> (B, C_out, L)."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_batch_and_seq_variants(self):
        """Shape contract holds across batch / sequence sizes."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=1,
        )
        for b, sl in [(1, 5), (8, 20)]:
            x = torch.randn(b, self.feature_channel, sl)
            y = model(x)
            self.assertEqual(y.shape, (b, self.output_channel, sl))

    def test_forward_no_attn_blocks(self):
        """n_attn_blocks=0 still yields the right shape."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=1,
            n_attn_blocks=0,
        )
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        self.assertEqual(
            model(x).shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_eval_deterministic(self):
        """Eval-mode forward is deterministic (no future leakage wobble)."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        model.eval()
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        torch.testing.assert_close(y1, y2)

    def test_backward(self):
        """Gradients flow back to the LSTM."""
        model = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        model(x).sum().backward()
        for p in model.lstm.parameters():
            self.assertIsNotNone(p.grad)

    def test_save_load_consistency(self):
        """Save/load round-trip preserves eval output."""
        model1 = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = AELSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
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
            model = AELSTM(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            ).cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)


class TestAELSTMIntegration(unittest.TestCase):
    """Integration tests."""

    def test_pft_output_channels(self):
        """output_channel=15 (PFT-decomposed setup) keeps the time axis."""
        set_seed()
        model = AELSTM(
            feature_channel=8, output_channel=15, hidden_size=32, num_layers=1
        )
        x = torch.randn(2, 8, 12)
        self.assertEqual(model(x).shape, (2, 15, 12))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSelfAttentionBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestAELSTM))
    suite.addTests(loader.loadTestsFromTestCase(TestAELSTMIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
