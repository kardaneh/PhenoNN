# Copyright 2026 CNRS
# Authors: Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
import sys
import os
from phenonn.models.rnn import BaseRNN, RNN_LSTM, RNN_GRU
from phenonn.utils.model_utils import ModelUtils


class TestBaseRNN(unittest.TestCase):
    """Unit tests for BaseRNN class."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16 * 32
        self.feature_channel = 6
        self.output_channel = 4
        self.seq_length = 10
        self.hidden_size = 96
        self.num_layers = 3

        if self.logger:
            self.logger.info("Test setup - BaseRNN tests")

    # ------------------------------------------------------------------------
    # Initialization Tests
    # ------------------------------------------------------------------------

    def test_init_lstm(self):
        """Test BaseRNN initialization with LSTM."""
        if self.logger:
            self.logger.info("Testing BaseRNN initialization with LSTM")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        self.assertIsInstance(model, BaseRNN)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertEqual(model.num_layers, self.num_layers)
        self.assertEqual(model.output_channel, self.output_channel)
        self.assertEqual(model.rnn_type, "lstm")
        self.assertIsInstance(model.rnn, torch.nn.LSTM)
        self.assertIsInstance(model.final, torch.nn.Conv1d)

        if self.logger:
            self.logger.success("✅ BaseRNN LSTM initialization test passed")

    def test_init_gru(self):
        """Test BaseRNN initialization with GRU."""
        if self.logger:
            self.logger.info("Testing BaseRNN initialization with GRU")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="gru",
        )

        self.assertIsInstance(model, BaseRNN)
        self.assertEqual(model.rnn_type, "gru")
        self.assertIsInstance(model.rnn, torch.nn.GRU)

        if self.logger:
            self.logger.success("✅ BaseRNN GRU initialization test passed")

    def test_init_invalid_type(self):
        """Test BaseRNN initialization with invalid rnn_type."""
        if self.logger:
            self.logger.info("Testing BaseRNN initialization with invalid type")

        with self.assertRaises(ValueError):
            BaseRNN(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                rnn_type="invalid",
            )

        if self.logger:
            self.logger.success("✅ BaseRNN invalid type test passed")

    # ------------------------------------------------------------------------
    # Forward Pass Tests
    # ------------------------------------------------------------------------

    def test_forward_lstm_shape(self):
        """Test LSTM forward pass output shape."""
        if self.logger:
            self.logger.info("Testing LSTM forward pass shape")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

        if self.logger:
            self.logger.success(f"✅ LSTM forward shape test passed: {y.shape}")

    def test_forward_gru_shape(self):
        """Test GRU forward pass output shape."""
        if self.logger:
            self.logger.info("Testing GRU forward pass shape")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="gru",
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

        if self.logger:
            self.logger.success(f"✅ GRU forward shape test passed: {y.shape}")

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        if self.logger:
            self.logger.info("Testing forward with different batch sizes")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        batch_sizes = [1, 4, 16, 32, 64]

        for bs in batch_sizes:
            x = torch.randn(bs, self.feature_channel, self.seq_length)
            y = model(x)
            self.assertEqual(y.shape, (bs, self.output_channel, self.seq_length))

        if self.logger:
            self.logger.success("✅ different batch sizes test passed")

    def test_forward_with_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        if self.logger:
            self.logger.info("Testing forward with different sequence lengths")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        seq_lengths = [1, 5, 10, 20, 50]

        for sl in seq_lengths:
            x = torch.randn(self.batch_size, self.feature_channel, sl)
            y = model(x)
            self.assertEqual(y.shape, (self.batch_size, self.output_channel, sl))

        if self.logger:
            self.logger.success("✅ different sequence lengths test passed")

    # ------------------------------------------------------------------------
    # Hidden State Tests
    # ------------------------------------------------------------------------

    def test_init_hidden_lstm(self):
        """Test hidden state initialization for LSTM."""
        if self.logger:
            self.logger.info("Testing hidden state initialization for LSTM")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        batch_size = 16
        device = torch.device("cpu")
        hidden = model.init_hidden(batch_size, device)

        # LSTM returns tuple (hidden, cell)
        self.assertIsInstance(hidden, tuple)
        self.assertEqual(len(hidden), 2)

        expected_shape = (model.num_layers, batch_size, model.hidden_size)
        self.assertEqual(hidden[0].shape, expected_shape)
        self.assertEqual(hidden[1].shape, expected_shape)

        if self.logger:
            self.logger.success("✅ LSTM hidden state test passed")

    def test_init_hidden_gru(self):
        """Test hidden state initialization for GRU."""
        if self.logger:
            self.logger.info("Testing hidden state initialization for GRU")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="gru",
        )

        batch_size = 16
        device = torch.device("cpu")
        hidden = model.init_hidden(batch_size, device)

        # GRU returns single tensor
        self.assertIsInstance(hidden, torch.Tensor)

        expected_shape = (model.num_layers, batch_size, model.hidden_size)
        self.assertEqual(hidden.shape, expected_shape)

        if self.logger:
            self.logger.success("✅ GRU hidden state test passed")

    # ------------------------------------------------------------------------
    # Gradient Tests
    # ------------------------------------------------------------------------

    def test_gradient_flow_lstm(self):
        """Test gradient flow through LSTM model."""
        if self.logger:
            self.logger.info("Testing gradient flow for LSTM")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        x = torch.randn(
            self.batch_size, self.feature_channel, self.seq_length, requires_grad=True
        )
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

        if self.logger:
            self.logger.success("✅ LSTM gradient flow test passed")

    def test_gradient_flow_gru(self):
        """Test gradient flow through GRU model."""
        if self.logger:
            self.logger.info("Testing gradient flow for GRU")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="gru",
        )

        x = torch.randn(
            self.batch_size, self.feature_channel, self.seq_length, requires_grad=True
        )
        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

        if self.logger:
            self.logger.success("✅ GRU gradient flow test passed")

    def test_gradient_accumulation(self):
        """Test that gradients accumulate properly."""
        if self.logger:
            self.logger.info("Testing gradient accumulation")

        model = BaseRNN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            rnn_type="lstm",
        )

        # Forward pass 1
        x1 = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y1 = model(x1)
        loss1 = y1.sum()
        loss1.backward()

        # Save first gradients
        grad1 = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        # Forward pass 2
        x2 = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y2 = model(x2)
        loss2 = y2.sum()
        loss2.backward()

        # Gradients should have accumulated
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertIsNotNone(grad1.get(name))
                self.assertNotEqual(param.grad.sum(), grad1[name].sum())

        if self.logger:
            self.logger.success("✅ gradient accumulation test passed")


class TestRNN_LSTM(unittest.TestCase):
    """Unit tests for RNN_LSTM class."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_channel = 121
        self.output_channel = 120
        self.seq_length = 10
        self.hidden_size = 64
        self.num_layers = 2

        if self.logger:
            self.logger.info("Test setup - RNN_LSTM tests")

    def test_init(self):
        """Test RNN_LSTM initialization."""
        if self.logger:
            self.logger.info("Testing RNN_LSTM initialization")

        model = RNN_LSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        param_count = ModelUtils.get_parameter_number(model, self.logger)
        print(
            f"Model Parameters - Total: {param_count['Total']:,}, Trainable: {param_count['Trainable']:,}"
        )

        self.assertIsInstance(model, RNN_LSTM)
        self.assertIsInstance(model, BaseRNN)
        self.assertEqual(model.rnn_type, "lstm")
        self.assertIsInstance(model.rnn, torch.nn.LSTM)

        if self.logger:
            self.logger.success("✅ RNN_LSTM initialization test passed")

    def test_forward(self):
        """Test RNN_LSTM forward pass."""
        if self.logger:
            self.logger.info("Testing RNN_LSTM forward pass")

        model = RNN_LSTM(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

        if self.logger:
            self.logger.success("✅ RNN_LSTM forward test passed")


class TestRNN_GRU(unittest.TestCase):
    """Unit tests for RNN_GRU class."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_channel = 6
        self.output_channel = 4
        self.seq_length = 10
        self.hidden_size = 64
        self.num_layers = 2

        if self.logger:
            self.logger.info("Test setup - RNN_GRU tests")

    def test_init(self):
        """Test RNN_GRU initialization."""
        if self.logger:
            self.logger.info("Testing RNN_GRU initialization")

        model = RNN_GRU(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.assertIsInstance(model, RNN_GRU)
        self.assertIsInstance(model, BaseRNN)
        self.assertEqual(model.rnn_type, "gru")
        self.assertIsInstance(model.rnn, torch.nn.GRU)

        if self.logger:
            self.logger.success("✅ RNN_GRU initialization test passed")

    def test_forward(self):
        """Test RNN_GRU forward pass."""
        if self.logger:
            self.logger.info("Testing RNN_GRU forward pass")

        model = RNN_GRU(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

        if self.logger:
            self.logger.success("✅ RNN_GRU forward test passed")


class TestRNNIntegration(unittest.TestCase):
    """Integration tests for RNN models."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def test_device_transfer_cpu_to_cuda(self):
        """Test model can be moved to CUDA if available."""
        if self.logger:
            self.logger.info("Testing device transfer CPU -> CUDA")

        model = RNN_LSTM(
            feature_channel=6, output_channel=4, hidden_size=64, num_layers=2
        )

        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(16, 6, 10).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)
            if self.logger:
                self.logger.success("✅ CUDA transfer test passed")
        else:
            if self.logger:
                self.logger.info("CUDA not available, skipping test")

    def test_model_parameters_count(self):
        """Test model has expected number of parameters."""
        if self.logger:
            self.logger.info("Testing model parameter count")

        model = RNN_LSTM(
            feature_channel=6, output_channel=4, hidden_size=64, num_layers=2
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)

        if self.logger:
            self.logger.success(
                f"✅ Parameter count test passed: {total_params:,} params"
            )

    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        if self.logger:
            self.logger.info("Testing model serialization")

        import tempfile

        model1 = RNN_LSTM(
            feature_channel=6, output_channel=4, hidden_size=64, num_layers=2
        )

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        # Load model
        model2 = RNN_LSTM(
            feature_channel=6, output_channel=4, hidden_size=64, num_layers=2
        )
        model2.load_state_dict(torch.load(temp_file))

        # Test forward pass produces same output
        x = torch.randn(16, 6, 10)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)

        torch.testing.assert_close(y1, y2)

        # Cleanup
        os.unlink(temp_file)

        if self.logger:
            self.logger.success("✅ Model serialization test passed")


# For backward compatibility with direct unittest runs
def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBaseRNN))
    suite.addTests(loader.loadTestsFromTestCase(TestRNN_LSTM))
    suite.addTests(loader.loadTestsFromTestCase(TestRNN_GRU))
    suite.addTests(loader.loadTestsFromTestCase(TestRNNIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
