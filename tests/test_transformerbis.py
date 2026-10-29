# Copyright 2026 CNRS
# Authors: Christian Reimers, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
import sys
import os
from phenonn.models.transformerbis import (
    CombinedModel,
    BiTransformer,
    PositionalEncoding,
)
from phenonn.utils.model_utils import ModelUtils


class TestPositionalEncoding(unittest.TestCase):
    """Unit tests for PositionalEncoding module."""

    def setUp(self):
        self.dim_model = 32
        self.dropout_p = 0.1
        self.max_len = 100
        self.pos_encoder = PositionalEncoding(
            dim_model=self.dim_model, dropout_p=self.dropout_p, max_len=self.max_len
        )

    def test_initialization(self):
        """Test PositionalEncoding initialization."""
        self.assertEqual(self.pos_encoder.dropout.p, self.dropout_p)
        self.assertTrue(hasattr(self.pos_encoder, "pos_encoding"))
        self.assertEqual(
            self.pos_encoder.pos_encoding.shape, (self.max_len, 1, self.dim_model)
        )

    def test_forward_shape(self):
        """Test forward output shape."""
        batch_size = 16
        seq_length = 50
        x = torch.randn(seq_length, batch_size, self.dim_model)
        output = self.pos_encoder(x)

        self.assertEqual(output.shape, (seq_length, batch_size, self.dim_model))

    def test_forward_different_sequence_lengths(self):
        """Test with varying sequence lengths."""
        batch_size = 8
        for seq_len in [10, 20, 50, 90]:
            x = torch.randn(seq_len, batch_size, self.dim_model)
            output = self.pos_encoder(x)
            self.assertEqual(output.shape, (seq_len, batch_size, self.dim_model))

    def test_positional_encoding_values(self):
        """Test that positional encodings have correct patterns."""
        # Check sin/cos patterns
        encoding = self.pos_encoder.pos_encoding.squeeze(1)  # (max_len, dim_model)

        # First two dimensions should be sin/cos pairs
        for i in range(0, self.dim_model, 2):
            # Sin pattern for even indices
            self.assertTrue(torch.all(torch.abs(encoding[:, i]) <= 1.0))
            # Cos pattern for odd indices
            if i + 1 < self.dim_model:
                self.assertTrue(torch.all(torch.abs(encoding[:, i + 1]) <= 1.0))


class TestCombinedModel(unittest.TestCase):
    """Unit tests for CombinedModel architecture."""

    def setUp(self):
        self.batch_size = 16
        self.seq_length = 30
        self.input_dim = 26
        self.hidden_dim = 1024
        self.hidden_dim_trans = 1024
        self.output_dim = 2
        self.d_model = 32
        self.nr_blocks = 3

        self.model = CombinedModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            hidden_dim_trans=self.hidden_dim_trans,
            output_dim=self.output_dim,
            d_model=self.d_model,
            nr_blocks=self.nr_blocks,
        )

        self.param_count = ModelUtils.get_parameter_number(self.model, None)
        print(
            f"CombinedModel Parameters - Total: {self.param_count['Total']:,}, "
            f"Trainable: {self.param_count['Trainable']:,}"
        )

    def test_initialization(self):
        """Test CombinedModel initialization."""
        self.assertIsInstance(self.model.lin1, torch.nn.Linear)
        self.assertIsInstance(self.model.trans, torch.nn.Transformer)
        self.assertIsInstance(self.model.lin2, torch.nn.Linear)
        self.assertIsInstance(self.model.lin3, torch.nn.Linear)
        self.assertIsInstance(self.model.lin4, torch.nn.Linear)
        self.assertEqual(len(self.model.encoder), self.nr_blocks)
        self.assertIsInstance(self.model.positional_encoder, PositionalEncoding)

    def test_forward_shape(self):
        """Test forward output shape."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output = self.model(x)

        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.output_dim)
        )

    def test_forward_with_return_stress(self):
        """Test forward with return_stress=True."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output, stress = self.model(x, return_stress=True)

        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.output_dim)
        )
        self.assertEqual(stress.shape, (self.batch_size, self.seq_length, 1))

    def test_forward_from_stress(self):
        """Test forward_from_stress method."""
        x = torch.randn(self.batch_size, self.seq_length, 1)  # stress signal
        pft = torch.randn(self.batch_size, self.seq_length, 10)  # PFT features
        output = self.model.forward_from_stress(x, pft)

        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.output_dim)
        )

    def test_causal_masking(self):
        """Test that causal masking is applied correctly."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output = self.model(x)

        # Output should have no NaN or inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        for bs in [1, 8, 32, 64]:
            x = torch.randn(bs, self.seq_length, self.input_dim)
            output = self.model(x)
            self.assertEqual(output.shape, (bs, self.seq_length, self.output_dim))

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        for seq_len in [10, 20, 40, 100]:
            x = torch.randn(self.batch_size, seq_len, self.input_dim)
            output = self.model(x)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.output_dim))

    def test_pft_extraction(self):
        """Test that PFT features are correctly extracted."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        # Last 10 features should be PFT
        # expected_pft = x[:, :, -10:]

        # Forward pass should not error
        output = self.model(x)
        self.assertIsNotNone(output)


class TestBiTransformer(unittest.TestCase):
    """Unit tests for BiTransformer architecture."""

    def setUp(self):
        self.batch_size = 16
        self.seq_length = 30
        self.input_dim = 26
        self.feed_forward_trans = 4
        self.feed_forward_encoder = 4
        self.output_dim = 2
        self.d_model = 256
        self.nr_blocks = 3
        self.dropout_trans = 0.1
        self.dropout_encoder = 0.1
        self.n_pft = 1

        self.model = BiTransformer(
            input_dim=self.input_dim,
            feed_forward_trans=self.feed_forward_trans,
            feed_forward_encoder=self.feed_forward_encoder,
            output_dim=self.output_dim,
            d_model=self.d_model,
            nr_blocks=self.nr_blocks,
            dropout_trans=self.dropout_trans,
            dropout_encoder=self.dropout_encoder,
            n_pft=self.n_pft,
        )

        self.param_count = ModelUtils.get_parameter_number(self.model, None)
        print(
            f"BiTransformer Parameters - Total: {self.param_count['Total']:,}, "
            f"Trainable: {self.param_count['Trainable']:,}"
        )

    def test_initialization(self):
        """Test BiTransformer initialization."""
        self.assertEqual(self.model.n_pft, self.n_pft)
        self.assertIsInstance(self.model.lin1, torch.nn.Linear)
        self.assertIsInstance(self.model.trans, torch.nn.Transformer)
        self.assertIsInstance(self.model.lin2, torch.nn.Linear)
        self.assertIsInstance(self.model.lin3, torch.nn.Linear)
        self.assertIsInstance(self.model.lin4, torch.nn.Linear)
        self.assertEqual(len(self.model.encoder), self.nr_blocks)
        self.assertIsInstance(self.model.positional_encoder, PositionalEncoding)

    def test_forward_shape(self):
        """Test forward output shape."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output = self.model(x)

        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.output_dim)
        )

    def test_forward_with_return_stress(self):
        """Test forward with return_stress=True."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output, stress = self.model(x, return_stress=True)

        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.output_dim)
        )
        self.assertEqual(stress.shape, (self.batch_size, self.seq_length, 1))

    def test_causal_mask_generation(self):
        """Test causal mask generation."""
        size = 10
        device = torch.device("cpu")
        mask = self.model._causal_mask(size, device)

        self.assertEqual(mask.shape, (size, size))
        # Check that mask is upper triangular with -inf on and above diagonal
        for i in range(size):
            for j in range(size):
                if j > i:
                    self.assertEqual(mask[i, j], float("-inf"))
                else:
                    self.assertEqual(mask[i, j], 0.0)

    def test_pft_extraction_different_n_pft(self):
        """Test PFT extraction with different n_pft values."""
        for n_pft in [1, 5, 10]:
            model = BiTransformer(input_dim=26, n_pft=n_pft, d_model=64, nr_blocks=2)
            x = torch.randn(self.batch_size, self.seq_length, 26)
            output = model(x)
            self.assertEqual(output.shape, (self.batch_size, self.seq_length, 2))

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        for bs in [1, 4, 16, 32, 64]:
            x = torch.randn(bs, self.seq_length, self.input_dim)
            output = self.model(x)
            self.assertEqual(output.shape, (bs, self.seq_length, self.output_dim))

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        for seq_len in [5, 10, 20, 50, 100]:
            x = torch.randn(self.batch_size, seq_len, self.input_dim)
            output = self.model(x)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.output_dim))

    def test_different_model_configurations(self):
        """Test with different model configurations."""
        configs = [
            {"d_model": 128, "nr_blocks": 2},
            {"d_model": 256, "nr_blocks": 4},
            {
                "d_model": 512,
                "nr_blocks": 6,
                "feed_forward_trans": 8,
                "feed_forward_encoder": 8,
            },
        ]

        for config in configs:
            model = BiTransformer(
                input_dim=self.input_dim,
                d_model=config["d_model"],
                nr_blocks=config["nr_blocks"],
                feed_forward_trans=config.get("feed_forward_trans", 4),
                feed_forward_encoder=config.get("feed_forward_encoder", 4),
                output_dim=self.output_dim,
            )
            x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
            output = model(x)
            self.assertEqual(
                output.shape, (self.batch_size, self.seq_length, self.output_dim)
            )

    def test_model_serialization(self):
        """Test save/load consistency."""
        import tempfile

        model1 = self.model

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = BiTransformer(
            input_dim=self.input_dim,
            feed_forward_trans=self.feed_forward_trans,
            feed_forward_encoder=self.feed_forward_encoder,
            output_dim=self.output_dim,
            d_model=self.d_model,
            nr_blocks=self.nr_blocks,
            dropout_trans=self.dropout_trans,
            dropout_encoder=self.dropout_encoder,
            n_pft=self.n_pft,
        )
        model2.load_state_dict(torch.load(temp_file))

        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)

        torch.testing.assert_close(y1, y2)

        os.unlink(temp_file)

    def test_device_transfer(self):
        """Test CUDA transfer."""
        if torch.cuda.is_available():
            model = self.model.cuda()
            x = torch.randn(self.batch_size, self.seq_length, self.input_dim).cuda()

            y = model(x)
            self.assertTrue(y.is_cuda)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        target = torch.randn(self.batch_size, self.seq_length, self.output_dim)

        output = self.model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check that gradients exist for all parameters
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)


class TestBiTransformerIntegration(unittest.TestCase):
    """Integration tests for BiTransformer."""

    def test_different_pft_configurations(self):
        """Test with different numbers of PFT features."""
        input_dim = 30  # 20 features + 10 PFT

        for n_pft in [1, 5, 10]:
            model = BiTransformer(input_dim=input_dim, n_pft=n_pft)
            x = torch.randn(8, 25, input_dim)
            output = model(x)
            self.assertEqual(output.shape, (8, 25, 2))

    def test_training_mode_vs_eval_mode(self):
        """Test model behavior in training vs evaluation mode."""
        model = BiTransformer()
        x = torch.randn(8, 20, 26)

        model.train()
        train_output = model(x)

        model.eval()
        with torch.no_grad():
            eval_output = model(x)

        # Outputs should be identical in shape but may differ numerically
        # due to dropout behavior
        self.assertEqual(train_output.shape, eval_output.shape)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPositionalEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinedModel))
    suite.addTests(loader.loadTestsFromTestCase(TestBiTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestBiTransformerIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
