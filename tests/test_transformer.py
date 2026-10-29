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
from phenonn.models.transformer import EncoderTorch
from phenonn.utils.model_utils import ModelUtils


class TestEncoderTorch(unittest.TestCase):
    """Unit tests for EncoderTorch (PyTorch-based implementation)."""

    def setUp(self):
        self.batch_size = 32
        self.feature_channel = 121
        self.output_channel = 120
        self.embed_size = 256
        self.num_layers = 3
        self.heads = 4
        self.forward_expansion = 4
        self.seq_length = 10
        self.dropout = 0.2

        self.encoder = EncoderTorch(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            forward_expansion=self.forward_expansion,
            seq_length=self.seq_length,
            dropout=self.dropout,
        )
        self.param_count = ModelUtils.get_parameter_number(self.encoder, None)
        print(
            f"Model Parameters - Total: {self.param_count['Total']:,}, Trainable: {self.param_count['Trainable']:,}"
        )

    def test_initialization(self):
        """Test EncoderTorch initialization."""
        self.assertEqual(self.encoder.embed_size, self.embed_size)
        self.assertEqual(self.encoder.seq_length, self.seq_length)

        self.assertIsInstance(self.encoder.input_proj, torch.nn.Linear)
        self.assertIsInstance(self.encoder.position_embedding, torch.nn.Embedding)
        self.assertIsInstance(self.encoder.encoder, torch.nn.TransformerEncoder)
        self.assertIsInstance(self.encoder.dropout, torch.nn.Dropout)
        self.assertIsInstance(self.encoder.final, torch.nn.Conv1d)

    def test_forward_shape(self):
        """Test forward output shape."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = self.encoder(x)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_with_padding_mask(self):
        """Test forward with src_key_padding_mask."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)

        # mask shape: (batch, seq_len)
        padding_mask = torch.zeros(self.batch_size, self.seq_length).bool()

        y = self.encoder(x, src_key_padding_mask=padding_mask)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_with_attention_mask(self):
        """Test forward with attention mask (causal or full)."""
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)

        # mask shape: (seq_len, seq_len)
        attn_mask = torch.zeros(self.seq_length, self.seq_length)

        y = self.encoder(x, mask=attn_mask)

        self.assertEqual(
            y.shape, (self.batch_size, self.output_channel, self.seq_length)
        )

    def test_forward_different_batch_sizes(self):
        """Test different batch sizes."""
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, self.feature_channel, self.seq_length)
            y = self.encoder(x)
            self.assertEqual(y.shape, (bs, self.output_channel, self.seq_length))

    def test_model_serialization(self):
        """Test save/load consistency."""
        import tempfile

        model1 = self.encoder

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        model2 = EncoderTorch(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            forward_expansion=self.forward_expansion,
            seq_length=self.seq_length,
            dropout=self.dropout,
        )
        model2.load_state_dict(torch.load(temp_file))

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)

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
            model = self.encoder.cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()

            y = model(x)
            self.assertTrue(y.is_cuda)


class TestEncoderTorchIntegration(unittest.TestCase):
    """Integration tests."""

    def test_different_sequence_lengths(self):
        """Test variable sequence lengths."""
        for sl in [5, 10, 20]:
            encoder = EncoderTorch(
                feature_channel=6,
                output_channel=4,
                embed_size=64,
                num_layers=2,
                heads=4,
                forward_expansion=4,
                seq_length=sl,
                dropout=0.1,
            )

            x = torch.randn(16, 6, sl)
            y = encoder(x)

            self.assertEqual(y.shape, (16, 4, sl))

    def test_different_embed_sizes(self):
        """Test variable embedding sizes."""
        for es in [32, 64, 128]:
            encoder = EncoderTorch(
                feature_channel=6,
                output_channel=4,
                embed_size=es,
                num_layers=2,
                heads=4,
                forward_expansion=4,
                seq_length=10,
                dropout=0.1,
            )

            x = torch.randn(16, 6, 10)
            y = encoder(x)

            self.assertEqual(y.shape, (16, 4, 10))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderTorch))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoderTorchIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
