# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import sys
import unittest
from types import SimpleNamespace
import torch
from phenonn.utils.evaluater import (
    nan_safe_loss,
    nan_safe_mse,
    make_loss_fn,
)


def _args(**overrides):
    """Namespace with sensible defaults for make_loss_fn."""
    base = dict(
        loss_type="mse",
        huber_beta=1.0,
        gradient_loss_weight=0.5,
        peak_penalty_weight=0.0,
        corr_loss_weight=0.0,
        amp_loss_weight=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBaseLossTypes(unittest.TestCase):
    """Base loss types: mse, mae, huber/smoothl1, nmse, nmae, gradient."""

    def setUp(self):
        torch.manual_seed(0)
        self.pred = torch.randn(2, 1, 36)
        self.target = torch.randn(2, 1, 36)

    def test_mse_zero_when_identical(self):
        loss = nan_safe_loss(self.pred, self.pred.clone(), loss_type="mse")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_mse_positive_when_different(self):
        loss = nan_safe_loss(self.pred, self.target, loss_type="mse")
        self.assertGreater(loss.item(), 0.0)

    def test_mae_zero_when_identical(self):
        loss = nan_safe_loss(self.pred, self.pred.clone(), loss_type="mae")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_huber_matches_smoothl1(self):
        h = nan_safe_loss(self.pred, self.target, loss_type="huber", huber_beta=1.0)
        s = nan_safe_loss(self.pred, self.target, loss_type="smoothl1", huber_beta=1.0)
        self.assertAlmostEqual(h.item(), s.item(), places=6)

    def test_huber_beta_changes_loss(self):
        small = nan_safe_loss(self.pred, self.target, loss_type="huber", huber_beta=0.1)
        large = nan_safe_loss(
            self.pred, self.target, loss_type="huber", huber_beta=10.0
        )
        self.assertNotAlmostEqual(small.item(), large.item(), places=6)

    def test_nmse_positive(self):
        loss = nan_safe_loss(self.pred, self.target, loss_type="nmse")
        self.assertGreater(loss.item(), 0.0)

    def test_nmae_positive(self):
        loss = nan_safe_loss(self.pred, self.target, loss_type="nmae")
        self.assertGreater(loss.item(), 0.0)

    def test_gradient_loss_positive(self):
        loss = nan_safe_loss(
            self.pred, self.target, loss_type="gradient", gradient_loss_weight=0.5
        )
        self.assertGreater(loss.item(), 0.0)


class TestNaNSafety(unittest.TestCase):
    """NaN masking behaviour."""

    def test_partial_nan_target_is_masked(self):
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[1.0, float("nan"), 3.0, float("nan")]])
        loss = nan_safe_loss(pred, target, loss_type="mse")
        # Two valid positions, both perfect → loss is 0.
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_all_nan_target_gives_zero_gradient_scalar(self):
        pred = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        target = torch.full_like(pred, float("nan"))
        loss = nan_safe_loss(pred, target, loss_type="mse")
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(pred.grad is None or torch.all(pred.grad == 0))

    def test_nan_target_matches_dropped_positions(self):
        """Loss with NaN positions must equal loss on the dropped subset."""
        pred_full = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target_full = torch.tensor([[1.5, 2.0, 3.5, 4.0]])
        target_nan = target_full.clone()
        target_nan[0, 1] = float("nan")
        target_nan[0, 3] = float("nan")

        loss_full = nan_safe_loss(pred_full, target_full, loss_type="mse")
        loss_nan = nan_safe_loss(pred_full, target_nan, loss_type="mse")

        pred_red = torch.tensor([[1.0, 3.0]])
        target_red = torch.tensor([[1.5, 3.5]])
        loss_red = nan_safe_loss(pred_red, target_red, loss_type="mse")

        self.assertAlmostEqual(loss_nan.item(), loss_red.item(), places=6)
        self.assertNotAlmostEqual(loss_nan.item(), loss_full.item(), places=6)


class TestOptionalPenalties(unittest.TestCase):
    """peak, corr, and amp penalties."""

    def test_peak_penalty_zero_when_peaks_align(self):
        pred = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        target = pred.clone()
        loss = nan_safe_loss(pred, target, loss_type="mse", peak_penalty_weight=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_peak_penalty_positive_when_peaks_differ(self):
        pred = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        base = nan_safe_loss(pred, target, loss_type="mse")
        penal = nan_safe_loss(pred, target, loss_type="mse", peak_penalty_weight=1.0)
        self.assertGreater(penal.item(), base.item())

    def test_corr_term_penalises_flat_prediction(self):
        target = torch.linspace(0.0, 1.0, 36).unsqueeze(0).unsqueeze(0)
        flat_pred = torch.zeros_like(target)
        loss = nan_safe_loss(flat_pred, target, loss_type="mse", corr_weight=1.0)
        self.assertGreater(loss.item(), 0.5)

    def test_corr_term_small_when_linear_match(self):
        target = torch.linspace(0.0, 1.0, 36).unsqueeze(0).unsqueeze(0)
        pred = target.clone()
        loss = nan_safe_loss(pred, target, loss_type="mse", corr_weight=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_amp_term_penalises_wrong_amplitude(self):
        target = torch.linspace(0.0, 1.0, 36).unsqueeze(0).unsqueeze(0)
        flat_pred = torch.zeros_like(target)
        loss = nan_safe_loss(flat_pred, target, loss_type="mse", amp_weight=1.0)
        self.assertGreater(loss.item(), 0.0)

    def test_amp_term_zero_when_amplitude_matches(self):
        target = torch.linspace(0.0, 1.0, 36).unsqueeze(0).unsqueeze(0)
        pred = target.clone()
        loss = nan_safe_loss(pred, target, loss_type="mse", amp_weight=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)


class TestErrorPath(unittest.TestCase):
    """Input validation."""

    def test_invalid_loss_type_raises(self):
        pred = torch.zeros(1, 1, 36)
        target = torch.zeros(1, 1, 36)
        with self.assertRaises(ValueError):
            nan_safe_loss(pred, target, loss_type="does_not_exist")


class TestMakeLossFn(unittest.TestCase):
    """The closure returned by make_loss_fn."""

    def setUp(self):
        torch.manual_seed(0)
        self.pred = torch.randn(2, 1, 36)
        self.target = torch.randn(2, 1, 36)

    def test_returns_callable(self):
        fn = make_loss_fn(_args(loss_type="mse"))
        self.assertTrue(callable(fn))
        loss = fn(self.pred, self.target)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)

    def test_missing_optional_penalties_default_to_zero(self):
        """Optional weights default to 0 when absent from the namespace."""
        args = SimpleNamespace(
            loss_type="mse", huber_beta=1.0, gradient_loss_weight=0.5
        )
        fn = make_loss_fn(args)
        loss = fn(self.pred, self.target)
        self.assertTrue(torch.isfinite(loss))

    def test_propagates_penalties(self):
        fn = make_loss_fn(_args(loss_type="mse", peak_penalty_weight=1.0))
        with_penalty = fn(self.pred, self.target).item()
        base = nan_safe_loss(self.pred, self.target, loss_type="mse").item()
        self.assertGreater(with_penalty, base)


class TestNanSafeMSEAlias(unittest.TestCase):
    """nan_safe_mse backward-compatibility alias."""

    def test_alias_matches_mse(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 1, 36)
        target = torch.randn(2, 1, 36)
        a = nan_safe_mse(pred, target).item()
        b = nan_safe_loss(pred, target, loss_type="mse").item()
        self.assertAlmostEqual(a, b, places=6)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBaseLossTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestNaNSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestOptionalPenalties))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorPath))
    suite.addTests(loader.loadTestsFromTestCase(TestMakeLossFn))
    suite.addTests(loader.loadTestsFromTestCase(TestNanSafeMSEAlias))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
