# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Unit tests for :mod:`phenonn.evaluation.leakage` and the protocol CLI."""

import unittest
from types import SimpleNamespace
from unittest import mock

from phenonn.evaluation.leakage import (
    LeakageError,
    normalise_protocol,
    parse_years,
    reconcile_resumed_clim_years,
    validate_climatology_years,
    validate_normalisation_years,
    validate_split,
)


class TestParseYears(unittest.TestCase):
    """Year-spec parsing."""

    def test_range_list_and_empty(self):
        """Ranges, comma lists, empty values and unsorted lists."""
        self.assertEqual(parse_years("1992-2009"), list(range(1992, 2010)))
        self.assertEqual(parse_years("2010,2012"), [2010, 2012])
        self.assertEqual(parse_years(""), [])
        self.assertEqual(parse_years([2000, 1999]), [1999, 2000])


class TestProtocolNormalisation(unittest.TestCase):
    """The protocol name is case-insensitive and canonicalised."""

    def test_canonical_names(self):
        """Every accepted spelling maps to A, B or none."""
        cases = {
            "A": "A",
            "a": "A",
            " a ": "A",
            "B": "B",
            "b": "B",
            "none": "none",
            "NONE": "none",
            "None": "none",
            "": "none",
            None: "none",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalise_protocol(raw), expected)

    def test_invalid_name_raises(self):
        """Unknown protocols are rejected with the accepted set."""
        with self.assertRaises(ValueError):
            normalise_protocol("C")


class TestValidateSplit(unittest.TestCase):
    """Split rules for protocols A, B and the default ad-hoc mode."""

    def test_default_protocol_is_none(self):
        """Omitting ``protocol`` must not fail (the historical default)."""
        warnings = validate_split([2000, 2001], [2002])
        self.assertEqual(warnings, [])

    def test_protocol_b_valid(self):
        """Disjoint train/val/test decades pass."""
        validate_split(
            list(range(1992, 2007)),
            [2007, 2008, 2009],
            list(range(2010, 2020)),
            protocol="B",
        )

    def test_protocol_b_rejects_train_val_overlap(self):
        """Train and validation must be disjoint."""
        with self.assertRaisesRegex(LeakageError, "train ∩ val"):
            validate_split([1992, 1993], [1993, 1994], [2010], protocol="B")

    def test_protocol_b_rejects_val_test_overlap(self):
        """Validation and test must be disjoint."""
        with self.assertRaisesRegex(LeakageError, "val ∩ test"):
            validate_split([1992], [2007], [2007, 2010], protocol="B")

    def test_protocol_b_requires_test(self):
        """Protocol B needs an explicit test decade."""
        with self.assertRaisesRegex(LeakageError, "requires explicit test"):
            validate_split([1992], [2007], [], protocol="B")

    def test_protocol_a_allows_same_decade(self):
        """Protocol A deliberately reuses the validation decade for test."""
        warnings = validate_split([1992], [2010], [2010], protocol="A")
        self.assertTrue(any("overlap by design" in w for w in warnings))

    def test_protocol_a_rejects_different_test_decade(self):
        """Protocol A only reports on its validation decade."""
        with self.assertRaisesRegex(LeakageError, "must equal val years"):
            validate_split([1992], [2010], [2011], protocol="A")

    def test_lowercase_protocol_is_accepted(self):
        """A lowercase protocol must not crash the trainer's default path."""
        warnings = validate_split([1992], [2010], [2010], protocol="a")
        self.assertTrue(any("overlap by design" in w for w in warnings))
        validate_split([1992], [2007], [2010], protocol="b")


class TestYearValidators(unittest.TestCase):
    """Normalisation and climatology windows must stay inside training."""

    def test_norm_stats_must_be_training_years(self):
        """Normalisation statistics may not include held-out years."""
        validate_normalisation_years([1992, 2000], [1992, 2000, 2001])
        with self.assertRaisesRegex(LeakageError, "non-training years"):
            validate_normalisation_years([1992, 2015], [1992, 2000])

    def test_climatology_must_be_training_years(self):
        """The anomaly climatology may not include held-out years."""
        validate_climatology_years([1992, 2000], [1992, 2000, 2001], [2010])
        with self.assertRaisesRegex(LeakageError, "non-training years"):
            validate_climatology_years([1992, 2010], [1992, 2000], [2010])

    def test_climatology_overlap_with_prediction_is_warning_only(self):
        """Predicting on training years is a warning, not an error."""
        warnings = validate_climatology_years([1992], [1992, 2000], [1992])
        self.assertTrue(any("overlaps prediction years" in w for w in warnings))


class TestResumedClimYears(unittest.TestCase):
    """Old checkpoints stored ``--clim_years 1992-2010``."""

    def test_leaky_stored_window_is_rebuilt_from_train_years(self):
        """A stored window outside training years is dropped with a warning."""
        args = SimpleNamespace(
            anomaly_mode=True,
            clim_years="1992-2010",
            train_years="1992-2009",
        )
        logger = mock.Mock()
        reconcile_resumed_clim_years(args, set(), logger)
        self.assertEqual(args.clim_years, "1992-2009")
        logger.warning.assert_called_once()

    def test_explicit_cli_window_is_left_for_the_strict_check(self):
        """An explicit bad value must still raise in ``load_shared``."""
        args = SimpleNamespace(
            anomaly_mode=True,
            clim_years="1992-2010",
            train_years="1992-2009",
        )
        logger = mock.Mock()
        reconcile_resumed_clim_years(args, {"clim_years"}, logger)
        self.assertEqual(args.clim_years, "1992-2010")
        logger.warning.assert_not_called()


class TestProtocolCLI(unittest.TestCase):
    """Both entry points must start without ``--protocol``."""

    def test_train_default_and_case_insensitive(self):
        """The trainer defaults to ``none`` and accepts lowercase names."""
        from phenonn.training.args import parse_args

        self.assertEqual(parse_args([]).protocol, "none")
        self.assertEqual(parse_args(["--protocol", "none"]).protocol, "none")
        self.assertEqual(parse_args(["--protocol", "NONE"]).protocol, "none")
        self.assertEqual(parse_args(["--protocol", "a"]).protocol, "A")
        self.assertEqual(parse_args(["--protocol", "b"]).protocol, "B")

    def test_predict_default_and_case_insensitive(self):
        """The predictor defaults to ``none`` and accepts lowercase names."""
        from phenonn.prediction.args import parse_args

        base = ["--checkpoint", "dummy.pth.tar"]
        self.assertEqual(parse_args(base).protocol, "none")
        self.assertEqual(parse_args(base + ["--protocol", "NONE"]).protocol, "none")
        self.assertEqual(parse_args(base + ["--protocol", "a"]).protocol, "A")


if __name__ == "__main__":
    unittest.main()
