# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Prediction-side leakage checks for old checkpoints.

This module imports :mod:`phenonn.prediction.predict` and therefore torch.  It
is kept separate from ``tests.test_leakage`` so the pure leakage tests stay
dependency-light.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from phenonn.evaluation.leakage import LeakageError


class TestPredictClimatology(unittest.TestCase):
    """Old checkpoints may store a climatology without its years."""

    @staticmethod
    def _args(protocol):
        return SimpleNamespace(
            clim_years="",
            no_climatology=False,
            clim_target_dir="",
            allow_leakage=False,
            protocol=protocol,
            test_years="2010-2019",
        )

    @staticmethod
    def _ckpt():
        return {"anomaly_clim": {"site": [0.0] * 36}, "clim_years": []}

    def test_old_climatology_warns_in_adhoc_mode(self):
        """Without a stored window, ``none`` warns but still runs."""
        from phenonn.prediction.predict import resolve_climatology

        logger = mock.Mock()
        clim, _ = resolve_climatology(
            self._args("none"),
            self._ckpt(),
            {"train_years": "1992-2009"},
            "",
            [],
            [2010],
            logger,
        )
        self.assertIsNotNone(clim)
        self.assertTrue(
            any("no clim_years" in str(call) for call in logger.warning.call_args_list)
        )

    def test_old_climatology_raises_for_protocol_b(self):
        """Under Protocol B an unverifiable window is a hard error."""
        from phenonn.prediction.predict import resolve_climatology

        with self.assertRaises(LeakageError):
            resolve_climatology(
                self._args("b"),
                self._ckpt(),
                {"train_years": "1992-2009"},
                "",
                [],
                [2010],
                mock.Mock(),
            )


class TestMetricClimatology(unittest.TestCase):
    """The stored metric climatology wins over a rebuild."""

    @staticmethod
    def _args(protocol="none", **extra):
        base = SimpleNamespace(
            clim_years="",
            no_climatology=False,
            clim_target_dir="",
            allow_leakage=False,
            protocol=protocol,
            test_years="2010-2019",
        )
        for key, value in extra.items():
            setattr(base, key, value)
        return base

    def test_metric_clim_is_used_without_recompute(self):
        from phenonn.prediction.predict import resolve_climatology

        table = {"site": np.zeros(36, dtype=np.float32)}
        logger = mock.Mock()
        clim, years = resolve_climatology(
            self._args(),
            {
                "metric_clim": table,
                "metric_clim_years": [1992, 2009],
                "anomaly_clim": {"other": np.ones(36, dtype=np.float32)},
            },
            {"train_years": "1992-2009"},
            "",
            ["site"],
            [2010],
            logger,
        )
        self.assertIs(clim, table)
        self.assertEqual(years, [1992, 2009])
        self.assertTrue(
            any("metric_clim" in str(call) for call in logger.info.call_args_list)
        )

    def test_metric_clim_window_is_checked_for_leakage(self):
        from phenonn.prediction.predict import resolve_climatology

        with self.assertRaises(LeakageError):
            resolve_climatology(
                self._args("A"),
                {
                    "metric_clim": {"site": np.zeros(36, dtype=np.float32)},
                    "metric_clim_years": [2010],
                },
                {"train_years": "1992-2009"},
                "",
                ["site"],
                [2010],
                mock.Mock(),
            )

    def test_partial_metric_clim_falls_back_to_rebuild(self):
        from phenonn.prediction.predict import resolve_climatology

        logger = mock.Mock()
        clim, _ = resolve_climatology(
            self._args(),
            {
                "metric_clim": {"site_a": np.zeros(36, dtype=np.float32)},
                "metric_clim_years": [1992, 2009],
            },
            {"train_years": "1992-2009"},
            "",
            ["site_a", "site_b"],
            [2010],
            logger,
        )
        self.assertIn("site_b", clim)
        self.assertTrue(
            any("rebuilding" in str(call) for call in logger.info.call_args_list)
        )


class TestLoadCheckpointSchema(unittest.TestCase):
    """The normalised checkpoint must keep the climatology window."""

    def _load(self, raw):
        import torch

        from phenonn.prediction.predict import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pth.tar")
            torch.save(raw, path)
            return load_checkpoint(SimpleNamespace(checkpoint=path), "cpu", mock.Mock())

    def test_top_level_clim_years_survive(self):
        raw = {
            "state_dict": {"w": None},
            "args": {"clim_years": "1992-2009"},
            "epoch": 4,
            "anomaly_clim": {"site": [0.0] * 36},
            "anomaly_mode": True,
            "clim_years": list(range(1992, 2010)),
        }
        ckpt = self._load(raw)
        self.assertEqual(ckpt["clim_years"], list(range(1992, 2010)))

    def test_clim_years_fall_back_to_training_args(self):
        raw = {
            "state_dict": {"w": None},
            "args": {"clim_years": "1992-2009"},
            "epoch": 4,
            "anomaly_clim": {"site": [0.0] * 36},
            "anomaly_mode": True,
        }
        ckpt = self._load(raw)
        self.assertEqual(ckpt["clim_years"], list(range(1992, 2010)))

    def test_metric_clim_and_years_survive(self):
        raw = {
            "state_dict": {"w": None},
            "args": {},
            "epoch": 4,
            "metric_clim": {"site": [0.0] * 36},
            "metric_clim_years": [1992, 2009],
        }
        ckpt = self._load(raw)
        self.assertIn("site", ckpt["metric_clim"])
        self.assertEqual(ckpt["metric_clim_years"], [1992, 2009])


if __name__ == "__main__":
    unittest.main()
