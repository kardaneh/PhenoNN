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

import unittest
from types import SimpleNamespace
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
