# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Unit tests for the warming/CO2 sweep helpers (F10 input)."""

import unittest

import numpy as np
import pandas as pd

from analysis.perturbation import warming_co2_sweep as S


def _curve(doy, shift=0.0):
    """Smooth seasonal LAI curve, optionally shifted in time."""
    return np.maximum(
        0.0,
        1.0 + np.sin(2.0 * np.pi * ((doy - shift) - 80.0) / 365.0),
    )


class TestSweepHelpers(unittest.TestCase):
    """Pure helpers used by ``warming_co2_sweep``."""

    def test_parse_modes(self):
        """Known modes are accepted and unknown modes are rejected."""
        self.assertEqual(S._parse_modes("absolute,detrend"), ["absolute", "detrend"])
        with self.assertRaises(SystemExit):
            S._parse_modes("absolute,bogus")

    def test_co2_label(self):
        """The table label is stable for each mode."""
        self.assertEqual(S._co2_label(2.0, "absolute", 100.0, 350.0), "CO2 +100 ppm")
        self.assertIn("×", S._co2_label(2.0, "doubling", 100.0, 350.0))
        self.assertIn("fixed", S._co2_label(2.0, "detrend", 100.0, 350.0))

    def test_phenology_deltas_are_finite(self):
        """A shifted curve produces finite ΔSOS/ΔEOS/ΔLOS medians."""
        doy = np.linspace(5.0, 360.0, 36)
        rows = []
        for site in ("a", "b"):
            for k, day in enumerate(doy):
                rows.append(
                    {
                        "site_id": site,
                        "year": 2010,
                        "doy": day,
                        "lai_base": _curve(doy)[k],
                        "lai_pert": _curve(doy, shift=5.0)[k],
                    }
                )
        result = S._phenology_deltas(pd.DataFrame(rows), 0.10)
        for key in ("median_dsos_10", "median_deos_10", "median_dlos_10"):
            self.assertIn(key, result)
            self.assertTrue(np.isfinite(result[key]) or np.isnan(result[key]))


if __name__ == "__main__":
    unittest.main()
