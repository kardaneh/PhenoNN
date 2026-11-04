# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Regression tests for the climatology baseline helper."""

import unittest

import pandas as pd

from analysis.diagnostics.climatology_baseline import add_climatology


class TestClimatologyBaseline(unittest.TestCase):
    """The helper must accept prediction bundles that already carry lai_clim."""

    def test_existing_lai_clim_does_not_collide(self):
        """A pre-existing lai_clim column is replaced, not suffixed."""
        df = pd.DataFrame(
            {
                "site_id": ["a", "a", "a", "a"],
                "year": [2000, 2001, 2000, 2001],
                "doy": [1, 1, 2, 2],
                "lai_obs": [1.0, 3.0, 2.0, 4.0],
                "lai_pred": [0.0, 0.0, 0.0, 0.0],
                "lai_clim": [99.0, 99.0, 99.0, 99.0],
            }
        )

        out = add_climatology(df, pooled=False, clim_years={2000, 2001})

        self.assertIn("lai_clim", out.columns)
        self.assertNotIn("lai_clim_x", out.columns)
        self.assertNotIn("lai_clim_y", out.columns)
        self.assertEqual(
            out.loc[(out["site_id"] == "a") & (out["doy"] == 1), "lai_clim"].iloc[0],
            2.0,
        )
        self.assertEqual(
            out.loc[(out["site_id"] == "a") & (out["doy"] == 2), "lai_clim"].iloc[0],
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
