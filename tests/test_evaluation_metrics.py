# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Unit tests for :mod:`phenonn.evaluation.metrics`."""

import datetime
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from phenonn.evaluation import metrics as M


def _frame(site_means=(1.0, 3.0), n_years=2, noise=0.0, seed=0):
    """Build a canonical 36-dekad frame for a couple of sites and years."""
    rng = np.random.default_rng(seed)
    rows = []
    # Canonical 36-dekad calendar: days 5/15/25 of every month.
    doys = [
        datetime.date(2001, month, day).timetuple().tm_yday
        for month in range(1, 13)
        for day in (5, 15, 25)
    ]
    for s, mu in enumerate(site_means):
        clim = mu + np.sin(np.linspace(0, 2 * np.pi, 36))
        for y in range(2010, 2010 + n_years):
            obs = clim + rng.normal(0, 0.2, 36)
            pred = obs + rng.normal(0, noise, 36)
            for k, d in enumerate(doys):
                rows.append(
                    {
                        "site_id": f"s{s}",
                        "year": y,
                        "doy": d,
                        "lai_obs": obs[k],
                        "lai_pred": pred[k],
                        "lai_clim": clim[k],
                    }
                )
    return pd.DataFrame(rows)


class TestEvaluatePredictions(unittest.TestCase):
    """Pooled, centred and anomaly metrics."""

    def test_perfect_prediction(self):
        """A perfect prediction scores R2 = 1 and zero error."""
        m = M.evaluate_predictions(_frame(noise=0.0), n_boot=0)
        self.assertAlmostEqual(m["overall_r2"], 1.0, places=10)
        self.assertLess(m["rmse"], 1e-10)
        self.assertLess(abs(m["bias"]), 1e-10)
        self.assertAlmostEqual(m["slope"], 1.0, places=10)
        self.assertAlmostEqual(m["centered_r2"], 1.0, places=10)
        self.assertAlmostEqual(m["pearson_r2"], 1.0, places=10)
        self.assertEqual(m["per_site"]["pct_gt0"], 100.0)

    def test_centred_r2_removes_site_means(self):
        """A between-site offset only hurts pooled R2, not centred R2."""
        df = _frame(noise=0.0)
        df["lai_pred"] = df["lai_pred"] + df["site_id"].map({"s0": -1.0, "s1": 1.0})
        m = M.evaluate_predictions(df, n_boot=0)
        self.assertLess(m["overall_r2"], 0.9)
        self.assertGreater(m["centered_r2"], 0.999)

    def test_constant_observation_is_nan(self):
        """A constant target has undefined R2, not a huge negative value."""
        df = _frame(noise=0.1)
        df["lai_obs"] = 1.0
        m = M.evaluate_predictions(df, n_boot=0)
        self.assertTrue(np.isnan(m["overall_r2"]))
        self.assertTrue(np.isnan(m["centered_r2"]))
        self.assertGreaterEqual(m["rmse"], 0)

    def test_anomaly_metrics(self):
        """A climatology prediction has no anomaly skill on noisy data."""
        df = _frame(noise=0.0)
        df["lai_pred"] = df["lai_clim"]
        m = M.evaluate_predictions(df, n_boot=0)
        self.assertIsNotNone(m["anomaly"])
        self.assertLessEqual(m["anomaly"]["overall_r2"], 1e-6)

    def test_bootstrap_is_reproducible_and_ordered(self):
        """The seeded bootstrap returns identical, ordered bounds."""
        df = _frame(noise=0.15, n_years=3)
        a = M.evaluate_predictions(df, n_boot=40, seed=7)["ci95"]
        b = M.evaluate_predictions(df, n_boot=40, seed=7)["ci95"]
        self.assertEqual(a, b)
        lo, hi = a["overall_r2"]
        self.assertLessEqual(lo, hi)
        m = M.evaluate_predictions(df, n_boot=0)
        self.assertLessEqual(lo, m["overall_r2"] + 1e-6)
        self.assertLessEqual(m["overall_r2"] - 1e-6, hi)


class TestMetricTables(unittest.TestCase):
    """Per-site, per-site-year and per-DOY tables."""

    def test_per_site_and_doy_tables(self):
        """The derived tables have the documented shapes and columns."""
        df = _frame(noise=0.1)
        ps = M.per_site_metrics(df)
        self.assertTrue(
            {"site_id", "n", "r2", "rmse", "mae", "bias"}.issubset(ps.columns)
        )
        self.assertEqual(len(ps), 2)
        self.assertEqual(len(M.per_siteyear_metrics(df)), 4)
        rd = M.rmse_by_doy(df)
        self.assertEqual(len(rd), 36)
        self.assertIn("rmse_clim", rd.columns)

    def test_phenology_sos_eos(self):
        """SOS/EOS/LOS follow the requested threshold fractions."""
        doy = np.array([5, 15, 25, 35, 45, 55, 65, 75])
        lai = np.array([0.0, 0.0, 0.2, 1.0, 2.0, 1.0, 0.2, 0.0])
        out = M.sos_eos_curve(doy, lai, fractions=(0.25,))
        self.assertEqual(out["sos_25"], 35.0)
        self.assertEqual(out["eos_25"], 55.0)
        self.assertEqual(out["los_25"], 20.0)


class TestJsonable(unittest.TestCase):
    """Strict-JSON sanitisation of NaN and infinity."""

    def test_jsonable_sanitises_non_finite(self):
        """NaN and inf become ``null`` on disk."""
        from phenonn.evaluation.reproducibility import jsonable, write_json

        payload = {"a": float("nan"), "b": np.float32("inf"), "c": [1.0, np.nan]}
        clean = jsonable(payload)
        self.assertIsNone(clean["a"])
        self.assertIsNone(clean["b"])
        self.assertIsNone(clean["c"][1])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meta.json")
            write_json(path, payload)
            with open(path) as f:
                text = f.read()
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            with open(path) as f:
                self.assertIsNone(json.load(f)["a"])


if __name__ == "__main__":
    unittest.main()
