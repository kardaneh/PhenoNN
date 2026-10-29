# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Smoke tests for the hand-run analyses under ``analysis/``.

No DATA, checkpoints, or GPUs: every analysis module must import cleanly,
and every CLI exposing ``parse_args`` must answer ``--help`` with exit 0.
The real runs (documented in ``analysis/README.md``) stay manual.
"""

import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ANALYSIS_PACKAGES = [
    "analysis",
    "analysis.diagnostics",
    "analysis.feature_ablation",
    "analysis.memory_analysis",
    "analysis.figures",
    "analysis.perturbation",
    "analysis.pft_linearity",
    "analysis.year_generalization",
]

ANALYSIS_SCRIPTS = [
    "analysis._checkpoint_utils",
    "analysis.diagnostics.annual_trend_study",
    "analysis.diagnostics.climatology_baseline",
    "analysis.diagnostics.climatology_skill",
    "analysis.diagnostics.diagnose_grid_alignment",
    "analysis.diagnostics.diagnose_selected_pixels",
    "analysis.diagnostics.inspect_era5_pixelset",
    "analysis.diagnostics.pft_amplitude_correlation",
    "analysis.diagnostics.pft_curves",
    "analysis.diagnostics.pft_diagnostics",
    "analysis.diagnostics.print_lai_pft_sites",
    "analysis.diagnostics.spatial_plot_selected_pixels",
    "analysis.feature_ablation._common",
    "analysis.feature_ablation._hooks",
    "analysis.feature_ablation._profiles",
    "analysis.feature_ablation.aggregate",
    "analysis.feature_ablation.gen_jobs",
    "analysis.feature_ablation.run_ablate_combo_models",
    "analysis.feature_ablation.run_ablate_inproc",
    "analysis.feature_ablation.smoke",
    "analysis.feature_ablation.train_ablate",
    "analysis.figures.figures",
    "analysis.figures.make_all",
    "analysis.figures.registry",
    "analysis.figures.style",
    "analysis.figures.tables",
    "analysis.memory_analysis.run_memory_kernel",
    "analysis.perturbation._perturb",
    "analysis.perturbation.aggregate_perturbation",
    "analysis.perturbation.drought",
    "analysis.perturbation.heatwave",
    "analysis.perturbation.run_perturbation",
    "analysis.perturbation.warming_co2",
    "analysis.perturbation.warming_co2_sweep",
    "analysis.perturbation.winter_frost",
    "analysis.pft_linearity.compare_mixing",
    "analysis.year_generalization.rank_years_by_temp",
    "analysis.year_generalization.run_year_split",
]

STUDY_COMMAND_MODULES = [
    "analysis.memory_analysis.run_memory_kernel",
    "analysis.perturbation.drought",
    "analysis.perturbation.heatwave",
    "analysis.perturbation.warming_co2",
    "analysis.perturbation.warming_co2_sweep",
    "analysis.perturbation.winter_frost",
    "analysis.perturbation.aggregate_perturbation",
    "analysis.pft_linearity.compare_mixing",
    "analysis.year_generalization.rank_years_by_temp",
    "analysis.year_generalization.run_year_split",
    "analysis.figures.make_all",
]

# Only modules exposing parse_args; climatology_skill,
# diagnose_grid_alignment and spatial_plot_selected_pixels build their
# argparse inline in main() and are covered by the import test above.
HELP_MODULES = [
    "analysis.diagnostics.annual_trend_study",
    "analysis.diagnostics.climatology_baseline",
    "analysis.diagnostics.diagnose_selected_pixels",
    "analysis.diagnostics.inspect_era5_pixelset",
    "analysis.diagnostics.pft_amplitude_correlation",
    "analysis.diagnostics.pft_curves",
    "analysis.diagnostics.pft_diagnostics",
    "analysis.diagnostics.print_lai_pft_sites",
]


class TestAnalysisImports(unittest.TestCase):
    """Every analysis module imports without DATA or checkpoints."""

    def test_all_modules_import(self):
        for name in ANALYSIS_PACKAGES + ANALYSIS_SCRIPTS:
            with self.subTest(module=name):
                importlib.import_module(name)


class TestAnalysisHelp(unittest.TestCase):
    """Every CLI exposing parse_args answers --help with exit 0."""

    def test_help_exits_zero(self):
        for name in HELP_MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(
                    hasattr(module, "parse_args"), f"{name} has no parse_args"
                )
                # Scripts define parse_args() with no argv parameter
                # (argparse reads sys.argv); patch sys.argv for --help.
                argv = sys.argv
                sys.argv = [name, "--help"]
                try:
                    with redirect_stdout(io.StringIO()), self.assertRaises(
                        SystemExit
                    ) as ctx:
                        module.parse_args()
                finally:
                    sys.argv = argv
                self.assertEqual(ctx.exception.code, 0)


class TestStudyCommandHelp(unittest.TestCase):
    """Runnable study modules answer --help without DATA or checkpoints."""

    def test_help_exits_zero(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name in STUDY_COMMAND_MODULES:
            with self.subTest(module=name):
                result = subprocess.run(
                    [sys.executable, "-m", name, "--help"],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage", result.stdout.lower())


class TestStudyHelpers(unittest.TestCase):
    """Pure helpers used by the remaining analysis studies."""

    def test_doy_and_season_masks(self):
        from analysis.perturbation._perturb import doy_of_positions, season_mask

        doy = doy_of_positions(365)
        self.assertEqual(int(doy[0]), 1)
        self.assertEqual(int(doy[-1]), 365)
        self.assertEqual(int(season_mask(365, [(152, 243)]).sum()), 92)
        self.assertEqual(int(season_mask(365, [(335, 59)]).sum()), 90)
        self.assertTrue(season_mask(365, None).all())

    def test_perturbation_apply(self):
        import torch

        from analysis.perturbation._perturb import ChannelPerturbation, Perturbation
        from phenonn.utils.config import ALL_FEATURES

        stats = {"Tmean": {"mean": 10.0, "std": 2.0}}
        x = torch.zeros(1, len(ALL_FEATURES), 3)
        perturbation = Perturbation(
            [ChannelPerturbation("Tmean", add=5.0)],
            stats,
        )
        out = perturbation.apply(x)
        channel = ALL_FEATURES.index("Tmean")
        np.testing.assert_allclose(out[0, channel, :].numpy(), 2.5)
        self.assertTrue(torch.equal(out[0, 0, :], x[0, 0, :]))

    def test_aggregate_stats_zero_baseline(self):
        from analysis.perturbation.aggregate_perturbation import _stats

        df = pd.DataFrame(
            {
                "delta": [0.1, -0.1],
                "lai_base": [0.0, 0.0],
                "doy": [1, 200],
            }
        )
        stats = _stats(df)
        self.assertTrue(np.isnan(stats["rel_change_%"]))
        self.assertAlmostEqual(stats["mean_dLAI"], 0.0)

    def test_r2_rmse(self):
        from analysis.pft_linearity.compare_mixing import _r2_rmse

        values = np.array([1.0, 2.0, 3.0])
        r2, rmse = _r2_rmse(values, values)
        self.assertEqual(r2, 1.0)
        self.assertEqual(rmse, 0.0)

    def test_year_split_helpers(self):
        from analysis.year_generalization.run_year_split import (
            _rand_split,
            resolve_split,
        )

        years = list(range(1992, 2020))
        train, val = _rand_split(years, 1001)
        self.assertEqual(len(train), 18)
        self.assertEqual(len(val), 10)
        self.assertEqual(set(train + val), set(years))
        self.assertFalse(set(train) & set(val))

        train, val = resolve_split("chrono", "", years)
        self.assertEqual(train, years[:18])
        self.assertEqual(val, years[18:])

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"years_cold_to_hot": years}, f)
            path = f.name
        try:
            train, val = resolve_split("coldhot", path, years)
        finally:
            os.unlink(path)
        self.assertEqual(len(train), 18)
        self.assertEqual(len(val), 10)
        self.assertEqual(set(train + val), set(years))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisImports))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisHelp))
    suite.addTests(loader.loadTestsFromTestCase(TestStudyCommandHelp))
    suite.addTests(loader.loadTestsFromTestCase(TestStudyHelpers))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
