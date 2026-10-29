# Copyright 2026 CNRS
# Authors: Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Kept-surface CLI tests: `phenonn` entry point dispatch.

Only `--version` / `--help` flows — no training is launched.
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*argv):
    return subprocess.run(
        [sys.executable, "-m", "phenonn", *argv],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


class TestCli(unittest.TestCase):
    """Unit tests for phenonn.cli dispatch."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def test_version(self):
        """--version exits 0 and prints the version."""
        res = _run("--version")
        self.assertEqual(res.returncode, 0)
        self.assertIn("phenonn", res.stdout.lower())

    def test_help(self):
        """--help exits 0 and lists train/predict."""
        res = _run("--help")
        self.assertEqual(res.returncode, 0)
        self.assertIn("train", res.stdout)
        self.assertIn("predict", res.stdout)

    def test_unknown_command_fails(self):
        """Unknown commands exit non-zero."""
        res = _run("does_not_exist")
        self.assertNotEqual(res.returncode, 0)

    def test_train_help(self):
        """train --help forwards to the trainer argparse."""
        res = _run("train", "--help")
        self.assertEqual(res.returncode, 0)

    def test_predict_help(self):
        """predict --help forwards to the predictor argparse."""
        res = _run("predict", "--help")
        self.assertEqual(res.returncode, 0)

    def test_predict_n_curves_default(self):
        """Prediction caps curve plots at 100 unless the user overrides it."""
        from phenonn.prediction.args import parse_args

        args = parse_args(["--checkpoint", "dummy"])
        self.assertEqual(args.n_curves, 100)
        self.assertTrue(getattr(args, "_n_curves_defaulted", False))

        args = parse_args(["--checkpoint", "dummy", "--n_curves", "0"])
        self.assertEqual(args.n_curves, 0)
        self.assertFalse(getattr(args, "_n_curves_defaulted", False))

    def test_import_is_lazy(self):
        """Importing phenonn must not load torch on the CLI fast path."""
        res = subprocess.run(
            [
                sys.executable,
                "-c",
                "import phenonn, sys; print('torch' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "False")

    def test_package_does_not_import_analysis(self):
        """phenonn must stay importable without the analysis scripts."""
        res = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import phenonn, sys; "
                    "print(any(m == 'analysis' or m.startswith('analysis.') "
                    "for m in sys.modules))"
                ),
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "False")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCli))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
