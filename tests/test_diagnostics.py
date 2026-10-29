# Copyright 2026 CNRS
# Authors: Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import Mock
import sys

# Import the module to test
from phenonn.utils.diagnostics import (
    plot_metric_histories,
    plot_loss_histories,
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
)


class TestDiagnostics(unittest.TestCase):
    """Unit tests for diagnostics plotting functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create tests_plots directory
        self.plots_dir = os.path.join(os.path.dirname(__file__), "tests_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Create temporary directory for stats test
        self.temp_dir = os.path.join(self.plots_dir, "temp_stats")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create dummy metric histories
        self.train_history = {
            "VAR1_MAE": [12.5, 10.2, 8.1],
            "VAR1_MSE": [220.0, 180.0, 140.0],
            "VAR1_R2": [0.7, 0.8, 0.85],
            "VAR2_MAE": [15.0, 12.0, 9.5],
            "VAR2_MSE": [280.0, 230.0, 190.0],
            "VAR2_R2": [0.6, 0.7, 0.75],
            "VAR3_MAE": [13.8, 11.1, 8.9],
            "VAR3_MSE": [260.0, 215.0, 175.0],
            "VAR3_R2": [0.65, 0.74, 0.80],
        }

        self.valid_history = {
            "VAR1_MAE": [13.8, 11.4, 9.0],
            "VAR1_MSE": [250.0, 210.0, 170.0],
            "VAR1_R2": [0.65, 0.75, 0.8],
            "VAR2_MAE": [16.8, 13.7, 10.8],
            "VAR2_MSE": [320.0, 270.0, 220.0],
            "VAR2_R2": [0.55, 0.65, 0.7],
            "VAR3_MAE": [15.2, 12.6, 10.1],
            "VAR3_MSE": [295.0, 245.0, 205.0],
            "VAR3_R2": [0.60, 0.69, 0.76],
        }

        # Create dummy prediction and observation data
        np.random.seed(42)
        n_samples = 1000
        self.obs = np.random.uniform(0, 10, n_samples)
        self.pred = self.obs + np.random.normal(0, 0.5, n_samples)  # Add some noise

        # Create data with NaN values for testing
        self.obs_with_nan = self.obs.copy()
        self.pred_with_nan = self.pred.copy()
        self.obs_with_nan[0] = np.nan
        self.pred_with_nan[1] = np.nan

        # Create small dataset for scatter fallback
        self.obs_small = np.array([1, 2, 3, 4, 5])
        self.pred_small = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        # Create logger mock
        self.logger = Mock()

        # Create dummy GCC data for testing
        self.sites = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E", "Site_F"]

        # Create dataframe with multiple sites, years, and DOYs
        gcc_data = []
        for site_idx, site in enumerate(self.sites):
            # Different R² levels for each site
            if site in ["Site_A", "Site_B"]:
                noise_level = 0.1  # Low R² (high noise)
            elif site in ["Site_C", "Site_D"]:
                noise_level = 0.05  # Medium R²
            else:
                noise_level = 0.01  # High R² (low noise)

            for year in [2020, 2021, 2022]:
                for doy in range(1, 366, 10):  # Every 10 days
                    # True LAI values (sinusoidal pattern)
                    true_lai = 3 + 2 * np.sin(2 * np.pi * (doy - 100) / 365)
                    # Add noise based on site performance
                    pred_lai = true_lai + np.random.normal(0, noise_level)
                    obs_lai = true_lai + np.random.normal(0, noise_level * 0.8)

                    gcc_data.append(
                        {
                            "site": site,
                            "year": year,
                            "day_index": doy,
                            "lai_pred": max(0, pred_lai),
                            "lai_obs": max(0, obs_lai),
                        }
                    )

        self.gcc_df = pd.DataFrame(gcc_data)

        self.create_dummy_site_files()

    def create_dummy_site_files(self):
        """Create dummy CSV files for testing feature distributions."""
        self.temp_site_dir = os.path.join(self.plots_dir, "temp_site_data")
        os.makedirs(self.temp_site_dir, exist_ok=True)

        self.site_files = []
        np.random.seed(42)

        # Define required columns
        DYNAMIC_FEATURES = ["tmin", "tmax", "daylength", "vpd", "prcp", "srad", "swe"]
        STATIC_FEATURES = ["mat", "map"]
        required_cols = DYNAMIC_FEATURES + STATIC_FEATURES + ["year", "doy", "LAI"]

        # Create 10 dummy sites
        for site_id in range(10):
            data = {col: [] for col in required_cols}

            # Site-specific static values
            mat = np.random.uniform(-10, 25)  # mean annual temperature
            map_val = np.random.uniform(200, 2000)  # mean annual precipitation

            for year in [2020, 2021]:
                for doy in range(1, 366, 10):  # Every 10 days
                    # Temperature with seasonal pattern
                    t_mean = mat + 15 * np.sin(2 * np.pi * (doy - 100) / 365)
                    t_range = 10 + 5 * np.sin(2 * np.pi * (doy - 100) / 365)
                    tmin_val = t_mean - t_range / 2 + np.random.normal(0, 1)
                    tmax_val = t_mean + t_range / 2 + np.random.normal(0, 1)

                    # Other dynamic features
                    daylength = (
                        12
                        + 6 * np.sin(2 * np.pi * (doy - 80) / 365)
                        + np.random.normal(0, 0.5)
                    )
                    vpd_val = max(
                        0,
                        0.5
                        + 0.3 * np.sin(2 * np.pi * (doy - 100) / 365)
                        + np.random.normal(0, 0.1),
                    )
                    prcp_val = max(0, np.random.exponential(map_val / 365, 1)[0])
                    srad_val = max(
                        0,
                        200
                        + 150 * np.sin(2 * np.pi * (doy - 100) / 365)
                        + np.random.normal(0, 20),
                    )
                    swe_val = (
                        max(
                            0,
                            100 * np.exp(-(((doy - 30) / 50) ** 2))
                            + np.random.normal(0, 5),
                        )
                        if t_mean < 0
                        else 0
                    )

                    # LAI target
                    lai = max(
                        0,
                        3
                        + 2 * np.sin(2 * np.pi * (doy - 100) / 365)
                        + np.random.normal(0, 0.3),
                    )

                    # Append to data dictionary
                    data["year"].append(year)
                    data["doy"].append(doy)
                    data["tmin"].append(tmin_val)
                    data["tmax"].append(tmax_val)
                    data["daylength"].append(daylength)
                    data["vpd"].append(vpd_val)
                    data["prcp"].append(prcp_val)
                    data["srad"].append(srad_val)
                    data["swe"].append(swe_val)
                    data["mat"].append(mat)
                    data["map"].append(map_val)
                    data["LAI"].append(lai)

            df = pd.DataFrame(data)

            file_path = os.path.join(self.temp_site_dir, f"site_{site_id}.csv")
            df.to_csv(file_path, index=False)
            self.site_files.append(file_path)

        # Create a file with missing data for testing
        df_bad = pd.DataFrame(
            {
                "year": [2020, 2020],
                "doy": [1, 100],
                "tmin": [5, np.nan],
                "tmax": [15, 20],
                "daylength": [12, 13],
                "vpd": [0.5, 0.6],
                "prcp": [0, np.nan],
                "srad": [200, 210],
                "swe": [0, 0],
                "mat": [10, 10],
                "map": [800, 800],
                "LAI": [1.0, np.nan],
            }
        )

        bad_file = os.path.join(self.temp_site_dir, "bad_site.csv")
        df_bad.to_csv(bad_file, index=False)
        self.site_files.append(bad_file)

    def tearDown(self):
        """Clean up temporary files (keep tests_plots directory)."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if hasattr(self, "temp_site_dir") and os.path.exists(self.temp_site_dir):
            shutil.rmtree(self.temp_site_dir)

    # ------------------------------------------------------------------------
    # plot_metric_histories tests
    # ------------------------------------------------------------------------

    def test_plot_metric_histories(self):
        """Test metric histories plot generation."""
        output_file = os.path.join(self.plots_dir, "test_metrics.png")

        plot_metric_histories(
            self.train_history,
            self.valid_history,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    def test_plot_metric_histories_with_log_metrics(self):
        """Test metric histories plot with custom log_metrics parameter."""
        output_file = os.path.join(self.plots_dir, "test_metrics_log.png")

        plot_metric_histories(
            self.train_history,
            self.valid_history,
            filename=output_file,
            logger=self.logger,
            log_metrics=["VAR1_MAE", "VAR2_MSE"],
            cols=2,
        )

        self.assertTrue(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_loss_histories tests
    # ------------------------------------------------------------------------

    def test_plot_loss_histories(self):
        """Test loss histories plot generation."""
        output_file = os.path.join(self.plots_dir, "test_loss.png")
        train_loss = [0.5, 0.4, 0.3, 0.25]
        valid_loss = [0.55, 0.45, 0.35, 0.3]

        plot_loss_histories(
            train_loss,
            valid_loss,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))

    def test_plot_loss_histories_log_scale_false(self):
        """Test loss histories plot with log_scale=False."""
        output_file = os.path.join(self.plots_dir, "test_loss_linear.png")
        train_loss = [0.5, 0.4, 0.3, 0.25]
        valid_loss = [0.55, 0.45, 0.35, 0.3]

        plot_loss_histories(
            train_loss,
            valid_loss,
            filename=output_file,
            logger=self.logger,
            log_scale=False,
        )

        self.assertTrue(os.path.exists(output_file))

    # ------------------------------------------------------------------------
    # plot_pred_vs_obs tests
    # ------------------------------------------------------------------------

    def test_plot_pred_vs_obs_hexbin(self):
        """Test pred vs obs plot with hexbin (default)."""
        output_file = os.path.join(self.plots_dir, "test_pred_vs_obs_hexbin.png")

        metrics = plot_pred_vs_obs(
            self.pred,
            self.obs,
            filename=output_file,
            logger=self.logger,
            title="Test: Predicted vs Observed",
        )

        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(metrics, dict)
        self.assertIn("r2", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("bias", metrics)
        self.assertIn("n", metrics)
        self.assertEqual(metrics["n"], len(self.obs))

    def test_plot_pred_vs_obs_scatter(self):
        """Test pred vs obs plot with scatter (hexbin=False)."""
        output_file = os.path.join(self.plots_dir, "test_pred_vs_obs_scatter.png")

        metrics = plot_pred_vs_obs(
            self.pred_small,
            self.obs_small,
            filename=output_file,
            logger=self.logger,
            hexbin=False,
            title="Test: Scatter Plot",
        )

        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(metrics, dict)

    def test_plot_pred_vs_obs_with_nan(self):
        """Test pred vs obs plot handles NaN values correctly."""
        output_file = os.path.join(self.plots_dir, "test_pred_vs_obs_nan.png")

        metrics = plot_pred_vs_obs(
            self.pred_with_nan,
            self.obs_with_nan,
            filename=output_file,
            logger=self.logger,
        )

        self.assertTrue(os.path.exists(output_file))
        # Metrics should be computed on finite values only
        self.assertLess(metrics["n"], len(self.obs_with_nan))

    def test_plot_pred_vs_obs_custom_gridsize(self):
        """Test pred vs obs plot with custom gridsize."""
        output_file = os.path.join(self.plots_dir, "test_pred_vs_obs_gridsize.png")

        metrics = plot_pred_vs_obs(
            self.pred,
            self.obs,
            filename=output_file,
            logger=self.logger,
            gridsize=20,
        )

        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(metrics, dict)

    # ------------------------------------------------------------------------
    # plot_gcc_curves tests
    # ------------------------------------------------------------------------

    def test_plot_gcc_curves(self):
        """Test GCC curves plot generation."""
        output_file = os.path.join(self.plots_dir, "test_gcc_curves.png")

        result = plot_gcc_curves(
            self.gcc_df,
            filename=output_file,
            logger=self.logger,
            seed=42,
        )

        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)  # Returns 3 sites

        # Each value is (r2, mae, mse); check the R² bounds.
        for site, metrics in result.items():
            r2, mae, mse = metrics
            self.assertIsInstance(r2, float)
            self.assertGreaterEqual(r2, 0)
            self.assertLessEqual(r2, 1)

    def test_plot_gcc_curves_custom_columns(self):
        """Test GCC curves plot with custom column names."""
        # Rename columns
        df_custom = self.gcc_df.rename(
            columns={
                "site": "location",
                "year": "yr",
                "day_index": "doy",
                "lai_pred": "pred",
                "lai_obs": "obs",
            }
        )

        output_file = os.path.join(self.plots_dir, "test_gcc_curves_custom.png")

        result = plot_gcc_curves(
            df_custom,
            filename=output_file,
            logger=self.logger,
            site_col="location",
            year_col="yr",
            doy_col="doy",
            pred_col="pred",
            obs_col="obs",
            seed=42,
        )

        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)

    def test_plot_gcc_curves_all(self):
        """Test GCC curves all plot generation."""
        output_file = os.path.join(self.plots_dir, "test_gcc_curves_all.png")

        result = plot_gcc_curves_all(
            self.gcc_df,
            filename=output_file,
            logger=self.logger,
            cols=3,
        )

        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(result, dict)
        # Should return all 6 sites
        self.assertEqual(len(result), 6)

        # Each value is (r2, mae, mse); check the R² bounds.
        for site, metrics in result.items():
            r2, mae, mse = metrics
            self.assertIsInstance(r2, float)
            self.assertGreaterEqual(r2, 0)
            self.assertLessEqual(r2, 1)

        # Check that sites are sorted by R² descending (best first)
        r2_values = [m[0] for m in result.values()]
        self.assertGreaterEqual(r2_values[0], r2_values[-1])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDiagnostics))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
