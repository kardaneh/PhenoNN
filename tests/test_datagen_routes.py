# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Unit tests for the datagen routes and feature contract.

These tests use tiny synthetic NetCDF files only. They never touch the real
``pixelset1`` data. The two routes covered here are:

- 0.05 degree ERA5 identity features and 0.05 degree LAI/PFT targets;
- 0.1 degree deduplicated ERA5 features plus a 0.05 -> 0.1 parent map.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xarray as xr

import phenonn.datagen.config as dg_config
import phenonn.utils.config as shared_config
from phenonn.data.lai_dataset import RamLAIDataset, load_parent01_map
from phenonn.datagen.add_soil_moisture_proxy import (
    WINDOW,
    _process_year,
    _soil_moisture_index,
    _weights,
)
from phenonn.datagen.create_stats_file import (
    RunningStats,
    _scan_features_year,
    _scan_targets_year,
)


class TestSoilMoistureProxy(unittest.TestCase):
    def test_weights_are_causal_and_decay_linearly(self):
        w = _weights()
        self.assertEqual(w.shape, (WINDOW + 1,))
        self.assertAlmostEqual(float(w[0]), 1.0)
        self.assertAlmostEqual(float(w[-1]), 0.0)
        self.assertTrue(np.all(np.diff(w) < 0.0))

    def test_formula_and_zero_padded_warmup(self):
        precip = np.arange(1, 13, dtype=np.float64).reshape(-1, 1)
        smi = _soil_moisture_index(precip)
        w = _weights()

        self.assertEqual(smi.shape, precip.shape)
        self.assertAlmostEqual(float(smi[0, 0]), float(precip[0, 0] / 31.0))
        for t in range(precip.shape[0]):
            expected = 0.0
            for i in range(WINDOW + 1):
                if t - i >= 0:
                    expected += w[i] * precip[t - i, 0]
            self.assertAlmostEqual(float(smi[t, 0]), expected / 31.0)

    def test_process_year_writes_smi(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            in_dir.mkdir()
            src = in_dir / "ERA5_daily_pixelset_2000.nc"
            xr.Dataset(
                {
                    "tp_sum": (
                        ("time", "site"),
                        np.arange(6, dtype=np.float32).reshape(3, 2),
                    ),
                    "Tmin": (
                        ("time", "site"),
                        np.zeros((3, 2), dtype=np.float32),
                    ),
                },
                coords={"time": np.arange(3), "site": ["E0000_0000", "E0000_0001"]},
            ).to_netcdf(src, engine="netcdf4")

            _process_year(2000, in_dir, out_dir, complevel=1)

            with xr.open_dataset(
                out_dir / "ERA5_daily_pixelset_2000.nc", engine="netcdf4"
            ) as got:
                self.assertIn("SMI", got)
                self.assertEqual(got["SMI"].shape, (3, 2))
                self.assertAlmostEqual(float(got["SMI"].values[0, 0]), 0.0)
                self.assertIn("formula", got["SMI"].attrs)


class TestParentMapRoute(unittest.TestCase):
    def test_parent_map_deduplicates_and_covers_children(self):
        from phenonn.datagen.make_selected_pixels_01 import main

        rows = np.array([0, 1, 2, 3], dtype=np.int32)
        cols = np.array([0, 1, 2, 3], dtype=np.int32)
        site_ids = np.array(
            [f"pix_{r:04d}_{c:05d}" for r, c in zip(rows, cols)], dtype=object
        )

        expected = {}
        for site_id, row, col in zip(site_ids, rows, cols):
            lat_idx = (int(row) + 1) // 2
            lon_m180 = ((int(col) + 1) // 2) % 3600
            lon_idx = (lon_m180 + 1800) % 3600
            expected[str(site_id)] = f"E{lat_idx:04d}_{lon_idx:04d}"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sel_path = tmp_path / "selected_pixels005_1.nc"
            out_path = tmp_path / "selected_pixels01_1.nc"
            ds = xr.Dataset(
                {
                    "lat_idx": ("site", rows),
                    "lon_idx": ("site", cols),
                    "site_id": ("site", site_ids),
                }
            )
            ds.to_netcdf(sel_path, engine="netcdf4")

            argv = [
                "make_selected_pixels_01",
                "--selected_pixels",
                str(sel_path),
                "--output",
                str(out_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            self.assertTrue(out_path.exists())
            mapping = load_parent01_map(str(out_path))
            self.assertEqual(mapping, expected)
            self.assertEqual(len(set(mapping.values())), 3)

            with xr.open_dataset(out_path, engine="netcdf4") as got:
                self.assertEqual(got.sizes["site"], 3)
                self.assertEqual(got.sizes["site05"], 4)
                self.assertTrue(np.all(got["parent_site"].values >= 0))
                self.assertTrue(np.all(got["parent_site"].values < 3))

    def test_dataset_rejects_parent_map_with_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "parent_map matched none"):
                RamLAIDataset(
                    features_dir=str(tmp_path),
                    target_dir=str(tmp_path),
                    pft_dir=str(tmp_path),
                    years=[2000],
                    site_ids=["pix_0000_00000"],
                    parent_map={"some_other_site": "E0000_0000"},
                    verbose=False,
                )


class TestStatsAcrossRoutes(unittest.TestCase):
    def test_feature_and_target_stats_do_not_require_matching_site_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feat_path = tmp_path / "ERA5_daily_pixelset_2000.nc"
            tgt_path = tmp_path / "LAI_dekadal_2000.nc"

            xr.Dataset(
                {
                    "Tmin": (
                        ("time", "site"),
                        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                    ),
                    "SMI": (
                        ("time", "site"),
                        np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]),
                    ),
                },
                coords={
                    "time": np.arange(3),
                    "site": ["E0000_0000", "E0000_0001"],
                },
            ).to_netcdf(feat_path, engine="netcdf4")

            xr.Dataset(
                {"LAI": (("dekad", "site"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
                coords={
                    "dekad": np.arange(2),
                    "site": ["pix_0000_00000", "pix_0000_00001"],
                },
            ).to_netcdf(tgt_path, engine="netcdf4")

            feats = {"Tmin": RunningStats(), "SMI": RunningStats()}
            _scan_features_year(feat_path, feats, n_sites=0, seed=42)
            lai = RunningStats()
            _scan_targets_year(tgt_path, lai, n_sites=0, seed=42)

            self.assertEqual(feats["Tmin"].n, 6)
            self.assertEqual(feats["SMI"].n, 6)
            self.assertEqual(lai.n, 4)
            self.assertAlmostEqual(feats["Tmin"].mean_std()["mean"], 3.5)
            self.assertAlmostEqual(lai.mean_std()["mean"], 2.5)


class TestFeatureContract(unittest.TestCase):
    def test_report_contract_is_28_channels(self):
        self.assertEqual(dg_config.FEATURE_CHANNELS, 28)
        self.assertEqual(dg_config.PFT_START, 13)
        self.assertEqual(len(dg_config.DYNAMIC_FEATURES), 12)
        self.assertEqual(dg_config.CYCLIC_FEATURES, [])
        self.assertEqual(dg_config.DYNAMIC_FEATURES[-2:], ["daylength", "SMI"])

    def test_datagen_and_shared_config_agree(self):
        names = [
            "N_PFT",
            "N_DEKAD_YEAR",
            "METEO_BASE",
            "AUGMENTED_FEATURES",
            "DYNAMIC_FEATURES",
            "CYCLIC_FEATURES",
            "CO2_FEATURES",
            "PFT_COLS",
            "ALL_FEATURES",
            "FEATURE_CHANNELS",
            "PFT_START",
            "PFT_NAMES",
            "FEATURES_FNAME",
            "TARGETS_FNAME",
            "PFT_FNAME",
            "VALID_FNAME",
            "SELECTED_PIXELS_FNAME",
        ]
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(dg_config, name), getattr(shared_config, name))


if __name__ == "__main__":
    unittest.main()
