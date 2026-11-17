# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tests for the optional raw feature cache and the 1992 pre-filter."""

import glob
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import xarray as xr

from phenonn.data.feature_cache import (
    FeatureCache,
    feature_key,
    list_entries,
    remove_entry,
)
from phenonn.data.lai_dataset import RamLAIDataset, cached_climatology_lookup
from phenonn.utils.config import DYNAMIC_FEATURES, N_DEKAD_YEAR, N_PFT, PFT_COLS

SITES = ["pix_0000_00000", "pix_0000_00001", "pix_0000_00002"]
FEAT, TGT, PFT = "feat", "tgt", "pft"


def _n_days(year):
    return 366 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 365


def _write_year(root, year, rng, sites=SITES):
    """Write the three pixelset files of one year under ``root``."""
    feat_dir = os.path.join(root, FEAT)
    tgt_dir = os.path.join(root, TGT)
    pft_dir = os.path.join(root, PFT)
    for path in (feat_dir, tgt_dir, pft_dir):
        os.makedirs(path, exist_ok=True)

    n_t = _n_days(year)
    site_ids = np.asarray(sites)
    feats = {
        name: (("time", "site"), rng.random((n_t, len(sites))).astype("float32"))
        for name in DYNAMIC_FEATURES
    }
    xr.Dataset(
        feats,
        coords={
            "time": np.arange(n_t),
            "site": np.arange(len(sites)),
            "site_id": ("site", site_ids),
        },
    ).to_netcdf(os.path.join(feat_dir, f"ERA5_daily_pixelset_{year}.nc"))

    xr.Dataset(
        {
            "LAI": (
                ("dekad", "site"),
                rng.random((N_DEKAD_YEAR, len(sites))).astype("float32"),
            ),
        },
        coords={
            "dekad": np.arange(N_DEKAD_YEAR),
            "site": np.arange(len(sites)),
            "site_id": ("site", site_ids),
        },
    ).to_netcdf(os.path.join(tgt_dir, f"LAI_dekadal_{year}.nc"))

    xr.Dataset(
        {
            "pft_frac": (
                ("pft", "site"),
                rng.random((N_PFT, len(sites))).astype("float32"),
            ),
        },
        coords={
            "pft": np.arange(N_PFT),
            "site": np.arange(len(sites)),
            "site_id": ("site", site_ids),
        },
    ).to_netcdf(os.path.join(pft_dir, f"PFTmap_{year}.nc"))


def _dataset(root, years, cache_dir="", mode="off", norm_stats=None):
    return RamLAIDataset(
        features_dir=os.path.join(root, FEAT),
        target_dir=os.path.join(root, TGT),
        pft_dir=os.path.join(root, PFT),
        years=years,
        site_ids=SITES,
        seq_length=720,
        norm_stats=norm_stats,
        verbose=False,
        cache_dir=cache_dir,
        feature_cache=mode,
    )


class TestFeatureCache(unittest.TestCase):
    """Round-trip, invalidation and mutation safety."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.cache = os.path.join(self.root, "cache")
        rng = np.random.default_rng(0)
        _write_year(self.root, 2000, rng)
        _write_year(self.root, 2001, rng)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_read_is_identical(self):
        ds_write = _dataset(self.root, [2001], self.cache, "write")
        raw = {y: ds_write._dyn[y].copy() for y in ds_write._dyn}
        self.assertEqual(set(raw), {2000, 2001})
        self.assertIsNotNone(ds_write.cache_key)

        ds_read = _dataset(self.root, [2001], self.cache, "read")
        self.assertEqual(set(ds_read._dyn), set(raw))
        for year, arr in raw.items():
            np.testing.assert_array_equal(ds_read._dyn[year], arr)

    def test_read_matches_netcdf_with_normalisation(self):
        stats = {name: {"mean": 1.0, "std": 2.0} for name in DYNAMIC_FEATURES}
        stats.update({name: {"mean": 0.2, "std": 0.1} for name in PFT_COLS})

        ds_write = _dataset(self.root, [2001], self.cache, "write")
        raw = {y: ds_write._dyn[y].copy() for y in ds_write._dyn}

        ds_cache = _dataset(self.root, [2001], self.cache, "read", stats)
        ds_disk = _dataset(self.root, [2001], "", "off", stats)
        for year in raw:
            np.testing.assert_array_equal(ds_cache._dyn[year], ds_disk._dyn[year])

        # Normalisation must not have touched the cached raw values.
        entry = FeatureCache(self.cache, ds_write.cache_key)
        for year, arr in raw.items():
            np.testing.assert_array_equal(entry.read_year(year), arr)

    def test_key_changes_with_source_mtime(self):
        path = os.path.join(self.root, FEAT, "ERA5_daily_pixelset_2000.nc")
        first = feature_key(os.path.join(self.root, FEAT), [2000, 2001], SITES)
        st = os.stat(path)
        os.utime(path, (st.st_atime + 10, st.st_mtime + 10))
        second = feature_key(os.path.join(self.root, FEAT), [2000, 2001], SITES)
        self.assertNotEqual(first, second)

    def test_key_changes_with_site_list(self):
        feat = os.path.join(self.root, FEAT)
        self.assertNotEqual(
            feature_key(feat, [2000], SITES),
            feature_key(feat, [2000], SITES[:2]),
        )

    def test_partial_entry_is_a_miss(self):
        ds = _dataset(self.root, [2001], self.cache, "write")
        entry = FeatureCache(self.cache, ds.cache_key)
        self.assertTrue(entry.exists())
        os.remove(os.path.join(entry.path, "f2001.npy"))
        self.assertFalse(entry.exists())

        # A read falls back to the NetCDF files instead of failing.
        ds_read = _dataset(self.root, [2001], self.cache, "read")
        self.assertEqual(set(ds_read._dyn), {2000, 2001})

    def test_list_and_remove(self):
        _dataset(self.root, [2001], self.cache, "write")
        entries = list_entries(self.cache)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["years"], [2000, 2001])
        self.assertGreater(entries[0]["size"], 0)
        removed = remove_entry(entries[0]["key"], self.cache)
        self.assertEqual(len(removed), 1)
        self.assertEqual(list_entries(self.cache), [])

    def test_target_year_without_history_is_dropped_before_loading(self):
        ds = _dataset(self.root, [2000, 2001], self.cache, "off")
        self.assertEqual(set(ds._dyn), {2000, 2001})
        self.assertNotIn(1999, ds._dyn)
        self.assertNotIn(2000, ds._targ)
        self.assertIn(2001, ds._targ)
        self.assertEqual(len(ds), len(SITES))


class TestClimatologyCache(unittest.TestCase):
    """The small per-site climatology cache."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.cache = os.path.join(self.root, "cache")
        rng = np.random.default_rng(1)
        _write_year(self.root, 2000, rng)
        _write_year(self.root, 2001, rng)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_uses_cache(self):
        target = os.path.join(self.root, TGT)
        first = cached_climatology_lookup(
            target, [2000, 2001], SITES, self.cache, mode="auto"
        )
        self.assertEqual(set(first), set(SITES))
        files = glob.glob(os.path.join(self.cache, "climatology", "*.npz"))
        self.assertEqual(len(files), 1)
        with mock.patch(
            "phenonn.data.lai_dataset.compute_climatology_lookup",
            side_effect=AssertionError("the cache must avoid a recompute"),
        ):
            second = cached_climatology_lookup(
                target, [2000, 2001], SITES, self.cache, mode="read"
            )
        for site in SITES:
            np.testing.assert_allclose(second[site], first[site])

    def test_off_mode_recomputes(self):
        target = os.path.join(self.root, TGT)
        cached_climatology_lookup(target, [2000, 2001], SITES, self.cache, "auto")
        fake = {site: np.zeros(N_DEKAD_YEAR, dtype=np.float32) for site in SITES}
        with mock.patch(
            "phenonn.data.lai_dataset.compute_climatology_lookup",
            return_value=fake,
        ) as patched:
            other = cached_climatology_lookup(
                target, [2000, 2001], SITES, self.cache, mode="off"
            )
        patched.assert_called_once()
        np.testing.assert_array_equal(other[SITES[0]], fake[SITES[0]])


class TestMetricClimatology(unittest.TestCase):
    """The metric climatology stored in every checkpoint."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        rng = np.random.default_rng(2)
        _write_year(self.root, 2000, rng)
        _write_year(self.root, 2001, rng)

    def tearDown(self):
        self._tmp.cleanup()

    def _args(self, **extra):
        base = SimpleNamespace(
            clim_years="2000-2001",
            anomaly_mode=False,
            clim_target_dir="",
            target_dir=os.path.join(self.root, TGT),
            cache_dir="",
            cache_climatology=False,
        )
        for key, value in extra.items():
            setattr(base, key, value)
        return base

    def test_reuses_anomaly_table_for_validation_sites(self):
        from phenonn.training.train_full_ram import compute_metric_climatology

        anomaly = {
            site: np.ones(N_DEKAD_YEAR, dtype=np.float32)
            for site in SITES + ["pix_extra"]
        }
        metric, years = compute_metric_climatology(
            self._args(anomaly_mode=True),
            SITES[:2],
            anomaly,
            [2000],
            mock.Mock(),
        )
        self.assertEqual(set(metric), set(SITES[:2]))
        self.assertEqual(years, [2000, 2001])

    def test_builds_and_returns_the_window(self):
        from phenonn.training.train_full_ram import compute_metric_climatology

        metric, years = compute_metric_climatology(
            self._args(), SITES, None, [2000], mock.Mock()
        )
        self.assertEqual(set(metric), set(SITES))
        self.assertEqual(years, [2000, 2001])


if __name__ == "__main__":
    unittest.main()
