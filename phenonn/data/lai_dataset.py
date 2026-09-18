"""
PyTorch Dataset for LAI prediction on NetCDF cubes.

This is the new equivalent of LaiNN/phenocam/dataset_big.py, rewritten to
read from xarray-backed NetCDF files instead of per-year CSVs.

Sources read at construction time (all on the ERA5-Land 0.1° grid):

    features_dir/ERA5_features_{YYYY}.nc    dims (time=365, lat, lon),
                                            20 daily variables
    target_dir/LAI_dekadal_{YYYY}.nc        dims (dekad=36, lat, lon),
                                            one variable "LAI"
    pft_dir/PFTmap_{YYYY}.nc                dims (pft=15, lat, lon)
                                            OR 15 pft{k}_frac vars (lat, lon)

A "site" is identified by `pix_LLLL_OOOOO` where LLLL = latitude index and
OOOOO = longitude index in the ERA5-Land 0.1° grid (zero-padded). This keeps
diagnostic scripts written for LaiNN backward-compatible.

Each sample
-----------
    site_id : pix_LLLL_OOOOO
    year    : int (target year Y)
        features : (n_features, seq_length=720) float32
                   covers the last seq_length days of Y, requires Y-1 file
                   for the seq_length history to fit.
        target   : (1, 36) float32
                   the 36 dekadal LAI observations of year Y.

Loading strategy
----------------
EAGER. At construction time the full feature matrix and target vector are
materialised in RAM for every (site, year) pair. `__getitem__` then just
returns array slices wrapped in tensors. With 2000 sites × 5 years × 36
features × 720 days × 4 bytes ≈ 0.7 GB — well within budget for a single
epoch's working set.

Anomaly mode
------------
When `anomaly_clim` is provided, the target is replaced by
`LAI − climatology(site, dekad)` and LAI z-scoring is disabled. Symmetric
behaviour to the dataset_big.py implementation it replaces.
"""

import datetime
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from phenonn.utils.config import (
    ALL_FEATURES, CO2_FEATURES, CYCLIC_FEATURES,
    DYNAMIC_FEATURES, FEATURES_FNAME, LOG_TRANSFORM_FEATURES,
    METEO_BASE, N_DEKAD_YEAR, N_PFT, PFT_COLS, PFT_FNAME,
    SEQ_LENGTH_DEFAULT, TARGETS_FNAME, VALID_FNAME,
    add_co2_features,
)
from phenonn.utils.wrappers import _OBS_POSITIONS

# Daily-LAI mode (--daily_lai): 365-day interpolated target files and the
# 0-indexed positions of the 36 dekad observations within a 365-day year.
LAI_DAILY_FNAME = "LAI_daily_{year}.nc"
N_DAYS_YEAR = 365
_OBS_POSITIONS_ARR = np.array(_OBS_POSITIONS, dtype=np.int64)


def _clim36_to_365(clim36: np.ndarray) -> np.ndarray:
    """Linear-interpolate a (36,) dekadal climatology onto the 365-day grid at
    the dekad DOY anchors (same interp as the daily target). NaN if <2 finite
    dekads; the values at the 36 dekad DOYs are preserved exactly."""
    finite = np.isfinite(clim36)
    out = np.full(N_DAYS_YEAR, np.nan, dtype=np.float32)
    if int(finite.sum()) >= 2:
        doy = (_OBS_POSITIONS_ARR + 1).astype(np.float64)      # 1-based dekad DOYs
        out[:] = np.interp(np.arange(1, N_DAYS_YEAR + 1),
                           doy[finite], clim36[finite].astype(np.float64)
                           ).astype(np.float32)
    return out


# ── Site-ID encoding ────────────────────────────────────────────────────────


def encode_site_id(lat_idx: int, lon_idx: int) -> str:
    """`pix_LLLL_OOOOO` — 4-digit lat index, 5-digit lon index."""
    return f"pix_{int(lat_idx):04d}_{int(lon_idx):05d}"


def decode_site_id(site_id: str) -> Tuple[int, int]:
    """Inverse of `encode_site_id`. Raises on a malformed id."""
    parts = site_id.split("_")
    if len(parts) != 3 or parts[0] != "pix":
        raise ValueError(f"Malformed site_id {site_id!r}; expected pix_LLLL_OOOOO")
    return int(parts[1]), int(parts[2])


def generate_site_ids_from_range(
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
) -> List[str]:
    """Every pix id in the (row_min..row_max, col_min..col_max) inclusive box."""
    r0, r1 = row_range
    c0, c1 = col_range
    return [encode_site_id(r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)]


# ── CO2 LUT (Annee_YYYY=VALUE format) ───────────────────────────────────────


def load_co2_lut(path: str) -> Dict[int, float]:
    """Parse the LaiNN-style CO2 file: one `Annee_YYYY=VALUE` per line."""
    lut: Dict[int, float] = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.lower().startswith("annee_"):
                continue
            try:
                year = int(key.split("_", 1)[1])
                lut[year] = float(value.strip())
            except ValueError:
                continue
    if not lut:
        raise ValueError(f"No `Annee_YYYY=VALUE` entries parsed from {path!r}.")
    return lut


# ── PFT loading (handles both 3-D var or 15 separate vars) ──────────────────


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Return a single DataArray of dims (pft, latitude, longitude) from a PFT
    file, supporting three layouts:

      1. ORCHIDEE PFTmap (native ESACCI / IPSL output):
         `maxvegetfrac(time_counter=1, veget=15, lat, lon)`
         → squeeze time_counter, rename veget → pft, lat/lon → latitude/longitude.

      2. 15 separate variables `pft1_frac..pft15_frac` on (lat/latitude, lon/longitude).

      3. One 3-D variable on (pft, lat/latitude, lon/longitude) — fallback
         for `pft_fraction`, `PFT`, `pft`, `pft_frac`, `fraction`.
    """
    rename = {}
    for src, dst in [("lat", "latitude"), ("lon", "longitude"),
                     ("veget", "pft")]:
        if src in ds.coords or src in ds.dims:
            rename[src] = dst
    if rename:
        ds = ds.rename(rename)

    if "maxvegetfrac" in ds.data_vars:
        da = ds["maxvegetfrac"]
        if "time_counter" in da.dims:
            da = da.isel(time_counter=0, drop=True)
        return da

    pft_named = [f"pft{k}_frac" for k in range(1, N_PFT + 1)]
    if all(v in ds.data_vars for v in pft_named):
        return xr.concat(
            [ds[v].expand_dims(pft=[k - 1])
             for k, v in enumerate(pft_named, start=1)],
            dim="pft",
        )

    for cand in ("pft_fraction", "PFT", "pft", "pft_frac", "fraction"):
        if cand in ds.data_vars and "pft" in ds[cand].dims:
            return ds[cand]

    raise ValueError(
        f"Could not detect a PFT layout in {list(ds.data_vars)}. "
        f"Expected one of: 'maxvegetfrac(time_counter, veget, lat, lon)' "
        f"(ORCHIDEE), 15 'pftK_frac' vars, or a 3-D var with a 'pft' dim."
    )


# ── Pixelset-aware per-site extraction ──────────────────────────────────────


def _site_row_index(da: xr.DataArray, site_ids: List[str]) -> np.ndarray:
    """rows[i] = row of site_ids[i] in da's `site_id` coord (-1 if absent).

    Vectorised (pandas hash join in C). The pixelset `site_id` coord is identical
    across years, so callers build this ONCE and reuse it for every year's file.
    """
    file_ids = pd.Index(np.asarray(da["site_id"].values).astype(str))
    return file_ids.get_indexer(pd.Index([str(s) for s in site_ids]))


def _read_site_vectors(
    da: xr.DataArray,
    site_ids: List[str],
    value_dim: str,
    rows: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a (n_site, size(value_dim)) float32 matrix for the requested sites
    from a pixelset LAI/PFT DataArray — a `site` dim with a `site_id` coord,
    output of `phenonn.data_creation.build_pixelset_targets`. Unknown site_ids
    come back as NaN rows. `rows` (optional) skips the id→row lookup when the
    caller has precomputed it (the pixelset site_id is identical every year).
    """
    if rows is None:
        rows = _site_row_index(da, site_ids)
    out = np.full((len(site_ids), da.sizes[value_dim]), np.nan, dtype=np.float32)
    ok = rows >= 0
    if ok.any():
        # LAI/PFT pixelsets use VERY large site-chunks (~4-5e5 sites, ~10-15 MB),
        # so a scattered per-site isel decompresses those huge chunks repeatedly
        # (minutes). These variables are small in total (LAI ~240 MB, PFT ~100 MB),
        # so read the WHOLE variable ONCE sequentially, then gather rows in memory
        # — matches build_pixelset_targets' own read-once approach, no thrashing.
        full = da.transpose("site", value_dim).values      # one sequential read
        out[ok] = full[rows[ok]].astype(np.float32)
    return out


def _open_pft_array(path: str) -> xr.DataArray:
    """
    Open a pixelset PFT file and return its `pft_frac(pft, site)` DataArray
    (output of `phenonn.data_creation.build_pixelset_targets`).

    engine="netcdf4", decode_times=False keeps the whole process on one libhdf5
    (features/targets are read with netcdf4); mixing netcdf4 + h5netcdf loads
    two bundled libhdf5 and corrupts HDF5 ("H5DSget_num_scales <0").
    """
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    return ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]


# ── Anomaly climatology helper ──────────────────────────────────────────────


def compute_climatology_lookup(
    target_dir: str,
    clim_years: List[int],
    wanted_sites: List[str],
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Per-(site, dekad) mean of LAI across `clim_years`. Returns
    `{site_id: (36,) np.float32 array}`. NaN where no observation in the
    window — consumers (subtraction) propagate it as expected.
    """
    target_dir = os.fspath(target_dir)
    wanted = sorted(set(wanted_sites))
    if not wanted:
        return {}

    # Accumulator: sum and count per (site, dekad)
    sum_ = np.zeros((len(wanted), N_DEKAD_YEAR), dtype=np.float64)
    cnt  = np.zeros((len(wanted), N_DEKAD_YEAR), dtype=np.int32)

    for y in sorted(set(clim_years)):
        path = os.path.join(target_dir, TARGETS_FNAME.format(year=y))
        if not os.path.exists(path):
            if verbose:
                print(f"  [clim] {os.path.basename(path)} missing — skipped")
            continue
        with xr.open_dataset(path) as ds:
            vals = _read_site_vectors(ds["LAI"], wanted, "dekad")  # (n_site, 36)
        finite = np.isfinite(vals)
        sum_ += np.where(finite, vals, 0.0)
        cnt  += finite.astype(np.int32)

    mean = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return {sid: mean[i] for i, sid in enumerate(wanted)}


# ── Valid-pixels mask loading ───────────────────────────────────────────────


def load_valid_mask(
    valid_dir: str, year: int,
) -> Optional[xr.DataArray]:
    """Return the bool (lat, lon) mask for `year`, or None if missing."""
    path = os.path.join(valid_dir, VALID_FNAME.format(year=year))
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path)
    if "valid" in ds.data_vars:
        return ds["valid"].astype(bool)
    name = list(ds.data_vars)[0]
    return ds[name].astype(bool)


def load_selected_pixels(selected_path: str) -> List[str]:
    """
    Return the list of `pix_LLLL_OOOOO` IDs stored in `selected_pixels.nc`
    (output of `phenonn.data_creation.select_pixels`). One unique site pool reused
    across every training year.
    """
    ds = xr.open_dataset(selected_path)
    sids = ds["site_id"].values
    ds.close()
    return [str(s) for s in sids]


def load_parent01_map(selected_pixels_01: str) -> Dict[str, str]:
    """
    Return `{site_id05: site_id01}` from a `selected_pixels_01.nc`
    (output of `phenonn.data_creation.make_selected_pixels_01`).

    Lets a dataset keyed by native 0.05° sites read its ERA5 features from the
    deduplicated 0.1° `ERA5_daily_pixelset_{Y}.nc` (site_id = "E{lat}_{lon}")
    WITHOUT materialising a 4×-larger 0.05°-indexed feature file: each 0.05°
    site is mapped to the id of the containing 0.1° ERA5 cell, and the feature
    reader looks that row up. `site05`/`parent_site` are consumed together so
    the mapping is exactly the one baked by make_selected_pixels_01.
    """
    with xr.open_dataset(selected_pixels_01) as ds:
        site_id05   = np.asarray(ds["site_id05"].values).astype(str)
        parent_site = np.asarray(ds["parent_site"].values).astype(np.int64)
        site_id01   = np.asarray(ds["site_id"].values).astype(str)   # 0.1° cells
    return {s05: site_id01[p] for s05, p in zip(site_id05, parent_site)}


def list_valid_sites(
    valid_dir: str, years: List[int],
    bbox_rows: Optional[Tuple[int, int]] = None,
    bbox_cols: Optional[Tuple[int, int]] = None,
) -> Dict[int, List[str]]:
    """
    For each year in `years`, return the list of `pix_LLLL_OOOOO` IDs whose
    mask cell is True (i.e. the pixel passed the validity criteria during
    preprocessing). Optionally restrict to a sub-grid.
    """
    out: Dict[int, List[str]] = {}
    for y in years:
        mask = load_valid_mask(valid_dir, y)
        if mask is None:
            out[y] = []
            continue
        m = mask.values.astype(bool)
        if bbox_rows or bbox_cols:
            r0, r1 = bbox_rows if bbox_rows else (0, m.shape[0] - 1)
            c0, c1 = bbox_cols if bbox_cols else (0, m.shape[1] - 1)
            keep = np.zeros_like(m)
            keep[r0:r1 + 1, c0:c1 + 1] = True
            m = m & keep
        lats, lons = np.where(m)
        out[y] = [encode_site_id(int(la), int(lo)) for la, lo in zip(lats, lons)]
    return out


# ── Dataset class ───────────────────────────────────────────────────────────


class LAIDataset(Dataset):
    """
    Eager-loading LAI Dataset on NetCDF cubes.

    Parameters
    ----------
    features_dir, target_dir, pft_dir : str
        Folders containing the per-year .nc files. Naming follows
        `phenonn.utils.config.{FEATURES_FNAME, TARGETS_FNAME, PFT_FNAME}`.
    years : list[int]
        Target years for which (site, year) samples will be built. The
        feature window of year Y also requires the file of Y-1; samples for
        which Y-1 is missing are silently dropped.
    site_ids : list[str]
        Sites to evaluate, in `pix_LLLL_OOOOO` form.
    seq_length : int
        Feature window length in days (default 720 ≈ 2 years).
    norm_stats : dict or None
        `{feat: {"mean": …, "std": …}}` produced by `create_stats_file`. When
        provided, log1p is applied to `LOG_TRANSFORM_FEATURES` and every
        feature is z-scored. LAI is z-scored too unless `normalize_lai=False`
        (or anomaly mode is on).
    anomaly_clim : dict or None
        `{site_id: (36,) array}`. When set, target becomes (LAI − clim) and
        LAI z-scoring is disabled.
    co2_lut : dict or None
        `{year: ppm}`. When `add_co2_features=True` in config, broadcast as
        the `co2` feature.
    normalize_lai : bool
        Skip LAI z-scoring when False; features remain z-scored.
    """

    def __init__(
        self,
        features_dir: str,
        target_dir: str,
        pft_dir: str,
        years: List[int],
        site_ids: List[str],
        seq_length: int = SEQ_LENGTH_DEFAULT,
        norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
        anomaly_clim: Optional[Dict[str, np.ndarray]] = None,
        co2_lut: Optional[Dict[int, float]] = None,
        normalize_lai: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.features_dir = features_dir
        self.target_dir   = target_dir
        self.pft_dir      = pft_dir
        self.seq_length   = seq_length
        self.norm_stats   = norm_stats
        self.anomaly_clim = anomaly_clim
        self.co2_lut      = co2_lut
        self.normalize_lai = normalize_lai

        target_years   = sorted(set(int(y) for y in years))
        feature_years  = sorted(set(target_years) | {y - 1 for y in target_years})

        # ── Open everything lazily ──
        # Features are now keyed by a `site` dim (flat ERA5_daily_pixelset_*.nc).
        # We build a {site_id → site_index} map per year from the file's
        # `site_id` coord so we can `.isel(site=index)` for each sample.
        ds_features: Dict[int, xr.Dataset] = {}
        site_index_maps: Dict[int, Dict[str, int]] = {}
        for y in feature_years:
            path = os.path.join(features_dir, FEATURES_FNAME.format(year=y))
            if not os.path.exists(path):
                continue
            ds = xr.open_dataset(path, chunks={"time": -1})
            ds_features[y] = ds
            sids = ds["site_id"].values
            site_index_maps[y] = {str(s): i for i, s in enumerate(sids)}

        # ── Targets (LAI) and PFT → per-site vectors in RAM ──
        # Read from the pixelset files (output of build_pixelset_targets) by
        # site_id, like the ERA5 features. Per-site LAI (36) / PFT (15) is tiny.
        targ_lut: Dict[int, Dict[str, np.ndarray]] = {}
        pft_lut:  Dict[int, Dict[str, np.ndarray]] = {}
        for y in target_years:
            tpath = os.path.join(target_dir, TARGETS_FNAME.format(year=y))
            if os.path.exists(tpath):
                with xr.open_dataset(tpath) as dt:
                    m = _read_site_vectors(dt["LAI"], site_ids, "dekad")
                targ_lut[y] = {s: m[i] for i, s in enumerate(site_ids)}
            ppath = os.path.join(pft_dir, PFT_FNAME.format(year=y))
            if os.path.exists(ppath):
                m = _read_site_vectors(_open_pft_array(ppath), site_ids, "pft")
                pft_lut[y] = {s: m[i] for i, s in enumerate(site_ids)}

        self._cyclic_window = self._build_cyclic_window()

        # ── Materialise samples ──
        self.samples: List[Dict] = []
        n_drop_missing_year = 0
        n_drop_nan_feat     = 0
        n_drop_no_pft       = 0
        n_drop_short_hist   = 0
        n_drop_unknown_site = 0

        for year in target_years:
            if year not in targ_lut or year not in ds_features:
                n_drop_missing_year += len(site_ids)
                continue
            if (year - 1) not in ds_features:
                n_drop_short_hist += len(site_ids)
                continue
            if year not in pft_lut:
                n_drop_no_pft += len(site_ids)
                continue

            idx_map_y    = site_index_maps[year]
            idx_map_ym1  = site_index_maps[year - 1]
            targ_y, pft_y = targ_lut[year], pft_lut[year]
            for site_id in site_ids:
                site_idx_y   = idx_map_y.get(site_id)
                site_idx_ym1 = idx_map_ym1.get(site_id)
                if site_idx_y is None or site_idx_ym1 is None:
                    n_drop_unknown_site += 1
                    continue
                sample = self._build_sample(
                    site_id, year, site_idx_y, site_idx_ym1,
                    ds_features, targ_y.get(site_id), pft_y.get(site_id),
                )
                if sample is None:
                    n_drop_nan_feat += 1
                    continue
                self.samples.append(sample)

        for ds in ds_features.values():
            ds.close()

        if verbose:
            print(
                f"[LAIDataset] {len(self.samples):,} samples kept | "
                f"dropped: missing_year={n_drop_missing_year}, "
                f"short_hist={n_drop_short_hist}, "
                f"no_pft={n_drop_no_pft}, "
                f"unknown_site={n_drop_unknown_site}, "
                f"nan_features={n_drop_nan_feat}"
            )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_cyclic_window(self) -> Optional[np.ndarray]:
        """
        Pre-build the seq_length cyclic vector. Since (doy_sin, doy_cos) are
        deterministic and do NOT depend on (site, year), we materialise them
        once.
        """
        if not CYCLIC_FEATURES:
            return None
        # Cyclic features are computed on a non-leap calendar (365 d).
        # Tile the year and slice to seq_length to handle multi-year windows.
        doy = np.arange(1, 366, dtype=np.float64)
        sin = np.sin(2 * np.pi * doy / 365.25)
        cos = np.cos(2 * np.pi * doy / 365.25)
        full = np.tile(np.stack([sin, cos], axis=0), reps=(1, 3))  # 3 years buffer
        return full[:, -self.seq_length:].astype(np.float32)

    def _build_sample(
        self, site_id: str, year: int,
        site_idx_y: int, site_idx_ym1: int,
        ds_features: Dict[int, xr.Dataset],
        target: Optional[np.ndarray], pft_vec: Optional[np.ndarray],
    ) -> Optional[Dict]:
        """
        Return a fully-materialised sample dict for (site, year), or None if
        any constraint is violated. Features are pulled along the `site` dim of
        ERA5_daily_pixelset_{Y}.nc; the LAI `target` (36,) and `pft_vec` (15,)
        are passed pre-loaded per site by __init__ (see _read_site_vectors).
        """
        # ── PFT vector for this site / year (pre-loaded) ──
        if pft_vec is None or pft_vec.shape[0] != N_PFT:
            return None

        # ── Daily feature window (concat Y-1 + Y, then last seq_length days) ──
        # The features file's main dim is `site`, indexed by site_idx.
        try:
            ds_ym1 = ds_features[year - 1]
            ds_y   = ds_features[year]
            feat_ym1 = ds_ym1[DYNAMIC_FEATURES].isel(site=site_idx_ym1)
            feat_y   = ds_y  [DYNAMIC_FEATURES].isel(site=site_idx_y)
        except Exception:
            return None

        # Stack to a single (time, n_dyn) numpy array via load + to_array
        feat_ym1_np = feat_ym1.to_array().transpose("time", "variable").values
        feat_y_np   = feat_y.to_array().transpose("time", "variable").values
        feat_full   = np.concatenate([feat_ym1_np, feat_y_np], axis=0)
        if feat_full.shape[0] < self.seq_length:
            return None
        feat_full = feat_full[-self.seq_length:].astype(np.float32)
        if not np.all(np.isfinite(feat_full)):
            return None

        # ── Target vector (36 dekads of year Y, pre-loaded) ──
        if target is None or target.shape[0] != N_DEKAD_YEAR:
            return None
        target = target.astype(np.float32, copy=True)  # don't mutate the cached row

        # ── Anomaly mode: subtract climatology ──
        if self.anomaly_clim is not None:
            clim_vec = self.anomaly_clim.get(site_id)
            if clim_vec is None:
                return None
            target = (target - clim_vec.astype(np.float32))

        # ── Build the full feature matrix in canonical order ──
        feat_mat = self._assemble_feature_matrix(feat_full, pft_vec, year)
        if feat_mat is None:
            return None

        # ── LAI normalisation (independent of feature norm) ──
        if (self.normalize_lai
                and self.anomaly_clim is None
                and self.norm_stats is not None
                and "LAI" in self.norm_stats):
            mu = float(self.norm_stats["LAI"]["mean"])
            sd = float(self.norm_stats["LAI"]["std"])
            target = (target - mu) / max(sd, 1e-8)

        return {
            "site_id": site_id,
            "year":    int(year),
            "features": feat_mat,                  # (n_feat, seq_length) float32
            "target":   target.reshape(1, -1),     # (1, 36) float32
        }

    def _assemble_feature_matrix(
        self, dyn_window: np.ndarray, pft_vec: np.ndarray, year: int,
    ) -> Optional[np.ndarray]:
        """
        Concatenate DYNAMIC + CYCLIC + CO2 + PFT channels in canonical order.
        Applies log1p (on heavy-tailed dynamic feats) and z-scoring when
        `norm_stats` is provided. Returns (n_features, seq_length) float32.
        """
        n_t = dyn_window.shape[0]
        dyn = dyn_window.T.astype(np.float32)              # (n_dyn, seq_length)

        # log1p on heavy-tailed BEFORE z-score
        if self.norm_stats is not None:
            for i, name in enumerate(DYNAMIC_FEATURES):
                vals = dyn[i]
                if name in LOG_TRANSFORM_FEATURES:
                    vals = np.log1p(np.clip(vals, 0.0, None))
                mu = float(self.norm_stats[name]["mean"])
                sd = float(self.norm_stats[name]["std"])
                dyn[i] = (vals - mu) / max(sd, 1e-8)

        parts = [dyn]

        # Cyclic (deterministic, identical for every sample)
        if self._cyclic_window is not None:
            cyc = self._cyclic_window.copy()
            if self.norm_stats is not None:
                for i, name in enumerate(CYCLIC_FEATURES):
                    mu = float(self.norm_stats[name]["mean"])
                    sd = float(self.norm_stats[name]["std"])
                    cyc[i] = (cyc[i] - mu) / max(sd, 1e-8)
            parts.append(cyc)

        # CO2 (constant per year)
        if add_co2_features and CO2_FEATURES and self.co2_lut is not None:
            co2_val = self.co2_lut.get(int(year))
            if co2_val is None:
                return None
            co2 = np.full((1, n_t), float(co2_val), dtype=np.float32)
            if self.norm_stats is not None and "co2" in self.norm_stats:
                mu = float(self.norm_stats["co2"]["mean"])
                sd = float(self.norm_stats["co2"]["std"])
                co2 = (co2 - mu) / max(sd, 1e-8)
            parts.append(co2)

        # PFT (constant per year, broadcast over time)
        pft = np.broadcast_to(pft_vec.reshape(-1, 1), (N_PFT, n_t)).astype(np.float32).copy()
        if self.norm_stats is not None:
            for i, name in enumerate(PFT_COLS):
                if name not in self.norm_stats:
                    continue
                mu = float(self.norm_stats[name]["mean"])
                sd = float(self.norm_stats[name]["std"])
                pft[i] = (pft[i] - mu) / max(sd, 1e-8)
        parts.append(pft)

        return np.concatenate(parts, axis=0)

    # ── Required Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        feat = torch.from_numpy(s["features"])
        tgt  = torch.from_numpy(s["target"])
        return feat, tgt

    def get_site_info(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {"site_id": s["site_id"], "year": s["year"]}

    @property
    def feature_channels(self) -> int:
        return len(ALL_FEATURES)


# ── RAM-resident dataset (used by train_full_RAM and prediction.predict) ──


class RamLAIDataset(LAIDataset):
    """
    In-RAM variant of `LAIDataset`. Loads the raw daily features (per year,
    per requested site) plus targets / PFT into memory once; reuses the parent
    class's `_assemble_feature_matrix` / `_build_cyclic_window` so the produced
    samples are byte-for-byte identical to the on-disk path.

    Memory is the *raw* footprint (n_site × n_year × 365 × n_dyn × 4 B); the
    (n_features × seq_length) tensor is assembled on demand in __getitem__.
    """

    def __init__(
        self,
        features_dir: str,
        target_dir: str,
        pft_dir: str,
        years: List[int],
        site_ids: List[str],
        seq_length: int = SEQ_LENGTH_DEFAULT,
        norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
        anomaly_clim: Optional[Dict[str, np.ndarray]] = None,
        co2_lut: Optional[Dict[int, float]] = None,
        normalize_lai: bool = True,
        verbose: bool = True,
        threaded_read: bool = False,
        parent_map: Optional[Dict[str, str]] = None,
        daily_mode: str = "off",
        daily_target_dir: str = "",
    ) -> None:
        # Skip the parent's disk-walking __init__; set only what the reused
        # assembly helpers rely on.
        Dataset.__init__(self)
        # daily_mode selects how the LAI target is produced:
        #   "off"    → 36 dekad values (default, historical behaviour)
        #   "interp" → the 365-day linear-interpolated curve (train: loss on all days)
        #   "obs"    → 365-day grid, real dekads scattered at their DOY, NaN elsewhere
        #              (val/predict: the NaN-safe loss then scores only real obs days)
        if daily_mode not in ("off", "interp", "obs"):
            raise ValueError(f"daily_mode must be off/interp/obs, got {daily_mode!r}")
        self.daily_mode = daily_mode
        # parent_map {site_id05: site_id01}: when set, ERA5 features live on the
        # deduplicated 0.1° grid (ERA5_daily_pixelset with site_id "E{lat}_{lon}")
        # and each requested 0.05° site reads the row of its containing 0.1° cell.
        # Targets/PFT stay keyed by the 0.05° site_id, so every downstream map
        # (self._row, self.sample_index, LAI/PFT LUTs) keeps the 0.05° id.
        self.parent_map    = parent_map
        self.seq_length    = seq_length
        self.norm_stats    = norm_stats
        self.anomaly_clim  = anomaly_clim
        self.co2_lut       = co2_lut
        self.normalize_lai = normalize_lai
        self._cyclic_window = self._build_cyclic_window()

        target_years  = sorted(set(int(y) for y in years))
        feature_years = sorted(set(target_years) | {y - 1 for y in target_years})
        # unique, order-preserving
        site_list = list(dict.fromkeys(site_ids))

        # ── Raw daily features → RAM (parallel read, one file per feature year) ──
        self._dyn: Dict[int, np.ndarray] = {}      # year → (n_loaded, T, n_dyn)
        self._row: Dict[int, Dict[str, int]] = {}  # year → {site_id: row}

        def _read_year(y):
            path = os.path.join(features_dir, FEATURES_FNAME.format(year=y))
            if not os.path.exists(path):
                return None
            with xr.open_dataset(path) as ds:
                # `s` (self._row key) stays the 0.05° site id; only the ERA5 row
                # lookup goes through the 0.1° parent id when parent_map is set.
                # Without parent_map, feat site_id == requested site_id.
                pm = self.parent_map
                if pm is None:
                    keys = list(site_list)
                    feat = keys
                else:
                    keys = [s for s in site_list if s in pm]
                    feat = [pm[s] for s in keys]
                if not keys:
                    return None
                # Vectorised id→row lookup (pandas, C hash) over the file's coord.
                file_ids = pd.Index(np.asarray(ds["site_id"].values).astype(str))
                feat_rows = file_ids.get_indexer(pd.Index(feat))     # -1 if absent
                keep = feat_rows >= 0
                if not keep.any():
                    return None
                keys = [k for k, ok in zip(keys, keep) if ok]
                sel = feat_rows[keep].astype(np.int64)               # parent rows (may repeat)
                n_t = ds.sizes["time"]
                arr = np.empty((sel.size, n_t, len(DYNAMIC_FEATURES)),
                               dtype=np.float32)
                # Read each DISTINCT parent cell ONCE, in SORTED (chunk-aligned)
                # order, then broadcast back to the (possibly repeated) requested
                # sites. Turns millions of scattered NFS/HDF5 reads into a few
                # sequential chunk reads — bit-identical to a per-site isel.
                uniq, inv = np.unique(sel, return_inverse=True)      # uniq sorted
                for i, name in enumerate(DYNAMIC_FEATURES):
                    block = ds[name].isel(site=uniq).values          # (n_t, n_uniq)
                    arr[:, :, i] = block.T[inv]
            return y, arr, {s: k for k, s in enumerate(keys)}

        # Default is SEQUENTIAL on purpose: the bundled libhdf5/netCDF4 in the
        # cluster venv is not thread-safe — opening files concurrently (even
        # different ones) corrupts HDF5's global state and segfaults ("Can't open
        # HDF5 attribute" / "double free or corruption"). The per-variable
        # preallocated read in _read_year still avoids xarray's whole-cube
        # .to_array() copy (the real Level-A win).
        #
        # threaded_read=True (EXPERIMENTAL, opt-in) restores the concurrent
        # per-year read to overlap NFS latency — ONLY safe on a venv whose
        # netCDF4/h5py share a single thread-safe libhdf5.
        if threaded_read:
            from concurrent.futures import ThreadPoolExecutor
            n_workers = min(len(feature_years), 6) or 1
            with ThreadPoolExecutor(max_workers=n_workers) as _ex:
                results = list(_ex.map(_read_year, feature_years))
        else:
            results = (_read_year(y) for y in feature_years)
        _t0 = time.time()
        _nfy = len(feature_years)
        for _done, _res in enumerate(results, 1):
            if _res is None:
                if verbose:
                    print(f"  [load] features {_done}/{_nfy}  (year missing)  "
                          f"{time.time() - _t0:.0f}s", flush=True)
                continue
            y_res, arr, row_map = _res
            self._dyn[y_res] = arr
            self._row[y_res] = row_map
            if verbose:
                print(f"  [load] features {_done}/{_nfy}  year {y_res}  "
                      f"({arr.shape[0]:,} sites)  {time.time() - _t0:.0f}s",
                      flush=True)

        # ── Targets (LAI) and PFT → RAM (read by site_id from pixelset files) ──
        self._targ: Dict[int, Dict[str, np.ndarray]] = {}
        self._pft:  Dict[int, Dict[str, np.ndarray]] = {}
        _t0 = time.time()
        _lai_rows = None
        _pft_rows = None
        for _j, y in enumerate(target_years, 1):
            tpath = os.path.join(target_dir, TARGETS_FNAME.format(year=y))
            if os.path.exists(tpath):
                with xr.open_dataset(tpath) as dt:
                    if _lai_rows is None:                    # built once, reused
                        _lai_rows = _site_row_index(dt["LAI"], site_list)
                    vals = _read_site_vectors(dt["LAI"], site_list, "dekad",
                                              rows=_lai_rows)
                self._targ[y] = {s: vals[i] for i, s in enumerate(site_list)}
            ppath = os.path.join(pft_dir, PFT_FNAME.format(year=y))
            if os.path.exists(ppath):
                pa = _open_pft_array(ppath)
                if _pft_rows is None:                        # built once, reused
                    _pft_rows = _site_row_index(pa, site_list)
                pv = _read_site_vectors(pa, site_list, "pft", rows=_pft_rows)
                self._pft[y] = {s: pv[i] for i, s in enumerate(site_list)}
            if verbose:
                print(f"  [load] targets/pft {_j}/{len(target_years)}  year {y}  "
                      f"{time.time() - _t0:.0f}s", flush=True)

        # ── Daily LAI target (only when this instance trains on the 365-day
        #    interpolated curve). "obs"/"off" reuse the dekadal self._targ above.
        self._targ_daily: Dict[int, Dict[str, np.ndarray]] = {}
        if self.daily_mode == "interp":
            if not daily_target_dir:
                raise ValueError("daily_mode='interp' requires daily_target_dir")
            _daily_rows = None
            for y in target_years:
                dpath = os.path.join(daily_target_dir,
                                     LAI_DAILY_FNAME.format(year=y))
                if os.path.exists(dpath):
                    with xr.open_dataset(dpath) as dd:
                        if _daily_rows is None:              # built once, reused
                            _daily_rows = _site_row_index(dd["LAI"], site_list)
                        vals = _read_site_vectors(dd["LAI"], site_list, "day",
                                                  rows=_daily_rows)
                    self._targ_daily[y] = {s: vals[i]
                                           for i, s in enumerate(site_list)}

        # ── Daily climatology (anomaly + daily): interpolate each site's dekadal
        #    climatology to 365 days so the daily/obs targets can be turned into
        #    anomalies (subtracted in __getitem__). At the 36 dekad DOYs the daily
        #    clim equals the dekadal clim, so the "obs" val target gets the correct
        #    per-dekad anomaly. Built only for this dataset's sites with a clim.
        self._anom_clim_daily: Dict[str, np.ndarray] = {}
        if self.daily_mode != "off" and anomaly_clim is not None:
            for s in site_list:
                c36 = anomaly_clim.get(s)
                if c36 is not None:
                    self._anom_clim_daily[s] = _clim36_to_365(
                        np.asarray(c36, dtype=np.float32))

        # ── Pre-normalise ONCE (perf): apply log1p + z-score to the raw dynamic
        #    features and the PFT vectors in RAM a single time, and pre-build the
        #    (deterministic) normalised cyclic window + per-year CO2 scalar. With
        #    this, __getitem__ only slices + concatenates — the per-feature Python
        #    loop of _assemble_feature_matrix no longer runs per sample/epoch.
        #    Bit-identical to that loop (same μ/σ, same log1p/clip, float32).
        self._prenorm = norm_stats is not None
        self._co2_on  = bool(add_co2_features and CO2_FEATURES
                             and co2_lut is not None)
        self._cyclic_norm: Optional[np.ndarray] = None
        self._co2_val: Dict[int, float] = {}
        if self._prenorm:
            # Dynamic features (per year, in place): (n_loaded, T, n_dyn)
            for arr in self._dyn.values():
                for i, name in enumerate(DYNAMIC_FEATURES):
                    col = arr[:, :, i]
                    if name in LOG_TRANSFORM_FEATURES:
                        col = np.log1p(np.clip(col, 0.0, None))
                    mu = float(norm_stats[name]["mean"])
                    sd = float(norm_stats[name]["std"])
                    arr[:, :, i] = (col - mu) / max(sd, 1e-8)
            # PFT vectors (per site/year, in place). Vectorised over the 15 PFT
            # channels; float64 intermediate + float32 store → bit-identical to
            # the per-channel `(vec[i]-mu)/max(sd,1e-8)` loop above.
            pft_mask = np.array([n in norm_stats for n in PFT_COLS], dtype=bool)
            pft_mu   = np.array(
                [float(norm_stats[n]["mean"]) if n in norm_stats else 0.0
                 for n in PFT_COLS], dtype=np.float64)
            pft_sd   = np.array(
                [max(float(norm_stats[n]["std"]), 1e-8) if n in norm_stats else 1.0
                 for n in PFT_COLS], dtype=np.float64)
            for ydict in self._pft.values():
                for vec in ydict.values():
                    norm = (vec.astype(np.float64) - pft_mu) / pft_sd
                    vec[pft_mask] = norm[pft_mask].astype(np.float32)
            # Cyclic window (deterministic → normalise once)
            if self._cyclic_window is not None:
                cyc = self._cyclic_window.copy()
                for i, name in enumerate(CYCLIC_FEATURES):
                    mu = float(norm_stats[name]["mean"])
                    sd = float(norm_stats[name]["std"])
                    cyc[i] = (cyc[i] - mu) / max(sd, 1e-8)
                self._cyclic_norm = cyc
            # CO2 scalar per year (float32, normalised iff "co2" in stats)
            if self._co2_on:
                for y in target_years:
                    raw = co2_lut.get(int(y))
                    if raw is None:
                        continue
                    c = np.array([raw], dtype=np.float32)
                    if "co2" in norm_stats:
                        mu = float(norm_stats["co2"]["mean"])
                        sd = float(norm_stats["co2"]["std"])
                        c = (c - mu) / max(sd, 1e-8)
                    self._co2_val[int(y)] = float(c[0])

        # ── Build the valid sample index (same drop rules as LAIDataset) ──
        self.index: List[tuple] = []
        self.sample_index: Dict[tuple, int] = {}
        co2_on = bool(add_co2_features and CO2_FEATURES and co2_lut is not None)
        n_drop_missing_year = n_drop_short_hist = n_drop_no_pft = 0
        n_drop_unknown_site = n_drop_nan_feat = n_drop_co2 = 0

        for year in target_years:
            if year not in self._targ or year not in self._dyn:
                n_drop_missing_year += len(site_list)
                continue
            if self.daily_mode == "interp" and year not in self._targ_daily:
                n_drop_missing_year += len(site_list)
                continue
            if (year - 1) not in self._dyn:
                n_drop_short_hist += len(site_list)
                continue
            if year not in self._pft:
                n_drop_no_pft += len(site_list)
                continue
            if co2_on and year not in co2_lut:
                n_drop_co2 += len(site_list)
                continue
            row_y, row_ym1 = self._row[year], self._row[year - 1]
            targ_y, pft_y  = self._targ[year], self._pft[year]
            finite_y = self._finite_site_map(year)     # bulk NaN/length gate
            for site_id in site_list:
                if site_id not in row_y or site_id not in row_ym1:
                    n_drop_unknown_site += 1
                    continue
                pft_vec = pft_y.get(site_id)
                if pft_vec is None or pft_vec.shape[0] != N_PFT:
                    n_drop_no_pft += 1
                    continue
                tgt = targ_y.get(site_id)
                if tgt is None or tgt.shape[0] != N_DEKAD_YEAR:
                    n_drop_missing_year += 1
                    continue
                if self.anomaly_clim is not None and site_id not in self.anomaly_clim:
                    n_drop_unknown_site += 1
                    continue
                if not finite_y.get(site_id, False):
                    n_drop_nan_feat += 1
                    continue
                self.sample_index[(site_id, year)] = len(self.index)
                self.index.append((site_id, year))

        if verbose:
            print(
                f"[RamLAIDataset] {len(self.index):,} samples in RAM | "
                f"dropped: missing_year={n_drop_missing_year}, "
                f"short_hist={n_drop_short_hist}, no_pft={n_drop_no_pft}, "
                f"unknown_site={n_drop_unknown_site}, co2={n_drop_co2}, "
                f"nan_features={n_drop_nan_feat}"
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _window(self, site_id: str, year: int) -> Optional[np.ndarray]:
        """Last `seq_length` days of the (Y-1, Y) concat, or None if invalid."""
        ym1 = self._dyn[year - 1][self._row[year - 1][site_id]]   # (T, n_dyn)
        y   = self._dyn[year][self._row[year][site_id]]
        full = np.concatenate([ym1, y], axis=0)
        if full.shape[0] < self.seq_length:
            return None
        full = full[-self.seq_length:]
        if not np.all(np.isfinite(full)):
            return None
        return full.astype(np.float32, copy=False)

    def _finite_site_map(self, year: int) -> Dict[str, bool]:
        """Vectorised, bit-identical replacement for the per-sample `_window`
        length/finiteness gate used at index-build time. Returns {site_id:
        valid} for every site present in BOTH `year` and `year-1`, computed in
        bulk (one concat + isfinite per chunk) instead of one per sample."""
        row_y, row_ym1 = self._row[year], self._row[year - 1]
        dyn_y, dyn_ym1 = self._dyn[year], self._dyn[year - 1]
        common = [s for s in row_y if s in row_ym1]
        if dyn_ym1.shape[1] + dyn_y.shape[1] < self.seq_length:
            return {s: False for s in common}          # too short → all invalid
        rows_y   = np.fromiter((row_y[s]   for s in common), np.int64, len(common))
        rows_ym1 = np.fromiter((row_ym1[s] for s in common), np.int64, len(common))
        out: Dict[str, bool] = {}
        CHUNK = 8192
        for start in range(0, len(common), CHUNK):
            sl = slice(start, start + CHUNK)
            full = np.concatenate([dyn_ym1[rows_ym1[sl]], dyn_y[rows_y[sl]]],
                                  axis=1)[:, -self.seq_length:, :]
            fin = np.isfinite(full).all(axis=(1, 2))
            for k, s in enumerate(common[start:start + CHUNK]):
                out[s] = bool(fin[k])
        return out

    def _assemble_prenorm(self, dyn_window: np.ndarray,
                          pft_vec: np.ndarray, year: int) -> np.ndarray:
        """Assemble the feature matrix from already-normalised pieces (no per-
        feature loop). Channel order matches _assemble_feature_matrix exactly:
        [dynamic, cyclic, co2?, pft]."""
        n_t = dyn_window.shape[0]
        parts = [dyn_window.T]                        # (n_dyn, n_t), normalised
        if self._cyclic_norm is not None:
            parts.append(self._cyclic_norm)           # (2, n_t), normalised
        if self._co2_on:
            parts.append(np.full((1, n_t), self._co2_val[year],
                                 dtype=np.float32))
        parts.append(np.broadcast_to(pft_vec.reshape(-1, 1),
                                     (N_PFT, n_t)).astype(np.float32))
        return np.concatenate(parts, axis=0)

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index)

    def get_site_info(self, idx: int) -> Dict:
        site_id, year = self.index[idx]
        return {"site_id": site_id, "year": int(year)}

    def __getitem__(self, idx: int):
        site_id, year = self.index[idx]
        feat_full = self._window(site_id, year)            # filtered → not None
        pft_vec   = self._pft[year][site_id]
        feat_mat  = (self._assemble_prenorm(feat_full, pft_vec, year)
                     if self._prenorm
                     else self._assemble_feature_matrix(feat_full, pft_vec, year))

        if self.daily_mode == "interp":
            target = self._targ_daily[year][site_id].copy()          # (365,)
        elif self.daily_mode == "obs":
            # 365-day grid with the real dekad obs at their DOY, NaN elsewhere:
            # the NaN-safe loss/metric then scores ONLY the true observation days.
            target = np.full(N_DAYS_YEAR, np.nan, dtype=np.float32)
            target[_OBS_POSITIONS_ARR] = self._targ[year][site_id]
        else:
            target = self._targ[year][site_id].copy()                # (36,)
        if self.anomaly_clim is not None:
            if self.daily_mode != "off":
                target = target - self._anom_clim_daily[site_id]      # (365,) clim
            else:
                target = target - self.anomaly_clim[site_id].astype(np.float32)
        elif (self.normalize_lai and self.norm_stats is not None
              and "LAI" in self.norm_stats):
            mu = float(self.norm_stats["LAI"]["mean"])
            sd = float(self.norm_stats["LAI"]["std"])
            target = (target - mu) / max(sd, 1e-8)

        feat = torch.from_numpy(feat_mat)
        tgt  = torch.from_numpy(target.reshape(1, -1).astype(np.float32))
        return feat, tgt
