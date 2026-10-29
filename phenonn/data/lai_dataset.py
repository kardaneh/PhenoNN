# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
LAI dataset on NetCDF pixelset cubes.

A target "site" is identified by ``pix_LLLL_OOOOO`` where ``LLLL`` and
``OOOOO`` are indices on the native 0.05 degree LAI/PFT grid (zero-padded).
This keeps diagnostic scripts written for LaiNN backward-compatible.

Features may live on either resolution::

    features_dir/ERA5_daily_pixelset_{YYYY}.nc   dims (time, site)
    target_dir/LAI_dekadal_{YYYY}.nc             dims (dekad, site)
    pft_dir/PFTmap_{YYYY}.nc                     dims (pft, site)

On the 0.05 degree identity route, all three files use the same ``pix_...``
IDs. On the deduplicated 0.1 degree ERA5 route, the feature file uses
``E...`` cell IDs and ``parent_map`` maps each target ``pix_...`` site to its
``E...`` parent.

Each sample carries:

``site_id``
    Site identifier, of the form ``pix_LLLL_OOOOO``.
``year``
    Target year Y (integer).
``features``
    ``(n_features, seq_length)`` float32 array covering the last
    ``seq_length`` days of Y. Requires the Y-1 feature file for the
    history to fit.
``target``
    ``(1, 36)`` float32 array of the 36 dekadal LAI observations of
    year Y.

Loading strategy
----------------
EAGER, in RAM. The raw daily features (per year, per requested site)
plus the LAI and PFT vectors are read into memory once. ``__getitem__``
assembles the feature matrix on demand from slices of those arrays.
The result is bit-identical to the on-disk path the class replaces,
but with zero per-epoch I/O.

Memory footprint
----------------

The raw memory footprint is:

    n_site × n_year × 365 × n_dyn × 4 B

Anomaly mode
------------

When ``anomaly_clim`` is provided, the target is replaced by
``LAI - climatology(site, dekad)``, and LAI z-scoring is disabled.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from phenonn.utils.config import (
    ALL_FEATURES,
    CO2_FEATURES,
    CYCLIC_FEATURES,
    DYNAMIC_FEATURES,
    FEATURES_FNAME,
    LOG_TRANSFORM_FEATURES,
    N_DEKAD_YEAR,
    N_PFT,
    PFT_COLS,
    PFT_FNAME,
    SEQ_LENGTH_DEFAULT,
    TARGETS_FNAME,
    VALID_FNAME,
    add_co2_features,
)
from phenonn.utils.wrappers import _OBS_POSITIONS

# Daily-LAI mode (--daily_lai): 365-day interpolated target files and the
# 0-indexed positions of the 36 dekad observations within a 365-day year.
LAI_DAILY_FNAME = "LAI_daily_{year}.nc"
N_DAYS_YEAR = 365
_OBS_POSITIONS_ARR = np.array(_OBS_POSITIONS, dtype=np.int64)


# ── Dekad → daily climatology ───────────────────────────────────────────────


def _clim36_to_365(clim36: np.ndarray) -> np.ndarray:
    """
    Linear-interpolate a 36-dekad climatology onto the 365-day grid.

    The interpolation uses the same anchor points as the daily target
    (day-of-year of each dekad). Values at the 36 dekad DOYs are preserved
    exactly. Returns NaN if fewer than two dekads are finite.

    Parameters
    ----------
    clim36 : np.ndarray
        Array of shape ``(36,)`` with the dekadal climatology.

    Returns
    -------
    np.ndarray
        Array of shape ``(365,)`` on the daily grid.
    """
    finite = np.isfinite(clim36)
    out = np.full(N_DAYS_YEAR, np.nan, dtype=np.float32)
    if int(finite.sum()) >= 2:
        doy = (_OBS_POSITIONS_ARR + 1).astype(np.float64)  # 1-based dekad DOYs
        out[:] = np.interp(
            np.arange(1, N_DAYS_YEAR + 1),
            doy[finite],
            clim36[finite].astype(np.float64),
        ).astype(np.float32)
    return out


# ── Site-ID encoding ────────────────────────────────────────────────────────


def encode_site_id(lat_idx: int, lon_idx: int) -> str:
    """
    Build the canonical site id ``pix_LLLL_OOOOO``.

    Parameters
    ----------
    lat_idx : int
        Latitude index in the native grid.
    lon_idx : int
        Longitude index in the native grid.

    Returns
    -------
    str
        Zero-padded site identifier.
    """
    return f"pix_{int(lat_idx):04d}_{int(lon_idx):05d}"


def decode_site_id(site_id: str) -> Tuple[int, int]:
    """
    Inverse of :func:`encode_site_id`.

    Parameters
    ----------
    site_id : str
        Site identifier of the form ``pix_LLLL_OOOOO``.

    Returns
    -------
    lat_idx, lon_idx : tuple of int

    Raises
    ------
    ValueError
        If ``site_id`` is not of the expected form.
    """
    parts = site_id.split("_")
    if len(parts) != 3 or parts[0] != "pix":
        raise ValueError(f"Malformed site_id {site_id!r}; expected pix_LLLL_OOOOO")
    return int(parts[1]), int(parts[2])


def generate_site_ids_from_range(
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
) -> List[str]:
    """
    Every pix id in the inclusive (row, col) box.

    Parameters
    ----------
    row_range, col_range : tuple of int
        Inclusive ``(min, max)`` bounds on the latitude / longitude index.

    Returns
    -------
    list of str
    """
    r0, r1 = row_range
    c0, c1 = col_range
    return [encode_site_id(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


# ── CO2 LUT ─────────────────────────────────────────────────────────────────


def load_co2_lut(path: str) -> Dict[int, float]:
    """
    Parse a ``Annee_YYYY=VALUE`` CO2 file.

    Parameters
    ----------
    path : str
        Path to the CO2 lookup file.

    Returns
    -------
    dict
        Mapping ``{year: ppm}``.

    Raises
    ------
    ValueError
        If no valid entry could be parsed.
    """
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


# ── PFT-array detection ─────────────────────────────────────────────────────


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """
    Return a ``(pft, latitude, longitude)`` DataArray from a PFT file.

    Three layouts are supported:

    1. ORCHIDEE PFTmap
       ``maxvegetfrac(time_counter=1, veget=15, lat, lon)``.
    2. Fifteen separate variables ``pft1_frac … pft15_frac``.
    3. A single 3-D variable on ``(pft, lat, lon)`` with a name from
       ``("pft_fraction", "PFT", "pft", "pft_frac", "fraction")``.

    Parameters
    ----------
    ds : xr.Dataset
        Opened PFT file.

    Returns
    -------
    xr.DataArray

    Raises
    ------
    ValueError
        If no supported layout is found.
    """
    rename = {}
    for src, dst in [("lat", "latitude"), ("lon", "longitude"), ("veget", "pft")]:
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
            [ds[v].expand_dims(pft=[k - 1]) for k, v in enumerate(pft_named, start=1)],
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


# ── Pixelset readers ────────────────────────────────────────────────────────


def _site_row_index(da: xr.DataArray, site_ids: List[str]) -> np.ndarray:
    """
    Return the row of each requested site in ``da``, or ``-1`` if absent.

    Vectorised (pandas hash join). The pixelset ``site_id`` coordinate is
    identical across years, so callers build this once and reuse it.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with a ``site_id`` coordinate.
    site_ids : list of str

    Returns
    -------
    np.ndarray
        Int64 array of the same length as ``site_ids``.
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
    Return a ``(n_site, size(value_dim))`` float32 matrix for the sites.

    Unknown site ids come back as NaN rows. When ``rows`` is provided the
    id→row lookup is skipped; the caller can precompute it because the
    pixelset ``site_id`` is identical every year.

    Parameters
    ----------
    da : xr.DataArray
        Pixelset DataArray with a ``site_id`` coordinate.
    site_ids : list of str
    value_dim : str
        Name of the second dimension (``"dekad"``, ``"pft"``, ``"day"``).
    rows : np.ndarray, optional
        Precomputed row indices, as returned by :func:`_site_row_index`.

    Returns
    -------
    np.ndarray
        Shape ``(len(site_ids), size(value_dim))``, float32.
    """
    if rows is None:
        rows = _site_row_index(da, site_ids)
    out = np.full((len(site_ids), da.sizes[value_dim]), np.nan, dtype=np.float32)
    ok = rows >= 0
    if ok.any():
        # The pixelset is chunked along `site`; a scattered per-site isel would
        # decompress the same chunk repeatedly. Read the whole variable once,
        # then gather in memory — bit-identical, no thrashing.
        full = da.transpose("site", value_dim).values
        out[ok] = full[rows[ok]].astype(np.float32)
    return out


def _open_pft_array(path: str) -> xr.DataArray:
    """
    Open a pixelset PFT file and return its ``pft_frac(pft, site)`` array.

    Uses ``engine="netcdf4", decode_times=False`` to stay on one libhdf5
    (mixing netcdf4 and h5netcdf corrupts HDF5's global state).

    Parameters
    ----------
    path : str

    Returns
    -------
    xr.DataArray
    """
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    return ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]


# ── Anomaly climatology ─────────────────────────────────────────────────────


def compute_climatology_lookup(
    target_dir: str,
    clim_years: List[int],
    wanted_sites: List[str],
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Per-(site, dekad) mean of LAI over ``clim_years``.

    Parameters
    ----------
    target_dir : str
        Folder of ``LAI_dekadal_{Y}.nc`` files.
    clim_years : list of int
        Years used to build the mean.
    wanted_sites : list of str
    verbose : bool, optional
        Print a line per missing file.

    Returns
    -------
    dict
        ``{site_id: (36,) np.float32 array}``. NaN where no observation in
        the window.
    """
    target_dir = os.fspath(target_dir)
    wanted = sorted(set(wanted_sites))
    if not wanted:
        return {}

    sum_ = np.zeros((len(wanted), N_DEKAD_YEAR), dtype=np.float64)
    cnt = np.zeros((len(wanted), N_DEKAD_YEAR), dtype=np.int32)

    for y in sorted(set(clim_years)):
        path = os.path.join(target_dir, TARGETS_FNAME.format(year=y))
        if not os.path.exists(path):
            if verbose:
                print(f"  [clim] {os.path.basename(path)} missing — skipped")
            continue
        with xr.open_dataset(path) as ds:
            vals = _read_site_vectors(ds["LAI"], wanted, "dekad")
        finite = np.isfinite(vals)
        sum_ += np.where(finite, vals, 0.0)
        cnt += finite.astype(np.int32)

    mean = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return {sid: mean[i] for i, sid in enumerate(wanted)}


# ── Valid-pixels mask and site pools ────────────────────────────────────────


def load_valid_mask(valid_dir: str, year: int) -> Optional[xr.DataArray]:
    """
    Return the boolean ``(lat, lon)`` mask for ``year``, or ``None``.

    Parameters
    ----------
    valid_dir : str
        Folder of ``valid_pixels_{Y}.nc`` files.
    year : int

    Returns
    -------
    xr.DataArray or None
    """
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
    Return the site ids stored in a ``selected_pixels.nc`` file.

    Parameters
    ----------
    selected_path : str

    Returns
    -------
    list of str
    """
    ds = xr.open_dataset(selected_path)
    sids = ds["site_id"].values
    ds.close()
    return [str(s) for s in sids]


def load_parent01_map(selected_pixels_01: str) -> Dict[str, str]:
    """
    Return ``{site_id05: site_id01}`` from a ``selected_pixels01_1.nc``.

    Used to read ERA5 features on the deduplicated 0.1° grid from a
    dataset keyed on native 0.05° sites.

    Parameters
    ----------
    selected_pixels_01 : str

    Returns
    -------
    dict
    """
    with xr.open_dataset(selected_pixels_01) as ds:
        site_id05 = np.asarray(ds["site_id05"].values).astype(str)
        parent_site = np.asarray(ds["parent_site"].values).astype(np.int64)
        site_id01 = np.asarray(ds["site_id"].values).astype(str)
    return {s05: site_id01[p] for s05, p in zip(site_id05, parent_site)}


def list_valid_sites(
    valid_dir: str,
    years: List[int],
    bbox_rows: Optional[Tuple[int, int]] = None,
    bbox_cols: Optional[Tuple[int, int]] = None,
) -> Dict[int, List[str]]:
    """
    For each year, return the pixel ids whose mask cell is True.

    Parameters
    ----------
    valid_dir : str
    years : list of int
    bbox_rows, bbox_cols : tuple of int, optional
        Inclusive ``(min, max)`` bounds on the lat / lon index.

    Returns
    -------
    dict
        ``{year: [site_id, …]}``.
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
            keep[r0 : r1 + 1, c0 : c1 + 1] = True
            m = m & keep
        lats, lons = np.where(m)
        out[y] = [encode_site_id(int(la), int(lo)) for la, lo in zip(lats, lons)]
    return out


# ── RamLAIDataset ───────────────────────────────────────────────────────────


class RamLAIDataset(Dataset):
    """
    In-RAM LAI dataset on pixelset NetCDF cubes.

    The raw daily features (per year, per requested site) plus the LAI
    and PFT vectors are loaded into RAM once. ``__getitem__`` assembles
    the feature matrix on demand from slices of those arrays. Memory is
    the *raw* footprint (``n_site × n_year × 365 × n_dyn × 4 B``); the
    ``(n_features × seq_length)`` tensor is built on the fly.

    Parameters
    ----------
    features_dir, target_dir, pft_dir : str
        Folders containing the per-year NetCDF files. Naming follows
        ``phenonn.utils.config`` (``FEATURES_FNAME``, ``TARGETS_FNAME``,
        ``PFT_FNAME``).
    years : list of int
        Target years. The feature window of year Y also requires the
        file of Y-1 — samples for which Y-1 is missing are dropped.
    site_ids : list of str
        Sites to evaluate, in ``pix_LLLL_OOOOO`` form.
    seq_length : int
        Feature window length in days (default 720).
    norm_stats : dict or None
        Per-feature mean / std. When provided, ``log1p`` is applied to
        ``LOG_TRANSFORM_FEATURES`` before z-scoring. LAI is z-scored too
        unless ``normalize_lai=False`` (or anomaly mode is on).
    anomaly_clim : dict or None
        Per-site 36-dekad climatology. When set, the target becomes
        ``LAI - climatology`` and LAI z-scoring is disabled.
    co2_lut : dict or None
        Per-year CO2 concentration.
    normalize_lai : bool
        Skip LAI z-scoring when False. Features remain z-scored.
    verbose : bool
        Print a line per load step and one summary at the end.
    threaded_read : bool, optional
        EXPERIMENTAL. Read the per-year feature files with a thread
        pool. Unsafe on some netCDF4 / h5py builds; leave False unless
        the venv has a single thread-safe libhdf5.
    parent_map : dict or None, optional
        ``{site_id05: site_id01}``. When set, features are read from
        the deduplicated 0.1 degree file at the parent cell of each
        0.05 degree site. Targets / PFT stay keyed by the 0.05 degree id.
        A map with zero overlap raises ``ValueError``; a partial map is
        reported and the missing sites are dropped.
    daily_mode : str, optional
        Target mode. Supported values are ``"off"``, ``"interp"``,
        and ``"obs"``. The default is ``"off"``. ``"off"`` uses 36
        dekad values; ``"interp"`` uses a 365-day interpolated curve;
        ``"obs"`` places the 36 observations at their day-of-year
        positions and fills the other days with NaN.

    daily_target_dir : str, optional
        Directory containing ``LAI_daily_{Y}.nc`` files. Required
        when ``daily_mode`` is ``"interp"``.

    Examples
    --------
    >>> ds = RamLAIDataset(
    ...     features_dir="…/era5_pixelset",
    ...     target_dir="…/LAI_pixelset",
    ...     pft_dir="…/PFT_pixelset",
    ...     years=[2010, 2011],
    ...     site_ids=["pix_0123_04567"],
    ...     seq_length=720,
    ... )
    >>> features, target = ds[0]
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
        super().__init__()
        if daily_mode not in ("off", "interp", "obs"):
            raise ValueError(f"daily_mode must be off/interp/obs, got {daily_mode!r}")

        self.features_dir = features_dir
        self.target_dir = target_dir
        self.pft_dir = pft_dir
        self.seq_length = seq_length
        self.norm_stats = norm_stats
        self.anomaly_clim = anomaly_clim
        self.co2_lut = co2_lut
        self.normalize_lai = normalize_lai
        self.daily_mode = daily_mode
        self.parent_map = parent_map
        self._cyclic_window = self._build_cyclic_window()

        target_years = sorted(set(int(y) for y in years))
        feature_years = sorted(set(target_years) | {y - 1 for y in target_years})
        site_list = list(dict.fromkeys(site_ids))

        if parent_map is not None:
            mapped = [s for s in site_list if s in parent_map]
            if not mapped:
                raise ValueError(
                    "parent_map matched none of the requested site IDs; "
                    "check that it was built from the same selected_pixels file."
                )
            if len(mapped) < len(site_list) and verbose:
                print(
                    f"  [warn] parent_map is missing {len(site_list) - len(mapped):,} "
                    f"of {len(site_list):,} requested sites; those sites will "
                    "be dropped.",
                    flush=True,
                )

        # ── Phase 1: raw daily features → RAM ──────────────────────────────
        self._dyn: Dict[int, np.ndarray] = {}
        self._row: Dict[int, Dict[str, int]] = {}
        self._read_dynamic_features(
            feature_years,
            site_list,
            threaded_read,
            verbose,
        )

        # ── Phase 2: LAI / PFT / daily LAI vectors → RAM ───────────────────
        self._targ: Dict[int, Dict[str, np.ndarray]] = {}
        self._pft: Dict[int, Dict[str, np.ndarray]] = {}
        self._targ_daily: Dict[int, Dict[str, np.ndarray]] = {}
        self._read_target_and_pft(
            target_years,
            site_list,
            daily_target_dir,
            verbose,
        )

        # ── Phase 3: daily climatology (anomaly + daily mode only) ─────────
        self._anom_clim_daily: Dict[str, np.ndarray] = {}
        if self.daily_mode != "off" and anomaly_clim is not None:
            for s in site_list:
                c36 = anomaly_clim.get(s)
                if c36 is not None:
                    self._anom_clim_daily[s] = _clim36_to_365(
                        np.asarray(c36, dtype=np.float32)
                    )

        # ── Phase 4: pre-normalise the RAM arrays once ─────────────────────
        self._prenorm = norm_stats is not None
        self._co2_on = bool(add_co2_features and CO2_FEATURES and co2_lut is not None)
        self._cyclic_norm: Optional[np.ndarray] = None
        self._co2_val: Dict[int, float] = {}
        if self._prenorm:
            self._prenormalise(target_years)

        # ── Phase 5: build the (site, year) → index map ────────────────────
        self.index: List[Tuple[str, int]] = []
        self.sample_index: Dict[Tuple[str, int], int] = {}
        self._build_sample_index(target_years, site_list, verbose)

    # ── Phase 1: dynamic features ──────────────────────────────────────────

    def _read_dynamic_features(
        self,
        feature_years: List[int],
        site_list: List[str],
        threaded_read: bool,
        verbose: bool,
    ) -> None:
        """
        Read the daily features for every requested year into RAM.

        Populates ``self._dyn`` (``{year: (n_site, T, n_dyn)}``) and
        ``self._row`` (``{year: {site_id: row_index}}``).
        """

        def _read_year(y):
            path = os.path.join(self.features_dir, FEATURES_FNAME.format(year=y))
            if not os.path.exists(path):
                return None
            with xr.open_dataset(path) as ds:
                pm = self.parent_map
                if pm is None:
                    keys = list(site_list)
                    feat = keys
                else:
                    keys = [s for s in site_list if s in pm]
                    feat = [pm[s] for s in keys]
                if not keys:
                    return None
                # Vectorised id→row lookup over the file's coord.
                file_ids = pd.Index(np.asarray(ds["site_id"].values).astype(str))
                feat_rows = file_ids.get_indexer(pd.Index(feat))
                keep = feat_rows >= 0
                if not keep.any():
                    return None
                keys = [k for k, ok in zip(keys, keep) if ok]
                sel = feat_rows[keep].astype(np.int64)
                n_t = ds.sizes["time"]
                arr = np.empty((sel.size, n_t, len(DYNAMIC_FEATURES)), dtype=np.float32)
                # Read each DISTINCT parent cell once (sorted → chunk-aligned),
                # then broadcast back to the (possibly repeated) requested sites.
                uniq, inv = np.unique(sel, return_inverse=True)
                for i, name in enumerate(DYNAMIC_FEATURES):
                    block = ds[name].isel(site=uniq).values  # (n_t, n_uniq)
                    arr[:, :, i] = block.T[inv]
            return y, arr, {s: k for k, s in enumerate(keys)}

        # SEQUENTIAL by default: the bundled libhdf5 / netCDF4 in some venvs
        # is not thread-safe (concurrent opens corrupt HDF5's global state
        # and segfault). threaded_read=True is the opt-in escape hatch.
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
                    print(
                        f"  [load] features {_done}/{_nfy}  (year missing)  "
                        f"{time.time() - _t0:.0f}s",
                        flush=True,
                    )
                continue
            y_res, arr, row_map = _res
            self._dyn[y_res] = arr
            self._row[y_res] = row_map
            if verbose:
                print(
                    f"  [load] features {_done}/{_nfy}  year {y_res}  "
                    f"({arr.shape[0]:,} sites)  {time.time() - _t0:.0f}s",
                    flush=True,
                )

    # ── Phase 2: targets, PFT and optional daily target ────────────────────

    def _read_target_and_pft(
        self,
        target_years: List[int],
        site_list: List[str],
        daily_target_dir: str,
        verbose: bool,
    ) -> None:
        """
        Read the per-site LAI, PFT and (optionally) daily LAI vectors.

        Populates ``self._targ``, ``self._pft`` and — when
        ``daily_mode == "interp"`` — ``self._targ_daily``.
        """
        _t0 = time.time()
        _lai_rows = None
        _pft_rows = None
        for _j, y in enumerate(target_years, 1):
            tpath = os.path.join(self.target_dir, TARGETS_FNAME.format(year=y))
            if os.path.exists(tpath):
                with xr.open_dataset(tpath) as dt:
                    if _lai_rows is None:  # built once, reused across years
                        _lai_rows = _site_row_index(dt["LAI"], site_list)
                    vals = _read_site_vectors(
                        dt["LAI"], site_list, "dekad", rows=_lai_rows
                    )
                self._targ[y] = {s: vals[i] for i, s in enumerate(site_list)}
            ppath = os.path.join(self.pft_dir, PFT_FNAME.format(year=y))
            if os.path.exists(ppath):
                pa = _open_pft_array(ppath)
                if _pft_rows is None:  # built once, reused across years
                    _pft_rows = _site_row_index(pa, site_list)
                pv = _read_site_vectors(pa, site_list, "pft", rows=_pft_rows)
                self._pft[y] = {s: pv[i] for i, s in enumerate(site_list)}
            if verbose:
                print(
                    f"  [load] targets/pft {_j}/{len(target_years)}  year {y}  "
                    f"{time.time() - _t0:.0f}s",
                    flush=True,
                )

        if self.daily_mode == "interp":
            if not daily_target_dir:
                raise ValueError("daily_mode='interp' requires daily_target_dir")
            _daily_rows = None
            for y in target_years:
                dpath = os.path.join(daily_target_dir, LAI_DAILY_FNAME.format(year=y))
                if os.path.exists(dpath):
                    with xr.open_dataset(dpath) as dd:
                        if _daily_rows is None:
                            _daily_rows = _site_row_index(dd["LAI"], site_list)
                        vals = _read_site_vectors(
                            dd["LAI"], site_list, "day", rows=_daily_rows
                        )
                    self._targ_daily[y] = {s: vals[i] for i, s in enumerate(site_list)}

    # ── Phase 4: pre-normalisation ─────────────────────────────────────────

    def _prenormalise(self, target_years: List[int]) -> None:
        """
        Apply ``log1p`` and z-scoring to the RAM arrays once.

        After this call, ``__getitem__`` only slices and concatenates —
        the per-feature Python loop is skipped entirely. Bit-identical
        to the per-sample path (same ``μ``/``σ``, same ``log1p``, float32).
        """
        # Dynamic features (per year, in place).
        for arr in self._dyn.values():
            for i, name in enumerate(DYNAMIC_FEATURES):
                col = arr[:, :, i]
                if name in LOG_TRANSFORM_FEATURES:
                    col = np.log1p(np.clip(col, 0.0, None))
                mu = float(self.norm_stats[name]["mean"])
                sd = float(self.norm_stats[name]["std"])
                arr[:, :, i] = (col - mu) / max(sd, 1e-8)

        # PFT vectors (vectorised over the 15 channels, in place).
        pft_mask = np.array([n in self.norm_stats for n in PFT_COLS], dtype=bool)
        pft_mu = np.array(
            [
                float(self.norm_stats[n]["mean"]) if n in self.norm_stats else 0.0
                for n in PFT_COLS
            ],
            dtype=np.float64,
        )
        pft_sd = np.array(
            [
                max(float(self.norm_stats[n]["std"]), 1e-8)
                if n in self.norm_stats
                else 1.0
                for n in PFT_COLS
            ],
            dtype=np.float64,
        )
        for ydict in self._pft.values():
            for vec in ydict.values():
                norm = (vec.astype(np.float64) - pft_mu) / pft_sd
                vec[pft_mask] = norm[pft_mask].astype(np.float32)

        # Cyclic window (deterministic → normalise once).
        if self._cyclic_window is not None:
            cyc = self._cyclic_window.copy()
            for i, name in enumerate(CYCLIC_FEATURES):
                mu = float(self.norm_stats[name]["mean"])
                sd = float(self.norm_stats[name]["std"])
                cyc[i] = (cyc[i] - mu) / max(sd, 1e-8)
            self._cyclic_norm = cyc

        # CO2 scalar per year.
        if self._co2_on:
            for y in target_years:
                raw = self.co2_lut.get(int(y))
                if raw is None:
                    continue
                c = np.array([raw], dtype=np.float32)
                if "co2" in self.norm_stats:
                    mu = float(self.norm_stats["co2"]["mean"])
                    sd = float(self.norm_stats["co2"]["std"])
                    c = (c - mu) / max(sd, 1e-8)
                self._co2_val[int(y)] = float(c[0])

    # ── Phase 5: sample index ──────────────────────────────────────────────

    def _build_sample_index(
        self,
        target_years: List[int],
        site_list: List[str],
        verbose: bool,
    ) -> None:
        """
        Walk every ``(year, site)`` pair and record the valid ones.

        Populates ``self.index`` and ``self.sample_index``. Drop reasons
        accumulate in a small dict and are reported once.
        """
        co2_on = bool(add_co2_features and CO2_FEATURES and self.co2_lut is not None)
        c = dict(
            missing_year=0,
            short_hist=0,
            no_pft=0,
            unknown_site=0,
            nan_feat=0,
            co2=0,
        )

        for year in target_years:
            if year not in self._targ or year not in self._dyn:
                c["missing_year"] += len(site_list)
                continue
            if self.daily_mode == "interp" and year not in self._targ_daily:
                c["missing_year"] += len(site_list)
                continue
            if (year - 1) not in self._dyn:
                c["short_hist"] += len(site_list)
                continue
            if year not in self._pft:
                c["no_pft"] += len(site_list)
                continue
            if co2_on and year not in self.co2_lut:
                c["co2"] += len(site_list)
                continue
            row_y, row_ym1 = self._row[year], self._row[year - 1]
            targ_y, pft_y = self._targ[year], self._pft[year]
            finite_y = self._finite_site_map(year)
            for site_id in site_list:
                if site_id not in row_y or site_id not in row_ym1:
                    c["unknown_site"] += 1
                    continue
                pft_vec = pft_y.get(site_id)
                if pft_vec is None or pft_vec.shape[0] != N_PFT:
                    c["no_pft"] += 1
                    continue
                tgt = targ_y.get(site_id)
                if tgt is None or tgt.shape[0] != N_DEKAD_YEAR:
                    c["missing_year"] += 1
                    continue
                if self.anomaly_clim is not None and site_id not in self.anomaly_clim:
                    c["unknown_site"] += 1
                    continue
                if not finite_y.get(site_id, False):
                    c["nan_feat"] += 1
                    continue
                self.sample_index[(site_id, year)] = len(self.index)
                self.index.append((site_id, year))

        if verbose:
            print(
                f"[RamLAIDataset] {len(self.index):,} samples in RAM | "
                f"dropped: missing_year={c['missing_year']}, "
                f"short_hist={c['short_hist']}, no_pft={c['no_pft']}, "
                f"unknown_site={c['unknown_site']}, co2={c['co2']}, "
                f"nan_features={c['nan_feat']}"
            )

    # ── Cyclic window ──────────────────────────────────────────────────────

    def _build_cyclic_window(self) -> Optional[np.ndarray]:
        """
        Pre-build the ``seq_length`` cyclic vector.

        ``(doy_sin, doy_cos)`` are deterministic and do not depend on the
        site, so they are materialised once. Returns ``None`` when no
        cyclic features are configured.

        Returns
        -------
        np.ndarray or None
        """
        if not CYCLIC_FEATURES:
            return None
        doy = np.arange(1, 366, dtype=np.float64)
        sin = np.sin(2 * np.pi * doy / 365.25)
        cos = np.cos(2 * np.pi * doy / 365.25)
        full = np.tile(np.stack([sin, cos], axis=0), reps=(1, 3))  # 3-year buffer
        return full[:, -self.seq_length :].astype(np.float32)

    # ── Runtime helpers ────────────────────────────────────────────────────

    def _window(self, site_id: str, year: int) -> Optional[np.ndarray]:
        """
        Return the last ``seq_length`` days of the (Y-1, Y) concat.

        Parameters
        ----------
        site_id : str
        year : int

        Returns
        -------
        np.ndarray or None
            Shape ``(seq_length, n_dyn)``, or ``None`` if the history is
            too short or contains a NaN.
        """
        ym1 = self._dyn[year - 1][self._row[year - 1][site_id]]
        y = self._dyn[year][self._row[year][site_id]]
        full = np.concatenate([ym1, y], axis=0)
        if full.shape[0] < self.seq_length:
            return None
        full = full[-self.seq_length :]
        if not np.all(np.isfinite(full)):
            return None
        return full.astype(np.float32, copy=False)

    def _finite_site_map(self, year: int) -> Dict[str, bool]:
        """
        Bulk finiteness test for every site present in both years.

        Bit-identical to looping :meth:`_window` per sample, but vectorised
        in chunks of 8192 sites.

        Parameters
        ----------
        year : int

        Returns
        -------
        dict
            ``{site_id: valid}``.
        """
        row_y, row_ym1 = self._row[year], self._row[year - 1]
        dyn_y, dyn_ym1 = self._dyn[year], self._dyn[year - 1]
        common = [s for s in row_y if s in row_ym1]
        if dyn_ym1.shape[1] + dyn_y.shape[1] < self.seq_length:
            return {s: False for s in common}
        rows_y = np.fromiter((row_y[s] for s in common), np.int64, len(common))
        rows_ym1 = np.fromiter((row_ym1[s] for s in common), np.int64, len(common))
        out: Dict[str, bool] = {}
        CHUNK = 8192
        for start in range(0, len(common), CHUNK):
            sl = slice(start, start + CHUNK)
            full = np.concatenate([dyn_ym1[rows_ym1[sl]], dyn_y[rows_y[sl]]], axis=1)[
                :, -self.seq_length :, :
            ]
            fin = np.isfinite(full).all(axis=(1, 2))
            for k, s in enumerate(common[start : start + CHUNK]):
                out[s] = bool(fin[k])
        return out

    def _assemble_prenorm(
        self, dyn_window: np.ndarray, pft_vec: np.ndarray, year: int
    ) -> np.ndarray:
        """
        Assemble the feature matrix from the pre-normalised pieces.

        Channel order: ``[dynamic, cyclic, co2?, pft]``.

        Parameters
        ----------
        dyn_window : np.ndarray
            Shape ``(seq_length, n_dyn)``.
        pft_vec : np.ndarray
            Shape ``(15,)``.
        year : int

        Returns
        -------
        np.ndarray
            Shape ``(n_features, seq_length)``.
        """
        n_t = dyn_window.shape[0]
        parts = [dyn_window.T]
        if self._cyclic_norm is not None:
            parts.append(self._cyclic_norm)
        if self._co2_on:
            parts.append(np.full((1, n_t), self._co2_val[year], dtype=np.float32))
        parts.append(
            np.broadcast_to(pft_vec.reshape(-1, 1), (N_PFT, n_t)).astype(np.float32)
        )
        return np.concatenate(parts, axis=0)

    def _assemble_feature_matrix(
        self,
        dyn_window: np.ndarray,
        pft_vec: np.ndarray,
        year: int,
    ) -> Optional[np.ndarray]:
        """
        Assemble the feature matrix on the fly (no pre-normalisation).

        Used when ``norm_stats`` is None. Returns ``None`` when CO2 is
        enabled but the year is missing from the lookup.

        Parameters
        ----------
        dyn_window : np.ndarray
            Shape ``(seq_length, n_dyn)``.
        pft_vec : np.ndarray
            Shape ``(15,)``.
        year : int

        Returns
        -------
        np.ndarray or None
            Shape ``(n_features, seq_length)``.
        """
        n_t = dyn_window.shape[0]
        dyn = dyn_window.T.astype(np.float32)

        parts = [dyn]
        if self._cyclic_window is not None:
            parts.append(self._cyclic_window.copy())

        if add_co2_features and CO2_FEATURES and self.co2_lut is not None:
            co2_val = self.co2_lut.get(int(year))
            if co2_val is None:
                return None
            parts.append(np.full((1, n_t), float(co2_val), dtype=np.float32))

        pft = (
            np.broadcast_to(pft_vec.reshape(-1, 1), (N_PFT, n_t))
            .astype(np.float32)
            .copy()
        )
        parts.append(pft)

        return np.concatenate(parts, axis=0)

    # ── Target assembly ────────────────────────────────────────────────────

    def _build_target(self, site_id: str, year: int) -> np.ndarray:
        """
        Produce the target vector in the configured mode.

        Parameters
        ----------
        site_id : str
        year : int

        Returns
        -------
        np.ndarray
            ``(36,)`` when ``daily_mode == "off"``, ``(365,)`` otherwise.
        """
        if self.daily_mode == "interp":
            return self._targ_daily[year][site_id].copy()
        if self.daily_mode == "obs":
            target = np.full(N_DAYS_YEAR, np.nan, dtype=np.float32)
            target[_OBS_POSITIONS_ARR] = self._targ[year][site_id]
            return target
        return self._targ[year][site_id].copy()

    def _finalise_target(self, site_id: str, target: np.ndarray) -> np.ndarray:
        """
        Apply anomaly subtraction or LAI z-scoring.

        Parameters
        ----------
        site_id : str
        target : np.ndarray

        Returns
        -------
        np.ndarray
        """
        if self.anomaly_clim is not None:
            if self.daily_mode != "off":
                return target - self._anom_clim_daily[site_id]
            return target - self.anomaly_clim[site_id].astype(np.float32)
        if (
            self.normalize_lai
            and self.norm_stats is not None
            and "LAI" in self.norm_stats
        ):
            mu = float(self.norm_stats["LAI"]["mean"])
            sd = float(self.norm_stats["LAI"]["std"])
            return (target - mu) / max(sd, 1e-8)
        return target

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index)

    def get_site_info(self, idx: int) -> Dict:
        """
        Return the metadata of the sample at ``idx``.

        Parameters
        ----------
        idx : int

        Returns
        -------
        dict
            ``{"site_id": …, "year": …}``.
        """
        site_id, year = self.index[idx]
        return {"site_id": site_id, "year": int(year)}

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the sample at ``idx``.

        The body reads as six steps:

        1. Look up ``(site_id, year)``.
        2. Slice the last ``seq_length`` days of the (Y-1, Y) concat.
        3. Assemble the feature matrix.
        4. Build the target in the configured mode.
        5. Apply anomaly subtraction or z-scoring.
        6. Wrap in tensors.

        Parameters
        ----------
        idx : int

        Returns
        -------
        features : torch.Tensor
            Shape ``(n_features, seq_length)``.
        target : torch.Tensor
            Shape ``(1, 36)`` or ``(1, 365)``, depending on ``daily_mode``.
        """
        site_id, year = self.index[idx]

        feat_full = self._window(site_id, year)
        pft_vec = self._pft[year][site_id]

        feat_mat = (
            self._assemble_prenorm(feat_full, pft_vec, year)
            if self._prenorm
            else self._assemble_feature_matrix(feat_full, pft_vec, year)
        )

        target = self._build_target(site_id, year)
        target = self._finalise_target(site_id, target)

        feat = torch.from_numpy(feat_mat)
        tgt = torch.from_numpy(target.reshape(1, -1).astype(np.float32))
        return feat, tgt

    @property
    def feature_channels(self) -> int:
        """Number of input channels (``len(ALL_FEATURES)``)."""
        return len(ALL_FEATURES)
