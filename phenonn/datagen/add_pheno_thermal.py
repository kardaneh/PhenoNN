# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Add ORCHIDEE-inspired THERMAL phenology features to the ERA5 pixelset files.

All ten columns are cheap daily transforms of the existing ``Tmean`` (°C),
mirroring the temperature-based onset / senescence gates of the ORCHIDEE v4.2
phenology scheme (parameters from the CINECA "carbon-and-nitrogen-cycle" doc):

Onset accumulators (reset each hemisphere's winter solstice)
    ``gdd_0`` / ``gdd_5`` / ``gdd_10``
        Σ max(Tmean − base, 0), base 0 / 5 / 10 °C
    ``ncd_temp``
        # days Tmean < 5 °C (278.15 K, temperate chilling)
    ``ncd_bor``
        # days Tmean < 0 °C (273.15 K, boreal chilling)
    ``ngd``
        # days Tmean > −5 °C (268.15 K, ORCHIDEE NGD; the PFT threshold
        c0 = 23.9 d is left for the NN to learn)
    ``botta_threshold``
        964·exp(−0.0058·ncd_temp) − 12.8 (Botta 2000, eq. 239)
    ``botta_forcing``
        gdd_5 / max(botta_threshold, 1) (onset proximity)

Thermal gates (eq. 237 / 248: rising vs falling temperature)
    ``t_rising``
        mean_7d(Tmean) − mean_30d(Tmean)   (>0 ⇒ Tʷ > Tᵐ)
    ``t_falling``
        mean_14d(Tmean) − mean_30d(Tmean)  (<0 ⇒ T²ʷ < Tᵐ)

Hemisphere-aware annual reset
-----------------------------
The onset accumulators restart at the local mid-winter: 1 Jan in the Northern
hemisphere (lat ≥ 0), 1 Jul in the Southern hemisphere (lat < 0). Because a
Southern-hemisphere cycle spanning Jul(Y−1) → Jun(Y) straddles the file
boundary, each year Y is processed together with the PREVIOUS file (Y−1): the
two ``Tmean`` series are concatenated, the accumulators and trailing means are
computed over the 2-year span, then only year Y is written out. The Y−1 window
also supplies a clean warm-up for the 7/14/30-day means.

If the Y−1 file is missing (first year, or a gap such as 1994/2008) the year is
processed on its own: Northern-hemisphere sites are still correct (reset = start
of file), but Southern-hemisphere Jan–Jun accumulators are truncated — a
warning is printed.

Input / output
--------------
Input  (ERA5 pixelset on either the 0.05 or 0.1 degree route, must already
contain ``Tmean`` and ``latitude``)::

    {era5_dir}/ERA5_daily_pixelset_{Y}.nc     Tmean(time, site), latitude(site), …

Output (same file layout, +10 variables)::

    {out_dir}/ERA5_daily_pixelset_{Y}.nc      … + the 10 features above

``--out_dir`` may equal ``--era5_dir`` to overwrite in place.

Notes
-----
Torch-free. Reads use ``engine="netcdf4"``; the write is isolated in a
subprocess (``_ncwrite``) to dodge the netCDF4 / h5py dual-libhdf5 conflict.

Feeding these features to the model is a separate step: set
``add_pheno_features = True`` in :mod:`phenonn.datagen.config` (which appends
``DERIVED_FEATURES`` to ``DYNAMIC_FEATURES``) and rebuild the stats file.

Examples
--------
::

    python -m phenonn.datagen.add_pheno_thermal \\
        --era5_dir /path/to/data/era5_pixelset \\
        --out_dir  /path/to/data/era5_pheno \\
        --year_start 1992 --year_end 2019
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from phenonn.datagen.config import FEATURES_FNAME
from phenonn.datagen._ncwrite import to_netcdf_subprocess


TMEAN_VAR = "Tmean"  # daily mean 2 m temperature, °C

# ORCHIDEE thresholds expressed in °C (Tmean is stored in °C):
GDD_BASES = (0.0, 5.0, 10.0)  # gdd_0 / gdd_5 / gdd_10
NCD_TEMP_THRESH = 5.0  # 278.15 K — temperate chilling
NCD_BOR_THRESH = 0.0  # 273.15 K — boreal chilling
NGD_THRESH = -5.0  # 268.15 K — ORCHIDEE NGD growing day

# Botta et al. (2000), eq. 239: G_thres = c1·exp(−c2·NCD) − c3
BOTTA_C1 = 964.0
BOTTA_C2 = 0.0058
BOTTA_C3 = 12.8

# Trailing-mean windows (days) for the rising/falling gates (eq. 237 / 248):
WIN_WEEK = 7
WIN_2WEEK = 14
WIN_MONTH = 30

# Variables produced, in the DERIVED_FEATURES order of config.py.
NEW_VARS = [
    "gdd_0",
    "gdd_5",
    "gdd_10",
    "ncd_temp",
    "ncd_bor",
    "ngd",
    "botta_threshold",
    "botta_forcing",
    "t_rising",
    "t_falling",
]


# ── Core numerics (numpy, on a (time, site) Tmean series) ────────────────────


def _reset_masks(time_vals: np.ndarray, n_time: int):
    """
    Boolean masks identifying the NH and SH reset days along the time axis.

    Parameters
    ----------
    time_vals : ndarray
        ``time`` coordinate values of the concatenated (Y−1, Y) span.
    n_time : int
        Length of the time axis.

    Returns
    -------
    reset_nh : ndarray of shape (n_time,)
        True on 1 January.
    reset_sh : ndarray of shape (n_time,)
        True on 1 July.

    Notes
    -----
    Uses real dates when ``time_vals`` decoded to ``datetime64``; otherwise
    falls back to a day-of-year index (doy 1 for NH, doy 183 for SH — an
    approximation for non-datetime files).
    """
    if np.issubdtype(np.asarray(time_vals).dtype, np.datetime64):
        idx = pd.DatetimeIndex(time_vals)
        reset_nh = (idx.month == 1) & (idx.day == 1)
        reset_sh = (idx.month == 7) & (idx.day == 1)
        return np.asarray(reset_nh), np.asarray(reset_sh)
    doy = np.arange(n_time) % 365 + 1
    return doy == 1, doy == 183


def _accumulate_with_reset(daily: np.ndarray, reset: np.ndarray) -> np.ndarray:
    """
    Cumulative sum along time, restarting at each True in ``reset``.

    Parameters
    ----------
    daily : ndarray of shape (n_time, n_site)
        Daily contributions to accumulate.
    reset : ndarray of shape (n_time,)
        Boolean mask of reset days.

    Returns
    -------
    out : ndarray of shape (n_time, n_site)
        Segmented cumulative sum, dtype float64.

    Notes
    -----
    A 2-year window has at most 3 segments, so a simple segmented cumsum is
    sufficient.
    """
    out = np.empty_like(daily, dtype=np.float64)
    starts = list(np.flatnonzero(reset))
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts.append(daily.shape[0])
    for a, b in zip(starts[:-1], starts[1:]):
        out[a:b] = np.cumsum(daily[a:b], axis=0)
    return out


def _accumulate_hemispheric(
    daily: np.ndarray,
    reset_nh: np.ndarray,
    reset_sh: np.ndarray,
    is_nh: np.ndarray,
) -> np.ndarray:
    """
    Reset-aware cumulative sum with NH sites on ``reset_nh``, SH sites on ``reset_sh``.

    Parameters
    ----------
    daily : ndarray of shape (n_time, n_site)
        Daily contributions to accumulate.
    reset_nh : ndarray of shape (n_time,)
        NH reset mask (True on 1 January).
    reset_sh : ndarray of shape (n_time,)
        SH reset mask (True on 1 July).
    is_nh : ndarray of shape (n_site,)
        True for Northern-hemisphere sites (``latitude >= 0``).

    Returns
    -------
    out : ndarray of shape (n_time, n_site)
        Accumulated array, dtype float64.
    """
    out = np.empty_like(daily, dtype=np.float64)
    if is_nh.any():
        out[:, is_nh] = _accumulate_with_reset(daily[:, is_nh], reset_nh)
    if (~is_nh).any():
        out[:, ~is_nh] = _accumulate_with_reset(daily[:, ~is_nh], reset_sh)
    return out


def _trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """
    Causal trailing mean over the last ``w`` steps (inclusive of current).

    Parameters
    ----------
    x : ndarray of shape (n_time, n_site)
        Input time series.
    w : int
        Window length in steps.

    Returns
    -------
    out : ndarray of shape (n_time, n_site)
        Trailing mean, dtype float64. The first ``w - 1`` steps have
        progressively smaller effective windows (``min_periods=1``).
    """
    c = np.cumsum(x, axis=0)
    out = np.empty_like(x, dtype=np.float64)
    out[:w] = c[:w] / np.arange(1, w + 1, dtype=np.float64)[:, None]
    out[w:] = (c[w:] - c[:-w]) / w
    return out


def _compute_features(
    tmean: np.ndarray,
    time_vals: np.ndarray,
    lat_deg: np.ndarray,
) -> dict:
    """
    Compute all 10 thermal phenology features for a ``(time, site)`` Tmean series.

    Parameters
    ----------
    tmean : ndarray of shape (n_time, n_site)
        Daily mean 2 m air temperature in °C.
    time_vals : ndarray
        ``time`` coordinate values of the input series.
    lat_deg : ndarray of shape (n_site,)
        Site latitudes in degrees North.

    Returns
    -------
    feats : dict
        Mapping ``feature_name -> ndarray of shape (n_time, n_site)``, with
        one entry per variable in ``NEW_VARS``.
    """
    n_time = tmean.shape[0]
    reset_nh, reset_sh = _reset_masks(time_vals, n_time)
    is_nh = np.asarray(lat_deg) >= 0.0  # NaN lat → SH (rare edge)

    feats = {}
    for base in GDD_BASES:
        contrib = np.maximum(tmean - base, 0.0)
        feats[f"gdd_{int(base)}"] = _accumulate_hemispheric(
            contrib, reset_nh, reset_sh, is_nh
        )

    feats["ncd_temp"] = _accumulate_hemispheric(
        (tmean < NCD_TEMP_THRESH).astype(np.float64), reset_nh, reset_sh, is_nh
    )
    feats["ncd_bor"] = _accumulate_hemispheric(
        (tmean < NCD_BOR_THRESH).astype(np.float64), reset_nh, reset_sh, is_nh
    )
    feats["ngd"] = _accumulate_hemispheric(
        (tmean > NGD_THRESH).astype(np.float64), reset_nh, reset_sh, is_nh
    )

    feats["botta_threshold"] = (
        BOTTA_C1 * np.exp(-BOTTA_C2 * feats["ncd_temp"]) - BOTTA_C3
    )
    feats["botta_forcing"] = feats["gdd_5"] / np.maximum(feats["botta_threshold"], 1.0)

    m_week = _trailing_mean(tmean, WIN_WEEK)
    m_2week = _trailing_mean(tmean, WIN_2WEEK)
    m_month = _trailing_mean(tmean, WIN_MONTH)
    feats["t_rising"] = m_week - m_month
    feats["t_falling"] = m_2week - m_month
    return feats


# ── Per-year build ───────────────────────────────────────────────────────────


def _load_tmean(fpath: Path):
    """
    Load a pixelset and return its Tmean series and site axis.

    Parameters
    ----------
    fpath : pathlib.Path
        Path to an ``ERA5_daily_pixelset_{Y}.nc`` file.

    Returns
    -------
    tmean : ndarray of shape (n_time, n_site) or None
        Daily Tmean in float64, or ``None`` if ``Tmean`` is missing.
    time_vals : ndarray or None
        ``time`` coordinate values.
    site_vals : ndarray or None
        ``site`` coordinate values.
    ds : xarray.Dataset or None
        The loaded dataset.
    """
    with xr.open_dataset(fpath, engine="netcdf4") as ds:
        ds = ds.load()
    if TMEAN_VAR not in ds:
        return None
    tmean = ds[TMEAN_VAR].transpose("time", "site").values.astype(np.float64)
    return tmean, ds["time"].values, ds["site"].values, ds


def _process_year(
    year: int,
    era5_dir: Path,
    out_dir: Path,
    complevel: int,
) -> None:
    """
    Compute and write the thermal-phenology augmentation for one year.

    Parameters
    ----------
    year : int
        Year to process.
    era5_dir : pathlib.Path
        Folder of ``ERA5_daily_pixelset_{Y}.nc`` files.
    out_dir : pathlib.Path
        Folder where the augmented files are written (may equal ``era5_dir``).
    complevel : int
        zlib compression level.
    """
    fpath = era5_dir / FEATURES_FNAME.format(year=year)
    if not fpath.exists():
        print(f"  ✗ {year} skipped — missing {fpath.name}")
        return

    loaded = _load_tmean(fpath)
    if loaded is None:
        print(f"  ✗ {year} skipped — no '{TMEAN_VAR}' in {fpath.name}")
        return
    tmean_cur, time_cur, site_cur, ds = loaded

    lat_name = (
        "latitude"
        if "latitude" in ds.coords
        else ("lat" if "lat" in ds.coords else None)
    )
    if lat_name is None:
        print(f"  ✗ {year} skipped — no latitude coordinate in {fpath.name}")
        return
    lat_deg = np.asarray(ds[lat_name].values, dtype=np.float64)

    # Prepend the previous year (aligned site axis) for the SH mid-year reset and
    # the trailing-mean warm-up; fall back to single-file if absent/mismatched.
    prev_path = era5_dir / FEATURES_FNAME.format(year=year - 1)
    n_prev = 0
    tmean = tmean_cur
    time_all = time_cur
    if prev_path.exists():
        prev = _load_tmean(prev_path)
        if prev is not None and np.array_equal(prev[2], site_cur):
            tmean = np.concatenate([prev[0], tmean_cur], axis=0)
            time_all = np.concatenate([prev[1], time_cur])
            n_prev = prev[0].shape[0]
        else:
            print(
                f"    ! {year}: {prev_path.name} present but site axis differs "
                f"— processing {year} alone (SH Jan–Jun truncated)"
            )
    else:
        print(
            f"    ! {year}: no {prev_path.name} — processing alone "
            f"(SH Jan–Jun truncated)"
        )

    feats = _compute_features(tmean, time_all, lat_deg)

    for name in NEW_VARS:
        arr = feats[name][n_prev:].astype(np.float32)  # keep only year Y
        ds[name] = (("time", "site"), arr)
    ds["gdd_0"].attrs.update(
        _attr("Σ max(Tmean−0°C,0), hemispheric winter reset", "°C·day")
    )
    ds["gdd_5"].attrs.update(
        _attr("Σ max(Tmean−5°C,0), hemispheric winter reset", "°C·day")
    )
    ds["gdd_10"].attrs.update(
        _attr("Σ max(Tmean−10°C,0), hemispheric winter reset", "°C·day")
    )
    ds["ncd_temp"].attrs.update(
        _attr("# days Tmean<5°C (278.15K), winter reset", "day")
    )
    ds["ncd_bor"].attrs.update(_attr("# days Tmean<0°C (273.15K), winter reset", "day"))
    ds["ngd"].attrs.update(_attr("# days Tmean>−5°C (268.15K), ORCHIDEE NGD", "day"))
    ds["botta_threshold"].attrs.update(
        _attr("964·exp(−0.0058·ncd_temp)−12.8 (Botta 2000)", "°C·day")
    )
    ds["botta_forcing"].attrs.update(
        _attr("gdd_5 / max(botta_threshold,1), onset proximity", "1")
    )
    ds["t_rising"].attrs.update(_attr("mean_7d(Tmean)−mean_30d(Tmean) (eq.237)", "°C"))
    ds["t_falling"].attrs.update(
        _attr("mean_14d(Tmean)−mean_30d(Tmean) (eq.248)", "°C")
    )

    # Fresh encoding per var (drop source filter hints for the subprocess pickle)
    # but PRESERVE the source chunking (full time × block of sites) so the
    # per-site training read still hits few chunks. Same note as add_daylength.py.
    ref_chunks = ds[TMEAN_VAR].encoding.get("chunksizes")
    encoding = {}
    for name, var in ds.data_vars.items():
        orig_chunks = var.encoding.get("chunksizes")
        var.encoding = {}
        enc = {"zlib": True, "complevel": complevel}
        if orig_chunks is not None:
            enc["chunksizes"] = tuple(orig_chunks)
        encoding[name] = enc
    for name in NEW_VARS:
        if ref_chunks is not None:
            encoding[name]["chunksizes"] = tuple(ref_chunks)
        encoding[name]["_FillValue"] = np.float32(np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    to_netcdf_subprocess(
        ds,
        out_dir / FEATURES_FNAME.format(year=year),
        encoding=encoding,
        engine="netcdf4",
    )
    print(f"  ✓ {year}  (+{len(NEW_VARS)} thermal feats, {tmean_cur.shape[1]:,} sites)")


def _attr(formula: str, units: str) -> dict:
    """
    Build a standard attribute dictionary for a new variable.

    Parameters
    ----------
    formula : str
        Human-readable formula for ``long_name``.
    units : str
        Physical units of the variable.

    Returns
    -------
    attrs : dict
        Dictionary with keys ``"long_name"``, ``"units"`` and ``"source"``.
    """
    return {"long_name": formula, "units": units, "source": "add_pheno_thermal.py"}


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes ``era5_dir``, ``out_dir``,
        ``year_start``, ``year_end``, ``complevel``.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--era5_dir",
        required=True,
        help="Folder of ERA5_daily_pixelset_{Y}.nc (must contain Tmean).",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Where to write the augmented files "
        "(may equal --era5_dir to overwrite in place).",
    )
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--complevel", type=int, default=4)
    return p.parse_args()


def main():
    """
    Run the thermal-phenology augmentation over a year range.

    Iterates ``range(args.year_start, args.year_end + 1)`` calling
    :func:`_process_year`; per-year failures are caught, logged, and do not
    abort the run.
    """
    args = parse_args()
    era5_dir = Path(args.era5_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"ERA5 pixelset in : {era5_dir}")
    print(f"Augmented out    : {out_dir}")
    print(f"Years            : {args.year_start} → {args.year_end}")
    print(f"Adding           : {', '.join(NEW_VARS)}\n")

    for year in range(args.year_start, args.year_end + 1):
        try:
            _process_year(year, era5_dir, out_dir, args.complevel)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {year} failed — {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
