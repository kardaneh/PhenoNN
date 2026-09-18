#!/usr/bin/env python3
"""
diagnose_grid_alignment.py
==========================

Diagnose the geolocation mismatch between the ERA5-Land pixels and the LAI/PFT
pixels of the SAME site.

Both cubes are built from ONE `selected_pixels.nc` (lat_idx/lon_idx into the
LAI/PFT grid). LAI/PFT are extracted at grid[:, lat_idx, lon_idx]; ERA5 is
extracted at the SAME lat_idx and lon (lon_idx + N_LON//2) % N_LON, ASSUMING the
two grids are the identical 0.1deg / 3600-lon grid up to a 180deg lon origin
shift (see build_daily_dataset_pixelset.py, l.347-353). If that assumption is
wrong (half-pixel registration, different step, reversed latitude order), the
two sources point at different ground locations for the same site_id — that is
the ~0.5deg mismatch to characterise here.

Mode 1 (default, no extra files)
    Compare the `latitude`/`longitude` COORDS that survive in
    ERA5_daily_pixelset_{Y}.nc (the ERA5 grid location actually read, 0..360)
    against the reference `latitude`/`longitude` in selected_pixels.nc (the
    LAI/PFT grid location), joined on site_id. This measures the REALISED
    per-site offset — the thing that actually corrupts training.

Mode 2 (--era5_grid PATH)
    Additionally read the full ERA5 lat/lon vectors from a raw ERA5-Land file,
    print both grids' geometry (origin, step, direction, n), and recompute
    era5_lat[lat_idx], era5_lon[(lon_idx+N_LON//2) % N_LON] to pinpoint WHICH
    part of the assumption breaks (lat order? lon origin? step?).

Read-only, torch-free, scipy-free.

Usage
-----
    python -m plot_diagnositcs.diagnose_grid_alignment \\
        --selected_pixels /data/.../selected_pixels_boxes13.nc \\
        --era5_dir        /data/.../era5_pixelset13 \\
        --year 2000 \\
        [--era5_grid /bdd/ERA5-Land/NETCDF/GLOBAL_01/hourly/FC_SF/.../t2m.200001..nc] \\
        [--max_report 20]
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from phenonn.utils.config import FEATURES_FNAME


def _wrap180(x):
    """Map any longitude convention onto [-180, 180)."""
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def _infer_step(v: np.ndarray):
    """Median spacing of a 1-D coordinate vector (sign = direction)."""
    if v.size < 2:
        return float("nan")
    return float(np.median(np.diff(v)))


def _stats(name: str, d: np.ndarray):
    a = np.abs(d)
    print(f"  {name}: mean={d.mean():+.4f}  median={np.median(d):+.4f}  "
          f"min={d.min():+.4f}  max={d.max():+.4f}  |mean|={a.mean():.4f}  "
          f"|max|={a.max():.4f}  deg")


def _top_offsets(d: np.ndarray, round_to: float = 0.05, top: int = 8):
    """Value-count of deltas rounded to `round_to` — exposes a systematic shift."""
    r = np.round(d / round_to) * round_to
    vals, cnt = np.unique(r, return_counts=True)
    order = np.argsort(cnt)[::-1][:top]
    return [(float(vals[i]), int(cnt[i])) for i in order]


def _load_selected(path: Path):
    with xr.open_dataset(path) as sel:
        return {
            "site_id":   np.asarray(sel["site_id"].values).astype(str),
            "latitude":  np.asarray(sel["latitude"].values, dtype=float),
            "longitude": np.asarray(sel["longitude"].values, dtype=float),
            "lat_idx":   sel["lat_idx"].values.astype(np.int64),
            "lon_idx":   sel["lon_idx"].values.astype(np.int64),
        }


def _load_era5_coords(path: Path):
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        if "latitude" not in ds or "longitude" not in ds:
            raise SystemExit(
                f"{path.name} carries no latitude/longitude(site) coord — cannot "
                f"run mode 1. (Rebuild carries them via .isel, or use --era5_grid.)")
        return {
            "site_id":   np.asarray(ds["site_id"].values).astype(str),
            "latitude":  np.asarray(ds["latitude"].values, dtype=float),
            "longitude": np.asarray(ds["longitude"].values, dtype=float),
        }


def _grid_latlon(path: Path):
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        latn = "latitude" if ("latitude" in ds.coords or "latitude" in ds.dims) else "lat"
        lonn = "longitude" if ("longitude" in ds.coords or "longitude" in ds.dims) else "lon"
        return ds[latn].values.astype(float), ds[lonn].values.astype(float)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected_pixels", required=True)
    p.add_argument("--era5_dir", required=True,
                   help="Folder of ERA5_daily_pixelset_{Y}.nc.")
    p.add_argument("--year", type=int, required=True,
                   help="Which ERA5 pixelset year to read the coords from.")
    p.add_argument("--era5_grid", default="",
                   help="Optional raw ERA5-Land file to also report grid geometry "
                        "and reproduce the index->coord mapping (mode 2).")
    p.add_argument("--max_report", type=int, default=20,
                   help="How many worst-offset sites to list (default 20).")
    args = p.parse_args()

    sel = _load_selected(Path(args.selected_pixels).resolve())
    fpath = Path(args.era5_dir).resolve() / FEATURES_FNAME.format(year=args.year)
    era5 = _load_era5_coords(fpath)

    # ── Join on site_id ──
    pos = {s: i for i, s in enumerate(era5["site_id"])}
    idx = np.array([pos.get(s, -1) for s in sel["site_id"]])
    ok = idx >= 0
    n_tot, n_ok = sel["site_id"].size, int(ok.sum())
    print(f"selected_pixels : {n_tot:,} sites")
    print(f"ERA5 pixelset   : {fpath.name}  ({era5['site_id'].size:,} sites)")
    print(f"matched by id   : {n_ok:,}"
          + ("" if n_ok == n_tot else f"  (! {n_tot - n_ok:,} unmatched)"))
    if n_ok == 0:
        raise SystemExit("No site_id in common — wrong files?")

    ref_lat = sel["latitude"][ok]
    ref_lon = _wrap180(sel["longitude"][ok])
    e_lat = era5["latitude"][idx[ok]]
    e_lon = _wrap180(era5["longitude"][idx[ok]])

    dlat = e_lat - ref_lat
    dlon = _wrap180(e_lon - ref_lon)   # wrap the difference too (dateline safe)

    print("\n=== realised offset  ERA5(stored) - LAI/PFT(ref) ===")
    _stats("d_lat", dlat)
    _stats("d_lon", dlon)
    print("  most common d_lat (rounded 0.05):", _top_offsets(dlat))
    print("  most common d_lon (rounded 0.05):", _top_offsets(dlon))

    mag = np.hypot(dlat, dlon)
    worst = np.argsort(mag)[::-1][:max(0, args.max_report)]
    if worst.size:
        okid = sel["site_id"][ok]
        print(f"\n=== {worst.size} worst-offset sites "
              f"(|hypot(d_lat,d_lon)| deg) ===")
        print(f"  {'site_id':>18}  {'ref_lat':>8} {'ref_lon':>8}  "
              f"{'era_lat':>8} {'era_lon':>8}  {'d_lat':>7} {'d_lon':>7}")
        for j in worst:
            print(f"  {okid[j]:>18}  {ref_lat[j]:8.3f} {ref_lon[j]:8.3f}  "
                  f"{e_lat[j]:8.3f} {e_lon[j]:8.3f}  "
                  f"{dlat[j]:+7.3f} {dlon[j]:+7.3f}")

    # ── Mode 2 : raw ERA5 grid geometry + index reproduction ──
    if args.era5_grid:
        latv, lonv = _grid_latlon(Path(args.era5_grid).resolve())
        print("\n=== ERA5 raw grid geometry ===")
        print(f"  lat: {latv[0]:+.4f} .. {latv[-1]:+.4f}  step={_infer_step(latv):+.4f}"
              f"  n={latv.size}")
        print(f"  lon: {lonv[0]:+.4f} .. {lonv[-1]:+.4f}  step={_infer_step(lonv):+.4f}"
              f"  n={lonv.size}")
        # infer the LAI/PFT grid from the ref per-pixel coords (sparse but telling)
        rlat = np.unique(sel["latitude"])
        rlon = np.unique(_wrap180(sel["longitude"]))
        print("=== LAI/PFT ref coords (from selected_pixels) ===")
        print(f"  lat: {rlat.min():+.4f} .. {rlat.max():+.4f}  "
              f"min|step|={np.min(np.diff(rlat)) if rlat.size > 1 else float('nan'):.4f}")
        print(f"  lon: {rlon.min():+.4f} .. {rlon.max():+.4f}  "
              f"min|step|={np.min(np.diff(rlon)) if rlon.size > 1 else float('nan'):.4f}")

        n_lon = lonv.size
        lon_shift = (sel["lon_idx"] + n_lon // 2) % n_lon
        in_lat = sel["lat_idx"] < latv.size
        in_lon = lon_shift < n_lon
        good = in_lat & in_lon & ok
        if not good.any():
            print("  (! no site with in-range indices for this grid)")
            return
        e2_lat = latv[sel["lat_idx"][good]]
        e2_lon = _wrap180(lonv[lon_shift[good]])
        r2_lat = sel["latitude"][good]
        r2_lon = _wrap180(sel["longitude"][good])
        print("\n=== index->coord reproduced from raw grid  vs  ref ===")
        _stats("d_lat(idx)", e2_lat - r2_lat)
        _stats("d_lon(idx)", _wrap180(e2_lon - r2_lon))
        # does the reproduced coord match what the pixelset stored?
        s2 = era5["latitude"][idx[good]] - e2_lat
        print(f"  stored-vs-reproduced d_lat: |max|={np.abs(s2).max():.4f} "
              f"(0 => same grid+shift as extractor)")


if __name__ == "__main__":
    main()
