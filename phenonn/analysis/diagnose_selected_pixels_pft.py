#!/usr/bin/env python3
"""
diagnose_selected_pixels_pft.py
===============================

Multi-year diagnostics on a `selected_pixels.nc` file and the PFT composition of
its sites. PFT maps change from year to year, so this reads a whole DIRECTORY of
yearly PFT pixelset files (`PFTmap_{Y}.nc`) over [--year_start, --year_end].

Prints
------
  • Per YEAR: how many selected sites have each PFT fraction above 0.5 / 0.7 / 0.9.
  • Metric 1 — interannual turnover: for consecutive years, the per-pixel
    L1 change Σ_i |p_{t,i} − p_{t-1,i}| of the 15 fractions, averaged over pixels;
    printed for every year pair (raw 0..2 and the ½ total-variation 0..1).
  • Metric 4 — temporal instability (per cell), summary stats + distribution.

Plots (4 PNGs, derived from --output's stem)
--------------------------------------------
  • <stem>.png                    : 15-panel world map, one per PFT, of the MEAN
                                    fraction over all years (> --map_threshold).
  • <stem>_turnover_map.png       : metric 1 per cell, meaned over all year pairs.
  • <stem>_instability_hist.png   : histogram (cells per value) of metric 4.
  • <stem>_instability_map.png    : metric 4 spatial distribution.

Metric 4 — global temporal stability (per cell over T years)
------------------------------------------------------------
    p̄_i = (1/T) Σ_t p_{t,i}                              (mean fraction over years)
    Instability = (1/T) Σ_t ( ½ Σ_{i=1}^{15} |p_{t,i} − p̄_i| )
0 = perfectly stable in time; →1 = fractions swing wildly around their mean.

Inputs
------
`selected_pixels.nc` carries only the site geometry (latitude/longitude/site_id),
NOT the fractions — so you also pass --pft_dir (folder of PFTmap_{Y}.nc pixelsets
with `pft_frac(pft, site)`, from build_pixelset_targets). Each year is joined to
the sites by `site_id`.

Native-library isolation (important — read this if it crashes)
--------------------------------------------------------------
On the `.venv_ERA5` interpreter, netCDF4/HDF5 corrupts the process heap: a free()
after an HDF5 read can core-dump, and cartopy's GEOS/shapely stack makes it worse.
So the two never share a process (same pattern as spatial_plot_selected_pixels.py):

  • main process = the PLOTTER: numpy/matplotlib/cartopy only, never xarray.
  • reader child = a throwaway subprocess that imports ONLY xarray, reads every
    year, does the numpy aggregation, dumps compact arrays to a .npz, and
    os._exit(0)s to skip the HDF5 teardown that core-dumps.

Coastlines need **cartopy**; without it the maps degrade to a plain lon/lat grid.

Usage
-----
    python -m plot_diagnositcs.diagnose_selected_pixels_pft \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels.nc \\
        --pft_dir         /data/sbarbu/PhenoNN/data/PFT_pixelset \\
        --year_start 1992 --year_end 2018 \\
        --output          selected_pixels_pft_diag.png
"""

import argparse
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

PFT_FNAME = "PFTmap_{year}.nc"        # == phenonn.utils.config.PFT_FNAME (kept local)
THRESHOLDS = (0.5, 0.7, 0.9)


# ── Reader child (xarray only — no matplotlib/cartopy in this process) ────────


def _pft_array(dp):
    """Return (pft_frac (n_pft, n_site), site_id or None) from a PFT pixelset."""
    if "pft_frac" in dp.data_vars:
        da = dp["pft_frac"]
    elif "maxvegetfrac" in dp.data_vars:
        da = dp["maxvegetfrac"]
        if "time_counter" in da.dims:
            da = da.isel(time_counter=0, drop=True)
    else:
        cand = [v for v in dp.data_vars
                if ("pft" in dp[v].dims or "veget" in dp[v].dims)
                and "site" in dp[v].dims]
        if not cand:
            raise ValueError(
                f"No PFT variable with a (pft|veget, site) layout in "
                f"{list(dp.data_vars)}.")
        da = dp[cand[0]]
    pdim = "pft" if "pft" in da.dims else "veget"
    fr = da.transpose(pdim, "site").values.astype(np.float32)
    sid = (np.asarray(dp["site_id"].values).astype(str)
           if "site_id" in dp else None)
    return fr, sid


def _align(fr_src, sid_pft, sid_sel):
    """Reorder a year's (n_pft, n_src) fractions onto the selected sites,
    joining by site_id (NaN where a site is absent)."""
    n_pft = fr_src.shape[0]
    out = np.full((n_pft, sid_sel.size), np.nan, dtype=np.float32)
    if sid_pft is not None:
        col = {s: i for i, s in enumerate(sid_pft)}
        for j, s in enumerate(sid_sel):
            i = col.get(s)
            if i is not None:
                out[:, j] = fr_src[:, i]
    else:
        if fr_src.shape[1] != sid_sel.size:
            raise ValueError(
                f"PFT file has no site_id and its site count ({fr_src.shape[1]}) "
                f"!= selected sites ({sid_sel.size}); cannot align.")
        out = fr_src
    return out


def read_and_dump(sel_path, pft_dir, year_start, year_end, npz):
    """Read selected_pixels + every yearly PFT file, aggregate with numpy, dump
    compact arrays, hard-exit. os._exit(0) skips the HDF5 teardown that
    core-dumps; numpy work before it is safe."""
    import xarray as xr

    with xr.open_dataset(sel_path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        sid_sel = np.asarray(ds["site_id"].values).astype(str)
    n_site = sid_sel.size

    years, frs = [], []
    for y in range(year_start, year_end + 1):
        p = os.path.join(pft_dir, PFT_FNAME.format(year=y))
        if not os.path.exists(p):
            sys.stderr.write(f"[warn] missing {os.path.basename(p)} — skipped\n")
            continue
        with xr.open_dataset(p, decode_times=False) as dp:
            fr_src, sid_pft = _pft_array(dp)
        frs.append(_align(fr_src, sid_pft, sid_sel))
        years.append(y)
    if not frs:
        raise SystemExit(f"No PFTmap_*.nc found in {pft_dir} for "
                         f"{year_start}..{year_end}.")

    A = np.stack(frs, axis=0)                       # (T, n_pft, n_site) float32
    years = np.asarray(years, dtype=np.int32)
    T, n_pft, _ = A.shape
    present = np.isfinite(A).any(axis=1)            # (T, n_site)
    n_present = present.sum(axis=0)                 # (n_site,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices
        # Per-year threshold counts over sites (nan > t → False).
        counts = np.stack(
            [(A > t).sum(axis=2) for t in THRESHOLDS], axis=2)  # (T, n_pft, 3)

        # Mean fraction over years (ignoring absent years).
        fr_mean = np.nanmean(A, axis=0)            # (n_pft, n_site)

        # Metric 4 — temporal instability per cell.
        absdev = np.abs(A - fr_mean[None, :, :])   # (T, n_pft, n_site)
        d = 0.5 * np.nansum(absdev, axis=1)        # (T, n_site)  ½ Σ_i |p_t-p̄|
        d = np.where(present, d, np.nan)
        instab = np.where(n_present >= 2, np.nanmean(d, axis=0), np.nan)

        # Metric 1 — interannual turnover per cell + per year pair.
        if T >= 2:
            dA = np.abs(A[1:] - A[:-1])            # (T-1, n_pft, n_site)
            pair = present[1:] & present[:-1]      # (T-1, n_site)
            half = np.where(pair, 0.5 * np.nansum(dA, axis=1), np.nan)  # (T-1, n_site)
            turnover_cell = np.where(pair.sum(axis=0) >= 1,
                                     np.nanmean(half, axis=0), np.nan)
            turnover_year = np.array(
                [np.nanmean(half[t]) if pair[t].any() else np.nan
                 for t in range(T - 1)], dtype=np.float32)
        else:
            turnover_cell = np.full(n_site, np.nan, dtype=np.float32)
            turnover_year = np.zeros(0, dtype=np.float32)

    lon = np.where(lon > 180.0, lon - 360.0, lon)  # 0..360 → -180..180
    np.savez(npz,
             lon=lon, lat=lat, years=years, n_site=int(n_site),
             n_present=n_present.astype(np.int32),
             counts=counts.astype(np.int64),
             fr_mean=fr_mean.astype(np.float32),
             instab=instab.astype(np.float32),
             turnover_cell=turnover_cell.astype(np.float32),
             turnover_year=turnover_year.astype(np.float32))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ── Plotter (main process — no xarray/netCDF4 in scope) ──────────────────────


def _pft_names(n_pft):
    """PFT labels from phenonn.utils.config with a generic fallback (main process only,
    so the reader child never pulls phenon in)."""
    try:
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from phenonn.utils.config import PFT_NAMES
        names = list(PFT_NAMES)
    except Exception:                              # noqa: BLE001
        names = []
    names += [""] * max(0, n_pft - len(names))
    return names[:n_pft]


def _print_reports(d, names):
    lon, lat = d["lon"], d["lat"]
    years = d["years"]
    counts = d["counts"]              # (T, n_pft, 3)
    instab = d["instab"]
    turnover_year = d["turnover_year"]
    n_site = int(d["n_site"])
    n_present = d["n_present"]
    n_pft = counts.shape[1]

    print(f"{n_site:,} selected sites  |  {len(years)} years "
          f"[{years.min()}..{years.max()}]  |  "
          f"lat [{lat.min():.1f}, {lat.max():.1f}]  "
          f"lon [{lon.min():.1f}, {lon.max():.1f}]")

    # ── Per-year threshold counts ──
    print("\n" + "=" * 62)
    print("PER-YEAR PFT SITE COUNTS ABOVE A FRACTION THRESHOLD")
    print("=" * 62)
    for t, y in enumerate(years):
        print(f"\n--- {y} ---")
        print(f"  {'idx':>3}  {'PFT name':<22}{'>0.50':>8}{'>0.70':>8}{'>0.90':>8}")
        print("  " + "-" * 49)
        for k in range(n_pft):
            c = counts[t, k]
            print(f"  {k + 1:>3}  {names[k]:<22}"
                  f"{int(c[0]):>8,}{int(c[1]):>8,}{int(c[2]):>8,}")

    # ── Metric 1: interannual turnover ──
    print("\n" + "=" * 62)
    print("METRIC 1 — INTERANNUAL TURNOVER  (mean over pixels of Σ_i|Δp_i|)")
    print("=" * 62)
    if turnover_year.size:
        print(f"  {'year pair':<14}{'raw Σ|Δ| (0..2)':>18}{'½·Σ|Δ| (0..1)':>16}")
        print("  " + "-" * 46)
        for t in range(turnover_year.size):
            half = float(turnover_year[t])
            print(f"  {years[t]}→{years[t + 1]:<8}"
                  f"{2 * half:>18.4f}{half:>16.4f}")
        good = turnover_year[np.isfinite(turnover_year)]
        if good.size:
            print("  " + "-" * 46)
            print(f"  {'mean all pairs':<14}"
                  f"{2 * good.mean():>18.4f}{good.mean():>16.4f}")
    else:
        print("  (needs ≥ 2 years)")

    # ── Metric 4: temporal instability ──
    print("\n" + "=" * 62)
    print("METRIC 4 — TEMPORAL INSTABILITY  (½·Σ_i|p_t-p̄_i| meaned over years)")
    print("=" * 62)
    fin = instab[np.isfinite(instab)]
    n_ge2 = int(np.sum(n_present >= 2))
    print(f"  cells scored (≥2 valid years): {fin.size:,} / {n_site:,}  "
          f"(≥2-year cells: {n_ge2:,})")
    if fin.size:
        print(f"  mean={fin.mean():.4f}  median={np.median(fin):.4f}  "
              f"p90={np.percentile(fin, 90):.4f}  max={fin.max():.4f}")
        edges = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
        print("  distribution (cells per instability band):")
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = int(np.sum((fin >= lo) & (fin < hi)))
            print(f"    [{lo:.2f}, {hi if hi <= 1 else 1.0:.2f}) : {n:,}")


# ── Map helpers ───────────────────────────────────────────────────────────────


def _new_axes(fig, spec, have_cartopy, ccrs=None, cfeature=None):
    """Add one GeoAxes (or plain axes) at subplot `spec` and draw the frame."""
    if have_cartopy:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, alpha=0.3)
        return ax, {"transform": ccrs.PlateCarree()}
    ax = fig.add_subplot(spec)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.2, alpha=0.3, linestyle="--")
    return ax, {}


def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return True, ccrs, cfeature
    except ImportError:
        print("[warn] cartopy not installed — no coastlines "
              "(`pip install cartopy`). Plain lon/lat grids.")
        return False, None, None


def _plot_composition(d, names, args, have_cartopy, ccrs, cfeature):
    import matplotlib.pyplot as plt
    lon, lat, fr_mean = d["lon"], d["lat"], d["fr_mean"]
    years = d["years"]
    n_pft = fr_mean.shape[0]
    thr = args.map_threshold

    ncols = 3
    nrows = (n_pft + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 6.0, nrows * 3.2))
    gs = fig.add_gridspec(nrows, ncols)
    last_sc = None
    for k in range(n_pft):
        ax, tkw = _new_axes(fig, gs[k // ncols, k % ncols],
                            have_cartopy, ccrs, cfeature)
        ax.scatter(lon, lat, s=0.5, c="0.85", edgecolors="none", zorder=2, **tkw)
        m = fr_mean[k] > thr
        if m.any():
            last_sc = ax.scatter(lon[m], lat[m], s=args.markersize,
                                 c=fr_mean[k][m], cmap="viridis",
                                 vmin=thr, vmax=1.0, edgecolors="none",
                                 zorder=3, **tkw)
        ax.set_title(f"PFT{k + 1} {names[k]}  (n>{thr:g} = {int(m.sum()):,})",
                     fontsize=8)
    if last_sc is not None:
        cb = fig.colorbar(last_sc, ax=fig.axes, shrink=0.6, fraction=0.02, pad=0.02)
        cb.set_label(f"mean PFT fraction (> {thr:g})")
    fig.suptitle(f"{os.path.basename(args.selected_pixels)} — mean PFT "
                 f"composition over {years.min()}..{years.max()}", fontsize=11)
    _save(fig, args.output, args.dpi)


def _plot_scalar_map(d, val, title, cmap, out_path, args,
                     have_cartopy, ccrs, cfeature, vmax=None):
    import matplotlib.pyplot as plt
    lon, lat = d["lon"], d["lat"]
    fin = np.isfinite(val)
    if vmax is None:
        vmax = float(np.percentile(val[fin], 98)) if fin.any() else 1.0
    vmax = max(vmax, 1e-6)
    fig = plt.figure(figsize=(14, 7))
    ax, tkw = _new_axes(fig, fig.add_gridspec(1, 1)[0, 0],
                        have_cartopy, ccrs, cfeature)
    ax.scatter(lon[~fin], lat[~fin], s=0.5, c="0.85", edgecolors="none",
               zorder=2, **tkw)
    sc = ax.scatter(lon[fin], lat[fin], s=args.markersize, c=val[fin],
                    cmap=cmap, vmin=0.0, vmax=vmax, edgecolors="none",
                    zorder=3, **tkw)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(title)
    ax.set_title(f"{os.path.basename(args.selected_pixels)} — {title}")
    _save(fig, out_path, args.dpi)


def _plot_hist(val, title, out_path, args):
    import matplotlib.pyplot as plt
    fin = val[np.isfinite(val)]
    fig, ax = plt.subplots(figsize=(9, 5))
    if fin.size:
        ax.hist(fin, bins=60, color="#c0392b", alpha=0.85)
        ax.axvline(fin.mean(), color="k", lw=1, ls="--",
                   label=f"mean={fin.mean():.3f}")
        ax.axvline(np.median(fin), color="0.4", lw=1, ls=":",
                   label=f"median={np.median(fin):.3f}")
        ax.legend()
    ax.set_xlabel(title)
    ax.set_ylabel("number of cells")
    ax.set_title(title)
    _save(fig, out_path, args.dpi)


def _save(fig, out_path, dpi):
    import matplotlib.pyplot as plt
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def render(npz, args):
    import matplotlib
    matplotlib.use("Agg")                 # headless server (save to file)

    d = np.load(npz)
    n_pft = d["fr_mean"].shape[0]
    names = _pft_names(n_pft)

    _print_reports(d, names)

    have_cartopy, ccrs, cfeature = _try_cartopy()
    stem = os.path.splitext(args.output)[0]

    _plot_composition(d, names, args, have_cartopy, ccrs, cfeature)
    if d["turnover_cell"].size and np.isfinite(d["turnover_cell"]).any():
        _plot_scalar_map(d, d["turnover_cell"],
                         "interannual turnover (½·Σ|Δp|, mean over year pairs)",
                         "magma", stem + "_turnover_map.png", args,
                         have_cartopy, ccrs, cfeature)
    if np.isfinite(d["instab"]).any():
        _plot_hist(d["instab"], "temporal instability (metric 4)",
                   stem + "_instability_hist.png", args)
        _plot_scalar_map(d, d["instab"], "temporal instability (metric 4)",
                         "magma", stem + "_instability_map.png", args,
                         have_cartopy, ccrs, cfeature)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected_pixels", required=True,
                   help="selected_pixels*.nc (site dim: latitude/longitude/site_id).")
    p.add_argument("--pft_dir", required=True,
                   help="Folder of yearly PFT pixelset files PFTmap_{Y}.nc "
                        "(pft_frac(pft, site)); joined to the sites by site_id.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end",   type=int, required=True)
    p.add_argument("--output", default="selected_pixels_pft_diag.png",
                   help="Composition-map PNG; the turnover/instability PNGs "
                        "reuse its stem.")
    p.add_argument("--map_threshold", type=float, default=0.5,
                   help="Only map sites whose MEAN PFT fraction exceeds this "
                        "(default 0.5). The printed table always uses 0.5/0.7/0.9.")
    p.add_argument("--markersize", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=150)
    # Internal: set on the reader child so it dumps the .npz and exits.
    p.add_argument("--_read_npz", default="", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Reader child: aggregate + dump, then hard-exit ──
    if args._read_npz:
        read_and_dump(args.selected_pixels, args.pft_dir,
                      args.year_start, args.year_end, args._read_npz)
        return                              # unreachable (os._exit above)

    # ── Main / plotter: spawn the reader, then draw (no netCDF4 here) ──
    fd, npz = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--selected_pixels", args.selected_pixels,
             "--pft_dir", args.pft_dir,
             "--year_start", str(args.year_start),
             "--year_end", str(args.year_end),
             "--_read_npz", npz],
            check=True,
        )
        if not os.path.exists(npz) or os.path.getsize(npz) == 0:
            raise SystemExit("Reader child produced no data — check the paths.")
        render(npz, args)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"Reader child failed (exit {e.returncode}) while reading the .nc "
            f"files. If it core-dumped in HDF5, try engine=\"h5netcdf\" in "
            f"read_and_dump.")
    finally:
        if os.path.exists(npz):
            os.remove(npz)


if __name__ == "__main__":
    main()
