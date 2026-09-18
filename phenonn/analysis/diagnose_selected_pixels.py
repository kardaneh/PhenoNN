#!/usr/bin/env python3
"""
diagnose_selected_pixels.py
===========================

Combined spatial + PFT diagnostics for a `selected_pixels.nc`, merging what
`spatial_plot_selected_pixels.py` (maps + coastlines) and
`diagnose_selected_pixels_pft.py` (PFT composition, per-process HDF5 isolation)
each did. PFT maps vary year to year, so the composition is the MEAN fraction
per cell over a whole DIRECTORY of yearly `PFTmap_{Y}.nc` on [year_start,
year_end], joined to the sites by `site_id`.

Produces (stems derived from --output):
  1. <stem>_dominant_pft.png   world map, one colour per MAJORITY PFT
                               (argmax of the mean fraction) of each cell.
  2. <stem>_pure_gtNN.png      world map of only the cells whose top PFT mean
                               fraction exceeds --pure_threshold (default 0.6),
                               coloured by that PFT.
  3. <stem>_purity_table.{csv,png}
                               per PFT, the number of cells with purity (its
                               mean fraction) above 0.5 / 0.7 / 0.9 / 0.95
                               (also printed to the terminal).
  4. <stem>_density_DEGdeg.png world map split into --block_deg (default 10°)
                               blocks; each block is annotated with the number
                               of selected cells inside it (all PFTs combined).

Every map draws continent coastlines on a PLAIN matplotlib axes (no cartopy
GeoAxes): cartopy is used only to READ the Natural Earth shapefile, then the
coastlines are drawn as plain lon/lat polylines. On .venv_ERA5 a cartopy
GeoAxes aborts savefig ("free(): invalid size") the moment it renders any
VECTOR through its native transform (coastlines, borders, or ax.plot with a
transform) — while raster imshow, the shapefile read, and plain-matplotlib
rendering all work. If the shapefile can't be read the maps fall back to a bare
lon/lat grid with a printed warning.

Native-library isolation (one HDF5 read == one throwaway process)
-----------------------------------------------------------------
On `.venv_ERA5`, netCDF4/HDF5 corrupts the heap a little on each open; over
~28 reads of 1.6 M sites in ONE process the damage crosses a threshold and a
later free() aborts ("free(): invalid size", core dump) — it hit exactly at
the aggregation step even after the reads themselves all succeeded. So NO
process opens more than one .nc file:
  • main process = PLOTTER + AGGREGATOR (numpy/matplotlib/cartopy only, never
    xarray, never HDF5). It accumulates a running mean and plots. Even here NO
    big temporary is built: float32 accumulators + a final divide done in small
    SITE CHUNKS, else the ~0.2 GB single block of `sum/cnt` over (15, 1.6 M)
    aborts the same way ("free(): invalid size") at the aggregation step.
  • meta child = one throwaway subprocess that reads ONLY selected_pixels,
    dumps lon/lat/site_id, and os._exit(0)s.
  • one year child per PFT file = a throwaway subprocess that reads ONLY that
    single PFTmap_{Y}.nc, aligns it to the sites, dumps a compact .npz, and
    os._exit(0)s (skips the HDF5 teardown). The parent folds each year in then
    deletes its .npz, so peak disk is a single year (~0.1 GB), never all 28.

Usage
-----
    python -m phenonn.analysis.diagnose_selected_pixels \\
        --selected_pixels /data/.../selected_pixels_bis.nc \\
        --pft_dir         /data/.../PFT_pixelset \\
        --year_start 1992 --year_end 2019 \\
        --output          analyses/selected_pixels_diag.png
"""

import argparse
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

# Kept local so the reader child never imports phenonn (which pulls torch via the
# package __init__). Mirror of phenonn.utils.config.{PFT_FNAME, N_PFT, PFT_NAMES}.
PFT_FNAME = "PFTmap_{year}.nc"
N_PFT = 15
PFT_NAMES = [
    "bare soil", "trop BL evergreen", "trop BL raingreen", "temp NL evergreen",
    "temp BL evergreen", "temp BL summergreen", "bor NL evergreen",
    "bor BL summergreen", "bor NL summergreen", "temp C3 grass", "C4 grass",
    "C3 agriculture", "C4 agriculture", "trop C3 grass", "bor C3 grass",
]
THRESHOLDS = (0.5, 0.7, 0.9, 0.95)
SITE_CHUNK = 200_000     # sites per slice for the final divide (tiny temporaries)


# ── Reader child (xarray only — no matplotlib/cartopy in this process) ────────


def _pft_array(dp):
    """Return (pft_frac (n_pft, n_site), site_id or None) from a PFT pixelset.
    Same detection as diagnose_selected_pixels_pft._pft_array."""
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
            raise ValueError(f"No PFT variable with a (pft|veget, site) layout "
                             f"in {list(dp.data_vars)}.")
        da = dp[cand[0]]
    pdim = "pft" if "pft" in da.dims else "veget"
    fr = da.transpose(pdim, "site").values.astype(np.float32)
    sid = (np.asarray(dp["site_id"].values).astype(str)
           if "site_id" in dp else None)
    return fr, sid


def _align(fr_src, sid_pft, sid_sel):
    """Reorder a year's (n_pft, n_src) fractions onto the selected sites by
    site_id (NaN where a site is absent)."""
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


# engine="netcdf4" on EVERY open: on .venv_ERA5, letting xarray auto-detect can
# pull in h5netcdf/h5py, whose bundled libhdf5 clashes with netCDF4's in one
# process → heap corruption. Forcing netcdf4 keeps a child on a single libhdf5.


def read_meta(sel_path, npz):
    """Meta child: read ONLY selected_pixels, dump lon/lat/site_id, hard-exit."""
    import xarray as xr

    with xr.open_dataset(sel_path, engine="netcdf4") as ds:
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        sid_sel = np.asarray(ds["site_id"].values).astype(str)
    lon = np.where(lon > 180.0, lon - 360.0, lon)   # 0..360 → -180..180
    np.savez(npz, lon=lon, lat=lat, sid_sel=sid_sel)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def read_year(pft_path, meta_npz, npz):
    """Year child: read ONLY this PFTmap_{Y}.nc, align to the selected sites by
    site_id, dump the (n_pft, n_site) fractions, then hard-exit (skips the HDF5
    teardown). Never opens selected_pixels — takes the site_ids from meta_npz."""
    import xarray as xr

    sid_sel = np.asarray(np.load(meta_npz, allow_pickle=False)["sid_sel"]).astype(str)
    with xr.open_dataset(pft_path, engine="netcdf4", decode_times=False) as dp:
        fr_src, sid_pft = _pft_array(dp)
    fr = _align(fr_src, sid_pft, sid_sel)              # (n_pft, n_site) float32
    np.savez(npz, fr=fr)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


# ── Plotter (main process — no xarray/netCDF4 in scope) ──────────────────────


_COASTS = None


def _load_coastlines():
    """Natural Earth 110m coastlines as a list of (x, y) lon/lat polylines.

    Cartopy is used ONLY to LOCATE + READ the shapefile (that path is fine on
    .venv_ERA5); the lines are then drawn on a PLAIN matplotlib axes. We never
    build a cartopy GeoAxes: its vector transform (native cartopy.trace/GEOS)
    aborts savefig here with "free(): invalid size" — while raster imshow, the
    shapefile read, and plain-matplotlib rendering all work. Cached module-wide."""
    global _COASTS
    if _COASTS is not None:
        return _COASTS
    try:
        import cartopy.io.shapereader as shpreader
        fn = shpreader.natural_earth(resolution="110m", category="physical",
                                     name="coastline")
        polylines = []
        for geom in shpreader.Reader(fn).geometries():
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for ln in parts:
                x, y = ln.xy
                polylines.append((np.asarray(x), np.asarray(y)))
        _COASTS = polylines
        print(f"[coast] {len(polylines)} segments (Natural Earth 110m)")
    except Exception as e:                                    # noqa: BLE001
        print(f"[warn] no coastlines ({type(e).__name__}: {e}); plain grid.")
        _COASTS = []
    return _COASTS


def _new_axes(fig, spec, coasts):
    """A plain lon/lat axes ([-180,180]×[-90,90], equal aspect). Coastlines are
    overlaid AFTER the data (see _draw_coasts); a bare grid is drawn if none."""
    ax = fig.add_subplot(spec)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    if not coasts:
        ax.grid(True, linewidth=0.2, alpha=0.3, linestyle="--")
    return ax


def _draw_coasts(ax, coasts):
    """Overlay coastline polylines (plain ax.plot, no cartopy transform)."""
    for x, y in coasts:
        ax.plot(x, y, color="k", linewidth=0.4)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)


def _dominant_purity(fr_mean):
    """(dominant PFT idx, purity, has_data mask) per cell from the mean frac."""
    fm = np.where(np.isfinite(fr_mean), fr_mean, -np.inf)   # (n_pft, n_site)
    dominant = np.argmax(fm, axis=0)
    purity = np.max(fm, axis=0)
    has = np.isfinite(fr_mean).any(axis=0)
    purity = np.where(has, purity, np.nan)
    return dominant, purity, has


def _pft_colors():
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20", N_PFT)
    return cmap, [cmap(k) for k in range(N_PFT)]


def _legend(ax, colors):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5,
                      markerfacecolor=colors[k], markeredgecolor="none",
                      label=f"{k + 1}. {PFT_NAMES[k]}") for k in range(N_PFT)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=6, frameon=False, title="Dominant PFT")


def _rasterize(lon, lat, values, deg):
    """Bin site values onto a regular (nlat, nlon) grid at `deg` resolution
    (origin upper: row 0 = +90°). NaN where no site falls; last site wins per
    cell. Returns a masked array ready for imshow — this replaces a per-point
    scatter (1.6 M markers), whose cartopy transform aborts savefig with a heap
    overflow ("free(): invalid size"). One image artist = no per-point transform."""
    nlon = int(round(360.0 / deg))
    nlat = int(round(180.0 / deg))
    col = np.clip(((lon + 180.0) / deg).astype(np.int64), 0, nlon - 1)
    row = np.clip(((90.0 - lat) / deg).astype(np.int64), 0, nlat - 1)
    grid = np.full((nlat, nlon), np.nan, dtype=np.float32)
    grid[row, col] = np.asarray(values, dtype=np.float32)
    return np.ma.masked_invalid(grid)


def _imshow_pft(ax, grid, cmap):
    ax.imshow(grid, origin="upper", extent=[-180, 180, -90, 90], aspect="auto",
              cmap=cmap, vmin=-0.5, vmax=N_PFT - 0.5, interpolation="nearest")


def _map_dominant(d, args, coasts):
    import matplotlib.pyplot as plt
    lon, lat = d["lon"], d["lat"]
    dominant, purity, has = _dominant_purity(d["fr_mean"])
    cmap, colors = _pft_colors()
    grid = _rasterize(lon[has], lat[has], dominant[has], args.raster_deg)
    fig = plt.figure(figsize=(13, 6))
    ax = _new_axes(fig, 111, coasts)
    _imshow_pft(ax, grid, cmap)
    _draw_coasts(ax, coasts)
    ax.set_title(f"Dominant PFT per cell ({has.sum():,} cells, "
                 f"mean {d['years'][0]}–{d['years'][-1]})")
    _legend(ax, colors)
    _save(fig, f"{args._stem}_dominant_pft.png", args.dpi)


def _map_pure(d, args, coasts):
    import matplotlib.pyplot as plt
    lon, lat = d["lon"], d["lat"]
    dominant, purity, has = _dominant_purity(d["fr_mean"])
    thr = args.pure_threshold
    m = has & (purity > thr)
    cmap, colors = _pft_colors()
    grid = _rasterize(lon[m], lat[m], dominant[m], args.raster_deg)
    fig = plt.figure(figsize=(13, 6))
    ax = _new_axes(fig, 111, coasts)
    _imshow_pft(ax, grid, cmap)
    _draw_coasts(ax, coasts)
    ax.set_title(f"Cells with PFT fraction > {thr:g}  "
                 f"({m.sum():,} / {has.sum():,} cells)")
    _legend(ax, colors)
    _save(fig, f"{args._stem}_pure_gt{int(round(thr * 100)):02d}.png", args.dpi)


def _purity_table(d, args):
    """Per-PFT cell counts above each threshold → stdout + CSV + PNG."""
    import matplotlib.pyplot as plt
    fr_mean = d["fr_mean"]                                    # (n_pft, n_site)
    counts = np.array([[int(np.nansum(fr_mean[k] > t)) for t in THRESHOLDS]
                       for k in range(N_PFT)], dtype=np.int64)
    totals = counts.sum(axis=0)

    hdr = ["PFT"] + [f">{t:g}" for t in THRESHOLDS]
    lines = ["  ".join(f"{h:>16}" if i == 0 else f"{h:>8}"
                       for i, h in enumerate(hdr))]
    for k in range(N_PFT):
        row = [f"{k + 1}. {PFT_NAMES[k]}"] + [str(c) for c in counts[k]]
        lines.append("  ".join(f"{v:>16}" if i == 0 else f"{v:>8}"
                               for i, v in enumerate(row)))
    lines.append("  ".join(f"{v:>16}" if i == 0 else f"{v:>8}"
                           for i, v in enumerate(["TOTAL"] + [str(t) for t in totals])))
    table_txt = "\n".join(lines)
    print("\n== Pureté : nb de cellules avec fraction PFT > seuil ==")
    print(table_txt)

    csv = f"{args._stem}_purity_table.csv"
    with open(csv, "w") as f:
        f.write(",".join(hdr) + "\n")
        for k in range(N_PFT):
            f.write(",".join([f"{k + 1}. {PFT_NAMES[k]}"] +
                             [str(c) for c in counts[k]]) + "\n")
        f.write(",".join(["TOTAL"] + [str(t) for t in totals]) + "\n")
    print(f"[ok] {csv}")

    # PNG table
    fig, ax = plt.subplots(figsize=(7, 0.35 * (N_PFT + 3)))
    ax.axis("off")
    cell = [[f"{k + 1}. {PFT_NAMES[k]}"] + [f"{c:,}" for c in counts[k]]
            for k in range(N_PFT)]
    cell.append(["TOTAL"] + [f"{t:,}" for t in totals])
    tbl = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
    ax.set_title("Number of cells per PFT with purity > threshold", fontsize=10)
    _save(fig, f"{args._stem}_purity_table.png", args.dpi)


def _map_density(d, args, coasts):
    import matplotlib.pyplot as plt
    lon, lat = d["lon"], d["lat"]
    b = args.block_deg
    xe = np.arange(-180, 180 + b, b)
    ye = np.arange(-90, 90 + b, b)
    H, _, _ = np.histogram2d(lon, lat, bins=[xe, ye])        # (nlon, nlat)
    fig = plt.figure(figsize=(14, 7))
    ax = _new_axes(fig, 111, coasts)
    Hm = np.ma.masked_where(H.T == 0, H.T)
    mesh = ax.pcolormesh(xe, ye, Hm, cmap="YlOrRd", shading="flat")
    fig.colorbar(mesh, ax=ax, shrink=0.6, label="cells / block")
    _draw_coasts(ax, coasts)
    # annotate each non-empty block with its count
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            n = int(H[i, j])
            if n == 0:
                continue
            xc = 0.5 * (xe[i] + xe[i + 1]); yc = 0.5 * (ye[j] + ye[j + 1])
            ax.text(xc, yc, str(n), ha="center", va="center", fontsize=5,
                    color="black")
    ax.set_xticks(xe[::max(1, len(xe) // 12)]); ax.set_yticks(ye)
    ax.set_title(f"Cell density per {b:g}° block  "
                 f"(total {int(H.sum()):,} cells)")
    _save(fig, f"{args._stem}_density_{int(b)}deg.png", args.dpi)


def _save(fig, out_path, dpi):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def render(d, args):
    import matplotlib
    matplotlib.use("Agg")
    coasts = _load_coastlines()
    _map_dominant(d, args, coasts)
    _map_pure(d, args, coasts)
    _purity_table(d, args)
    _map_density(d, args, coasts)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected_pixels", required=True,
                   help="selected_pixels*.nc (site dim: latitude/longitude/site_id).")
    p.add_argument("--pft_dir", required=True,
                   help="Folder of PFTmap_{Y}.nc pixelsets (pft_frac(pft, site)); "
                        "joined to the sites by site_id.")
    p.add_argument("--year_start", type=int, required=True)
    p.add_argument("--year_end", type=int, required=True)
    p.add_argument("--output", default="selected_pixels_diag.png",
                   help="Output stem; the four figures derive their names from it.")
    p.add_argument("--pure_threshold", type=float, default=0.6,
                   help="Fraction threshold for the 'pure' map (default 0.6).")
    p.add_argument("--block_deg", type=float, default=10.0,
                   help="Block size (degrees) for the density map (default 10).")
    p.add_argument("--raster_deg", type=float, default=0.1,
                   help="Display grid resolution (deg) for the PFT maps; sites "
                        "are binned to it and drawn as one image (default 0.1; "
                        "use 0.05 for one pixel per native cell).")
    p.add_argument("--dpi", type=int, default=150)
    # Hidden: pick one throwaway-child role. Each opens at most one .nc file.
    p.add_argument("--_read_meta", default="", help=argparse.SUPPRESS)
    p.add_argument("--_read_year", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--_year_npz", default="", help=argparse.SUPPRESS)
    p.add_argument("--_meta_npz", default="", help=argparse.SUPPRESS)
    args = p.parse_args()
    args._stem = os.path.splitext(args.output)[0]
    return args


def _spawn(args, extra):
    """Run this file again as a throwaway child with the shared args + `extra`."""
    subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "--selected_pixels", args.selected_pixels,
         "--pft_dir", args.pft_dir,
         "--year_start", str(args.year_start),
         "--year_end", str(args.year_end),
         "--output", args.output, *extra],
        check=True)


def main():
    args = parse_args()

    # Throwaway-child roles (each opens ≤ 1 .nc file, then os._exit).
    if args._read_meta:
        read_meta(args.selected_pixels, args._read_meta)
        return                                       # unreachable (os._exit)
    if args._read_year:
        pft_path = os.path.join(args.pft_dir,
                                PFT_FNAME.format(year=args._read_year))
        read_year(pft_path, args._meta_npz, args._year_npz)
        return                                       # unreachable (os._exit)

    # Parent: never opens HDF5. Read meta in a child, then fold in each year via
    # its own child, deleting the .npz as we go (running NaN-aware mean).
    tmpdir = tempfile.mkdtemp(prefix="diag_selpix_")
    meta_npz = os.path.join(tmpdir, "meta.npz")
    try:
        _spawn(args, ["--_read_meta", meta_npz])
        meta = np.load(meta_npz, allow_pickle=False)
        lon, lat, sid_sel = meta["lon"], meta["lat"], meta["sid_sel"]
        n_site = sid_sel.size
        print(f"[read] selected_pixels: {n_site:,} sites", flush=True)

        n_years = args.year_end - args.year_start + 1
        sum_ = cnt = None
        years = []
        for i, y in enumerate(range(args.year_start, args.year_end + 1), start=1):
            if not os.path.exists(os.path.join(args.pft_dir,
                                               PFT_FNAME.format(year=y))):
                print(f"[warn] missing PFTmap_{y}.nc — skipped [{i}/{n_years}]",
                      flush=True)
                continue
            year_npz = os.path.join(tmpdir, f"year_{y}.npz")
            _spawn(args, ["--_read_year", str(y), "--_year_npz", year_npz,
                          "--_meta_npz", meta_npz])
            fr = np.load(year_npz, allow_pickle=False)["fr"]   # (n_pft, n_site)
            os.remove(year_npz)
            if sum_ is None:
                sum_ = np.zeros((fr.shape[0], n_site), dtype=np.float32)
                cnt = np.zeros((fr.shape[0], n_site), dtype=np.float32)
            finite = np.isfinite(fr)
            np.add(sum_, np.where(finite, fr, np.float32(0.0)), out=sum_)
            np.add(cnt, finite, out=cnt, casting="unsafe")
            del fr, finite
            years.append(y)
            print(f"[read] {y}  [{i}/{n_years}]", flush=True)
        if not years:
            raise SystemExit(f"No PFTmap_*.nc in {args.pft_dir} for "
                             f"{args.year_start}..{args.year_end}.")

        # Final mean in small SITE CHUNKS: every temporary is (n_pft, chunk), a
        # few MB — never the ~0.2 GB single block whose big alloc/free tripped the
        # allocator ("free(): invalid size", core dump) at [agg].
        print(f"[agg] averaging {len(years)} years over {n_site:,} sites "
              f"in chunks of {SITE_CHUNK:,}…", flush=True)
        n_pft = sum_.shape[0]
        fr_mean = np.empty((n_pft, n_site), dtype=np.float32)
        for a in range(0, n_site, SITE_CHUNK):
            b = min(a + SITE_CHUNK, n_site)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = sum_[:, a:b] / cnt[:, a:b]
            fr_mean[:, a:b] = np.where(cnt[:, a:b] > 0, m, np.float32(np.nan))
        del sum_, cnt

        d = {"lon": lon, "lat": lat, "fr_mean": fr_mean,
             "years": np.asarray(years, dtype=np.int32)}
        render(d, args)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Reader child failed (exit {e.returncode}) reading the "
                         f".nc files.")
    finally:
        for f in os.listdir(tmpdir) if os.path.isdir(tmpdir) else []:
            os.remove(os.path.join(tmpdir, f))
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
