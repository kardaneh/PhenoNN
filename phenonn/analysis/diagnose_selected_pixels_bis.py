#!/usr/bin/env python3
"""
diagnose_selected_pixels.py
===========================

Combined spatial + PFT diagnostics for a `selected_pixels.nc`, merging what
`spatial_plot_selected_pixels.py` (maps + coastlines) and
`diagnose_selected_pixels_pft.py` (PFT composition, per-process HDF5 isolation)
each did.

PFT maps vary year to year, so the composition is the MEAN fraction
per cell over a whole DIRECTORY of yearly PFTmap_{Y}.nc on
[year_start, year_end], joined to the sites by site_id.

Produces (stems derived from --output):
  1. <stem>_dominant_pft.png
       world map, one colour per MAJORITY PFT
       (argmax of the mean fraction) of each cell.

  2. <stem>_pure_gtNN.png
       world map of only the cells whose top PFT mean fraction exceeds
       --pure_threshold (default 0.6), coloured by that PFT.

  3. <stem>_purity_table.{csv,png}
       per PFT, the number of cells with purity
       above 0.5 / 0.7 / 0.9 / 0.95.

  4. <stem>_density_DEGdeg.png
       world map split into --block_deg (default 10°) blocks;
       each block is annotated with the number of selected cells inside it.

Native-library isolation
------------------------

On `.venv_ERA5`, netCDF4/HDF5 can corrupt the heap after many reads in the
same process. Therefore:

  * main process = plotting + aggregation only
  * meta child = reads selected_pixels.nc only
  * one child per PFTmap_{Y}.nc
  * every reader child opens at most ONE .nc file and exits with os._exit(0)

For a single year, no aggregation is performed:
the yearly PFT fractions are directly used as the mean.

Usage
-----

python diagnose_selected_pixels.py \
    --selected_pixels /data/.../selected_pixels.nc \
    --pft_dir /data/.../PFT_pixelset \
    --year_start 1992 \
    --year_end 2019 \
    --output analysis/selected_pixels_diag.png
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PFT_FNAME = "PFTmap_{year}.nc"
N_PFT = 15

PFT_NAMES = [
    "bare soil",
    "trop BL evergreen",
    "trop BL raingreen",
    "temp NL evergreen",
    "temp BL evergreen",
    "temp BL summergreen",
    "bor NL evergreen",
    "bor BL summergreen",
    "bor NL summergreen",
    "temp C3 grass",
    "C4 grass",
    "C3 agriculture",
    "C4 agriculture",
    "trop C3 grass",
    "bor C3 grass",
]

THRESHOLDS = (0.5, 0.7, 0.9, 0.95)

# Number of sites processed at once during final multi-year division.
SITE_CHUNK = 200_000


# ===========================================================================
# Reader child
# ===========================================================================

def _pft_array(dp):
    """
    Return:

        fr       : (n_pft, n_site) float32
        site_id  : array or None

    from a PFT pixelset.
    """

    if "pft_frac" in dp.data_vars:
        da = dp["pft_frac"]

    elif "maxvegetfrac" in dp.data_vars:
        da = dp["maxvegetfrac"]

        if "time_counter" in da.dims:
            da = da.isel(
                time_counter=0,
                drop=True
            )

    else:
        cand = [
            v
            for v in dp.data_vars
            if (
                ("pft" in dp[v].dims or "veget" in dp[v].dims)
                and "site" in dp[v].dims
            )
        ]

        if not cand:
            raise ValueError(
                "No PFT variable with a (pft|veget, site) layout "
                f"in {list(dp.data_vars)}."
            )

        da = dp[cand[0]]

    pdim = "pft" if "pft" in da.dims else "veget"

    fr = da.transpose(
        pdim,
        "site"
    ).values.astype(
        np.float32
    )

    if "site_id" in dp:
        sid = np.asarray(
            dp["site_id"].values
        ).astype(str)
    else:
        sid = None

    return fr, sid


def _align(fr_src, sid_pft, sid_sel):
    """
    Reorder a year's fractions onto the selected sites by site_id.

    Output:
        (n_pft, n_selected_sites)

    NaN is used when a selected site is absent from the PFT file.
    """

    n_pft = fr_src.shape[0]

    out = np.full(
        (n_pft, sid_sel.size),
        np.nan,
        dtype=np.float32
    )

    if sid_pft is not None:

        col = {
            s: i
            for i, s in enumerate(sid_pft)
        }

        for j, s in enumerate(sid_sel):

            i = col.get(s)

            if i is not None:
                out[:, j] = fr_src[:, i]

    else:

        if fr_src.shape[1] != sid_sel.size:
            raise ValueError(
                f"PFT file has no site_id and its site count "
                f"({fr_src.shape[1]}) != selected sites "
                f"({sid_sel.size}); cannot align."
            )

        out = fr_src

    return out


# ---------------------------------------------------------------------------
# IMPORTANT:
# Always use netcdf4 explicitly.
#
# This prevents xarray from selecting h5netcdf/h5py and potentially loading
# another libhdf5 into the same child process.
# ---------------------------------------------------------------------------

def read_meta(sel_path, npz):
    """
    Read ONLY selected_pixels.nc.

    Dump:
        lon
        lat
        site_id

    and hard-exit to avoid HDF5 teardown.
    """

    import xarray as xr

    with xr.open_dataset(
        sel_path,
        engine="netcdf4"
    ) as ds:

        lat = np.asarray(
            ds["latitude"].values,
            dtype=float
        )

        lon = np.asarray(
            ds["longitude"].values,
            dtype=float
        )

        sid_sel = np.asarray(
            ds["site_id"].values
        ).astype(str)

    # Convert 0..360 longitude to -180..180.
    lon = np.where(
        lon > 180.0,
        lon - 360.0,
        lon
    )

    np.savez(
        npz,
        lon=lon,
        lat=lat,
        sid_sel=sid_sel
    )

    sys.stdout.flush()
    sys.stderr.flush()

    # Critical: skip HDF5/netCDF4 teardown.
    os._exit(0)


def read_year(pft_path, meta_npz, npz):
    """
    Read ONLY one PFTmap_{Y}.nc.

    Align its sites to the selected pixels and dump a compact NPZ.

    Then hard-exit to skip HDF5 teardown.
    """

    import xarray as xr

    meta = np.load(
        meta_npz,
        allow_pickle=False
    )

    sid_sel = np.asarray(
        meta["sid_sel"]
    ).astype(str)

    del meta

    with xr.open_dataset(
        pft_path,
        engine="netcdf4",
        decode_times=False
    ) as dp:

        fr_src, sid_pft = _pft_array(dp)

    fr = _align(
        fr_src,
        sid_pft,
        sid_sel
    )

    np.savez(
        npz,
        fr=fr
    )

    sys.stdout.flush()
    sys.stderr.flush()

    # Critical: skip HDF5/netCDF4 teardown.
    os._exit(0)


# ===========================================================================
# Plotting process
# ===========================================================================

def _try_cartopy():

    try:

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return True, ccrs, cfeature

    except ImportError:

        print(
            "[warn] cartopy not installed — no coastlines "
            "(`pip install cartopy`). Plain lon/lat grids."
        )

        return False, None, None


def _new_axes(
    fig,
    spec,
    have_cartopy,
    ccrs=None,
    cfeature=None
):
    """
    Create a GeoAxes or regular lon/lat axes.
    """

    if have_cartopy:

        ax = fig.add_subplot(
            spec,
            projection=ccrs.PlateCarree()
        )

        ax.set_global()

        ax.coastlines(
            linewidth=0.4
        )

        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.15,
            alpha=0.3
        )

        return ax, {
            "transform": ccrs.PlateCarree()
        }

    ax = fig.add_subplot(spec)

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        -90,
        90
    )

    ax.set_aspect("equal")

    ax.grid(
        True,
        linewidth=0.2,
        alpha=0.3,
        linestyle="--"
    )

    return ax, {}


def _dominant_purity(fr_mean):
    """
    Compute dominant PFT and purity for each cell.

    Input:
        (n_pft, n_site)

    Returns:
        dominant
        purity
        has_data
    """

    fm = np.where(
        np.isfinite(fr_mean),
        fr_mean,
        -np.inf
    )

    dominant = np.argmax(
        fm,
        axis=0
    )

    purity = np.max(
        fm,
        axis=0
    )

    has = np.isfinite(
        fr_mean
    ).any(
        axis=0
    )

    purity = np.where(
        has,
        purity,
        np.nan
    )

    return dominant, purity, has


def _pft_colors():

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(
        "tab20",
        N_PFT
    )

    return cmap, [
        cmap(k)
        for k in range(N_PFT)
    ]


def _legend(
    ax,
    colors
):

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=colors[k],
            markeredgecolor="none",
            label=f"{k + 1}. {PFT_NAMES[k]}"
        )
        for k in range(N_PFT)
    ]

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=6,
        frameon=False,
        title="PFT (majoritaire)"
    )


def _map_dominant(
    d,
    args,
    hc,
    ccrs,
    cfeature
):

    import matplotlib.pyplot as plt

    lon = d["lon"]
    lat = d["lat"]

    dominant, purity, has = _dominant_purity(
        d["fr_mean"]
    )

    cmap, colors = _pft_colors()

    fig = plt.figure(
        figsize=(13, 6)
    )

    ax, kw = _new_axes(
        fig,
        111,
        hc,
        ccrs,
        cfeature
    )

    ax.scatter(
        lon[has],
        lat[has],
        c=dominant[has],
        cmap=cmap,
        vmin=-0.5,
        vmax=N_PFT - 0.5,
        s=args.markersize,
        linewidths=0,
        **kw
    )

    ax.set_title(
        f"PFT majoritaire par cellule "
        f"({has.sum():,} cellules, "
        f"moyenne {d['years'][0]}–{d['years'][-1]})"
    )

    _legend(
        ax,
        colors
    )

    _save(
        fig,
        f"{args._stem}_dominant_pft.png",
        args.dpi
    )


def _map_pure(
    d,
    args,
    hc,
    ccrs,
    cfeature
):

    import matplotlib.pyplot as plt

    lon = d["lon"]
    lat = d["lat"]

    dominant, purity, has = _dominant_purity(
        d["fr_mean"]
    )

    thr = args.pure_threshold

    m = (
        has
        & (purity > thr)
    )

    cmap, colors = _pft_colors()

    fig = plt.figure(
        figsize=(13, 6)
    )

    ax, kw = _new_axes(
        fig,
        111,
        hc,
        ccrs,
        cfeature
    )

    ax.scatter(
        lon[m],
        lat[m],
        c=dominant[m],
        cmap=cmap,
        vmin=-0.5,
        vmax=N_PFT - 0.5,
        s=args.markersize,
        linewidths=0,
        **kw
    )

    ax.set_title(
        f"Cellules à fraction PFT > {thr:g} "
        f"({m.sum():,} / {has.sum():,} cellules)"
    )

    _legend(
        ax,
        colors
    )

    _save(
        fig,
        f"{args._stem}_pure_gt"
        f"{int(round(thr * 100)):02d}.png",
        args.dpi
    )


def _purity_table(
    d,
    args
):

    import matplotlib.pyplot as plt

    fr_mean = d["fr_mean"]

    counts = np.array(
        [
            [
                int(
                    np.nansum(
                        fr_mean[k] > t
                    )
                )
                for t in THRESHOLDS
            ]
            for k in range(N_PFT)
        ],
        dtype=np.int64
    )

    totals = counts.sum(
        axis=0
    )

    hdr = [
        "PFT"
    ] + [
        f">{t:g}"
        for t in THRESHOLDS
    ]

    lines = [
        "  ".join(
            f"{h:>16}" if i == 0
            else f"{h:>8}"
            for i, h in enumerate(hdr)
        )
    ]

    for k in range(N_PFT):

        row = [
            f"{k + 1}. {PFT_NAMES[k]}"
        ] + [
            str(c)
            for c in counts[k]
        ]

        lines.append(
            "  ".join(
                f"{v:>16}" if i == 0
                else f"{v:>8}"
                for i, v in enumerate(row)
            )
        )

    lines.append(
        "  ".join(
            f"{v:>16}" if i == 0
            else f"{v:>8}"
            for i, v in enumerate(
                ["TOTAL"]
                + [
                    str(t)
                    for t in totals
                ]
            )
        )
    )

    table_txt = "\n".join(lines)

    print(
        "\n== Pureté : nb de cellules "
        "avec fraction PFT > seuil =="
    )

    print(table_txt)

    # CSV
    csv_path = (
        f"{args._stem}_purity_table.csv"
    )

    with open(
        csv_path,
        "w"
    ) as f:

        f.write(
            ",".join(hdr)
            + "\n"
        )

        for k in range(N_PFT):

            f.write(
                ",".join(
                    [
                        f"{k + 1}. {PFT_NAMES[k]}"
                    ]
                    + [
                        str(c)
                        for c in counts[k]
                    ]
                )
                + "\n"
            )

        f.write(
            ",".join(
                ["TOTAL"]
                + [
                    str(t)
                    for t in totals
                ]
            )
            + "\n"
        )

    print(
        f"[ok] {csv_path}"
    )

    # PNG table
    fig, ax = plt.subplots(
        figsize=(
            7,
            0.35 * (N_PFT + 3)
        )
    )

    ax.axis("off")

    cell = [
        [
            f"{k + 1}. {PFT_NAMES[k]}"
        ]
        + [
            f"{c:,}"
            for c in counts[k]
        ]
        for k in range(N_PFT)
    ]

    cell.append(
        ["TOTAL"]
        + [
            f"{t:,}"
            for t in totals
        ]
    )

    tbl = ax.table(
        cellText=cell,
        colLabels=hdr,
        loc="center",
        cellLoc="center"
    )

    tbl.auto_set_font_size(
        False
    )

    tbl.set_fontsize(
        8
    )

    tbl.scale(
        1,
        1.3
    )

    ax.set_title(
        "Nb de cellules par PFT "
        "avec pureté > seuil",
        fontsize=10
    )

    _save(
        fig,
        f"{args._stem}_purity_table.png",
        args.dpi
    )


def _map_density(
    d,
    args,
    hc,
    ccrs,
    cfeature
):

    import matplotlib.pyplot as plt

    lon = d["lon"]
    lat = d["lat"]

    b = args.block_deg

    xe = np.arange(
        -180,
        180 + b,
        b
    )

    ye = np.arange(
        -90,
        90 + b,
        b
    )

    H, _, _ = np.histogram2d(
        lon,
        lat,
        bins=[
            xe,
            ye
        ]
    )

    fig = plt.figure(
        figsize=(14, 7)
    )

    ax, kw = _new_axes(
        fig,
        111,
        hc,
        ccrs,
        cfeature
    )

    Hm = np.ma.masked_where(
        H.T == 0,
        H.T
    )

    mesh = ax.pcolormesh(
        xe,
        ye,
        Hm,
        cmap="YlOrRd",
        shading="flat",
        **(
            {
                "transform": ccrs.PlateCarree()
            }
            if hc
            else {}
        )
    )

    fig.colorbar(
        mesh,
        ax=ax,
        shrink=0.6,
        label="cellules / bloc"
    )

    # Annotate each non-empty block.
    for i in range(H.shape[0]):

        for j in range(H.shape[1]):

            n = int(
                H[i, j]
            )

            if n == 0:
                continue

            xc = 0.5 * (
                xe[i]
                + xe[i + 1]
            )

            yc = 0.5 * (
                ye[j]
                + ye[j + 1]
            )

            ax.text(
                xc,
                yc,
                str(n),
                ha="center",
                va="center",
                fontsize=5,
                color="black",
                **(
                    {
                        "transform":
                        ccrs.PlateCarree()
                    }
                    if hc
                    else {}
                )
            )

    if hc:

        ax.gridlines(
            xlocs=xe,
            ylocs=ye,
            linewidth=0.2,
            color="gray",
            alpha=0.4
        )

    else:

        ax.set_xticks(
            xe[
                ::max(
                    1,
                    len(xe) // 12
                )
            ]
        )

        ax.set_yticks(
            ye
        )

    ax.set_title(
        f"Densité de cellules par bloc "
        f"de {b:g}° "
        f"(total {int(H.sum()):,} cellules)"
    )

    _save(
        fig,
        f"{args._stem}_density_{int(b)}deg.png",
        args.dpi
    )


def _save(
    fig,
    out_path,
    dpi
):

    import matplotlib.pyplot as plt

    os.makedirs(
        os.path.dirname(
            os.path.abspath(out_path)
        ),
        exist_ok=True
    )

    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[ok] {out_path}"
    )


def render(
    d,
    args
):

    import matplotlib

    matplotlib.use("Agg")

    hc, ccrs, cfeature = _try_cartopy()

    _map_dominant(
        d,
        args,
        hc,
        ccrs,
        cfeature
    )

    _map_pure(
        d,
        args,
        hc,
        ccrs,
        cfeature
    )

    _purity_table(
        d,
        args
    )

    _map_density(
        d,
        args,
        hc,
        ccrs,
        cfeature
    )


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument(
        "--selected_pixels",
        required=True,
        help=(
            "selected_pixels*.nc "
            "(site dim: latitude/longitude/site_id)."
        )
    )

    p.add_argument(
        "--pft_dir",
        required=True,
        help=(
            "Folder of PFTmap_{Y}.nc pixelsets "
            "(pft_frac(pft, site)); joined to sites by site_id."
        )
    )

    p.add_argument(
        "--year_start",
        type=int,
        required=True
    )

    p.add_argument(
        "--year_end",
        type=int,
        required=True
    )

    p.add_argument(
        "--output",
        default="selected_pixels_diag.png",
        help=(
            "Output stem; the four figures derive "
            "their names from it."
        )
    )

    p.add_argument(
        "--pure_threshold",
        type=float,
        default=0.6,
        help=(
            "Fraction threshold for the 'pure' map "
            "(default 0.6)."
        )
    )

    p.add_argument(
        "--block_deg",
        type=float,
        default=10.0,
        help=(
            "Block size in degrees for the density "
            "map (default 10)."
        )
    )

    p.add_argument(
        "--markersize",
        type=float,
        default=4.0
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=150
    )

    # Hidden reader-child arguments.
    p.add_argument(
        "--_read_meta",
        default="",
        help=argparse.SUPPRESS
    )

    p.add_argument(
        "--_read_year",
        type=int,
        default=0,
        help=argparse.SUPPRESS
    )

    p.add_argument(
        "--_year_npz",
        default="",
        help=argparse.SUPPRESS
    )

    p.add_argument(
        "--_meta_npz",
        default="",
        help=argparse.SUPPRESS
    )

    args = p.parse_args()

    args._stem = os.path.splitext(
        args.output
    )[0]

    return args


# ===========================================================================
# Subprocess launcher
# ===========================================================================

def _spawn(
    args,
    extra
):
    """
    Run this file again as a throwaway child.
    """

    subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),

            "--selected_pixels",
            args.selected_pixels,

            "--pft_dir",
            args.pft_dir,

            "--year_start",
            str(args.year_start),

            "--year_end",
            str(args.year_end),

            "--output",
            args.output,

            *extra,
        ],
        check=True
    )


# ===========================================================================
# Main
# ===========================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------------
    # Reader child: selected_pixels
    # -----------------------------------------------------------------------

    if args._read_meta:

        read_meta(
            args.selected_pixels,
            args._read_meta
        )

        return

    # -----------------------------------------------------------------------
    # Reader child: one PFT year
    # -----------------------------------------------------------------------

    if args._read_year:

        pft_path = os.path.join(
            args.pft_dir,
            PFT_FNAME.format(
                year=args._read_year
            )
        )

        read_year(
            pft_path,
            args._meta_npz,
            args._year_npz
        )

        return

    # -----------------------------------------------------------------------
    # Main process.
    #
    # IMPORTANT:
    # This process never imports xarray/netCDF4.
    # -----------------------------------------------------------------------

    tmpdir = tempfile.mkdtemp(
        prefix="diag_selpix_"
    )

    meta_npz = os.path.join(
        tmpdir,
        "meta.npz"
    )

    try:

        # ---------------------------------------------------------------
        # Read selected_pixels in its own process.
        # ---------------------------------------------------------------

        _spawn(
            args,
            [
                "--_read_meta",
                meta_npz
            ]
        )

        meta = np.load(
            meta_npz,
            allow_pickle=False
        )

        lon = meta["lon"]
        lat = meta["lat"]
        sid_sel = meta["sid_sel"]

        n_site = sid_sel.size

        print(
            f"[read] selected_pixels: "
            f"{n_site:,} sites",
            flush=True
        )

        del meta

        # ---------------------------------------------------------------
        # Aggregation state.
        #
        # For ONE YEAR:
        #
        #     fr_mean = fr
        #
        # No sum_, no cnt, no division, no large temporary arrays.
        #
        # For MULTIPLE YEARS:
        #
        #     sum_ and cnt are accumulated.
        # ---------------------------------------------------------------

        sum_ = None
        cnt = None

        single_year_fr = None

        years = []

        n_years_requested = (
            args.year_end
            - args.year_start
            + 1
        )

        for i, y in enumerate(
            range(
                args.year_start,
                args.year_end + 1
            ),
            start=1
        ):

            pft_path = os.path.join(
                args.pft_dir,
                PFT_FNAME.format(
                    year=y
                )
            )

            if not os.path.exists(
                pft_path
            ):

                print(
                    f"[warn] missing PFTmap_{y}.nc "
                    f"— skipped [{i}/{n_years_requested}]",
                    flush=True
                )

                continue

            year_npz = os.path.join(
                tmpdir,
                f"year_{y}.npz"
            )

            # -----------------------------------------------------------
            # Read the year's PFT file in a throwaway process.
            # -----------------------------------------------------------

            _spawn(
                args,
                [
                    "--_read_year",
                    str(y),

                    "--_year_npz",
                    year_npz,

                    "--_meta_npz",
                    meta_npz,
                ]
            )

            # -----------------------------------------------------------
            # Load the aligned array.
            #
            # Shape:
            #
            #     (15, 1,649,351)
            #
            # ≈ 99 MB float32.
            # -----------------------------------------------------------

            fr = np.load(
                year_npz,
                allow_pickle=False
            )["fr"]

            os.remove(
                year_npz
            )

            years.append(y)

            print(
                f"[read] {y} "
                f"[{i}/{n_years_requested}]",
                flush=True
            )

            # -----------------------------------------------------------
            # If the requested period contains exactly ONE year,
            # this array IS the mean.
            #
            # Do not allocate sum_, cnt or another fr_mean.
            # -----------------------------------------------------------

            if n_years_requested == 1:

                single_year_fr = fr

                continue

            # -----------------------------------------------------------
            # Multi-year aggregation.
            # -----------------------------------------------------------

            if sum_ is None:

                sum_ = np.zeros_like(
                    fr,
                    dtype=np.float32
                )

                cnt = np.zeros_like(
                    fr,
                    dtype=np.float32
                )

            finite = np.isfinite(
                fr
            )

            # -----------------------------------------------------------
            # Instead of:
            #
            #     np.where(finite, fr, 0)
            #
            # which allocates another ~100 MB array,
            # modify the loaded year's array in place.
            # -----------------------------------------------------------

            fr[~finite] = 0.0

            sum_ += fr

            cnt += finite

            del fr
            del finite

        # -----------------------------------------------------------------
        # No files found.
        # -----------------------------------------------------------------

        if not years:

            raise SystemExit(
                f"No PFTmap_*.nc in {args.pft_dir} "
                f"for {args.year_start}..{args.year_end}."
            )

        # -----------------------------------------------------------------
        # SINGLE-YEAR CASE
        # -----------------------------------------------------------------

        if len(years) == 1:

            print(
                f"[agg] one year ({years[0]}): "
                f"no aggregation needed",
                flush=True
            )

            fr_mean = single_year_fr

            del single_year_fr

        # -----------------------------------------------------------------
        # MULTI-YEAR CASE
        # -----------------------------------------------------------------

        else:

            print(
                f"[agg] averaging "
                f"{len(years)} years over "
                f"{n_site:,} sites "
                f"in chunks of {SITE_CHUNK:,}…",
                flush=True
            )

            n_pft = sum_.shape[0]

            fr_mean = np.empty(
                (
                    n_pft,
                    n_site
                ),
                dtype=np.float32
            )

            # -----------------------------------------------------------
            # Divide in chunks to avoid constructing a huge temporary
            # (15, 1.65M).
            # -----------------------------------------------------------

            for a in range(
                0,
                n_site,
                SITE_CHUNK
            ):

                b = min(
                    a + SITE_CHUNK,
                    n_site
                )

                s = sum_[
                    :,
                    a:b
                ]

                c = cnt[
                    :,
                    a:b
                ]

                with np.errstate(
                    invalid="ignore",
                    divide="ignore"
                ):

                    out = (
                        s / c
                    )

                # Replace sites with zero observations by NaN.
                out[
                    c == 0
                ] = np.nan

                fr_mean[
                    :,
                    a:b
                ] = out

                del s
                del c
                del out

            del sum_
            del cnt

        # -----------------------------------------------------------------
        # Prepare plotting dictionary.
        # -----------------------------------------------------------------

        d = {
            "lon": lon,
            "lat": lat,
            "fr_mean": fr_mean,
            "years": np.asarray(
                years,
                dtype=np.int32
            ),
        }

        # -----------------------------------------------------------------
        # Plot.
        # -----------------------------------------------------------------

        render(
            d,
            args
        )

    except subprocess.CalledProcessError as e:

        raise SystemExit(
            "Reader child failed "
            f"(exit {e.returncode}) "
            "reading the .nc files."
        )

    finally:

        # Clean temporary files.
        if os.path.isdir(tmpdir):

            for f in os.listdir(
                tmpdir
            ):

                path = os.path.join(
                    tmpdir,
                    f
                )

                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

            try:
                os.rmdir(
                    tmpdir
                )
            except OSError:
                pass


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()
