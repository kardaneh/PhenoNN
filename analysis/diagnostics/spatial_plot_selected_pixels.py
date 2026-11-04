#!/usr/bin/env python3
"""
spatial_plot_selected_pixels.py
===============================

Plot the sites contained in a ``selected_pixels.nc`` file on a world map.
Two modes are supported:

- uniform scatter (default): every selected site as a red dot;
- density map (``--density``): hexbin density with a log-count colourbar,
  useful as the dataset overview figure.

The input is any file with the selected-pixels layout (a ``site`` dim carrying
``latitude`` / ``longitude``), e.g. the full site pool, a PFT subset
(``selected_pixels_PFT{n}.nc``), or a filtered PFT subset.

Coastlines need **cartopy** (``pip install cartopy``). If cartopy is not
installed, the script degrades to a plain gridded lon/lat panel.

Usage
-----
    python -m analysis.diagnostics.spatial_plot_selected_pixels \\
        --selected_pixels "$DATA/selected_pixels005_balanced.nc" \\
        --density --output runs/report_reproduction/D1_site_density.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless server (save to file, no display)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from analysis.figures.ipsl_aid import hexbin_density  # noqa: E402
from analysis.figures.style import PALETTE, apply_style, savefig  # noqa: E402

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _cartopy_data = os.environ.get("CARTOPY_DATA_DIR")
    if _cartopy_data:
        cartopy.config["data_dir"] = _cartopy_data

    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False


def load_points(path: str):
    """Return (lon, lat, site_ids, n_site, attrs)."""
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        site_ids = (
            np.asarray(ds["site_id"].values).astype(str)
            if "site_id" in ds
            else np.array([], dtype=str)
        )
        n_site = int(ds.sizes.get("site", lat.size))
        attrs = dict(ds.attrs)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lon, lat, site_ids, n_site, attrs


def _title(path: str, n_site: int, attrs: dict) -> str:
    name = os.path.basename(path)
    pft = attrs.get("filter_pft_orchidee")
    suffix = f" — PFT{int(pft)} (ORCHIDEE)" if pft is not None else ""
    return f"{name}{suffix}\n{n_site:,} selected sites"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--selected_pixels",
        required=True,
        help="selected_pixels*.nc (site dim with latitude/longitude).",
    )
    p.add_argument("--output", default="runs/spatial_selected_pixels.png")
    p.add_argument(
        "--density",
        action="store_true",
        help="Draw a hexbin density map with a log-count colourbar.",
    )
    p.add_argument("--gridsize", type=int, default=100)
    p.add_argument("--markersize", type=float, default=4.0)
    p.add_argument("--color", default=PALETTE["red"], help="Scatter marker colour.")
    p.add_argument(
        "--highlight_site",
        default="",
        help="Optional site_id to mark on the map (e.g. the F2/F11 site).",
    )
    args = p.parse_args()

    apply_style()
    lon, lat, site_ids, n_site, attrs = load_points(args.selected_pixels)
    print(
        f"{n_site:,} sites  |  lat [{lat.min():.1f}, {lat.max():.1f}]  "
        f"lon [{lon.min():.1f}, {lon.max():.1f}]"
    )

    fig = plt.figure(figsize=(14, 7))
    if HAVE_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="0.96", zorder=0)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
        gl.top_labels = gl.right_labels = False
        transform = ccrs.PlateCarree()
    else:
        print(
            "[warn] cartopy not installed — no coastlines. "
            'Install it with `uv pip install -e ".[full,figures]"` '
            "or `pip install cartopy`; plotting a plain lon/lat grid."
        )
        ax = plt.axes()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, linewidth=0.3, alpha=0.4, linestyle="--")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        transform = None

    if args.density:
        kwargs = {"transform": transform} if transform is not None else {}
        hb = hexbin_density(
            ax,
            lon,
            lat,
            gridsize=args.gridsize,
            **kwargs,
        )
        cbar = fig.colorbar(
            hb,
            ax=ax,
            orientation="horizontal",
            pad=0.06,
            shrink=0.6,
            aspect=40,
        )
        cbar.set_label(r"$\log_{10}[\mathrm{count}]$")
    else:
        kwargs = {"transform": transform} if transform is not None else {}
        ax.scatter(
            lon,
            lat,
            s=args.markersize,
            c=args.color,
            alpha=0.6,
            edgecolors="none",
            zorder=3,
            **kwargs,
        )

    if args.highlight_site:
        hit = np.flatnonzero(site_ids == str(args.highlight_site))
        if hit.size:
            idx = int(hit[0])
            ax.scatter(
                [lon[idx]],
                [lat[idx]],
                s=120,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=6,
                **({"transform": transform} if transform is not None else {}),
            )
            ax.annotate(
                str(args.highlight_site),
                (lon[idx], lat[idx]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                zorder=7,
            )
        else:
            print(f"[warn] highlight site not found: {args.highlight_site}")

    ax.set_title(
        f"Selected-site density\n{n_site:,} sites"
        if args.density
        else _title(args.selected_pixels, n_site, attrs)
    )
    fig.tight_layout()
    savefig(fig, args.output)
    print(f"Saved → {args.output} (+ PDF)")


if __name__ == "__main__":
    main()
