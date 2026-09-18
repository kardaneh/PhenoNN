#!/usr/bin/env python3
"""
spatial_plot_selected_pixels.py
===============================

Plot the sites contained in a `selected_pixels.nc` file on a world map
(planisphere), with continent coastlines drawn on a lon/lat grid so you can see
where the selected pixels fall relative to the landmasses.

The input is any file with the selected-pixels layout (a `site` dim carrying
`latitude` / `longitude`), e.g. the full pool from `phenonn.data_creation.select_pixels`,
a PFT subset from `phenonn.data_creation.greedy_pure_pft` (selected_pixels_PFT{n}.nc), or
a `phenonn.data_creation.filter_pixels_by_pft` output.

Coastlines need **cartopy** (`pip install cartopy`). If cartopy is not
installed, the script still runs but degrades to a plain gridded lon/lat scatter
with NO coastlines (a warning is printed).

Usage
-----
    python -m prediction.spatial_plot_selected_pixels \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_10%.nc \\
        --output spatial_selected_pixels.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")                 # headless server (save to file, no display)
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import xarray as xr                   # noqa: E402

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False


def load_points(path: str):
    """Return (lon, lat, n_site, attrs). Longitudes wrapped to [-180, 180] so a
    0..360 file plots correctly on the planisphere too."""
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        n_site = int(ds.sizes.get("site", lat.size))
        attrs = dict(ds.attrs)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lon, lat, n_site, attrs


def _title(path: str, n_site: int, attrs: dict) -> str:
    name = os.path.basename(path)
    pft = attrs.get("filter_pft_orchidee")
    suffix = f" — PFT{int(pft)} (ORCHIDEE)" if pft is not None else ""
    return f"{name}{suffix}\n{n_site:,} selected sites"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected_pixels", required=True,
                   help="selected_pixels*.nc (site dim with latitude/longitude).")
    p.add_argument("--output", default="spatial_selected_pixels.png")
    p.add_argument("--markersize", type=float, default=4.0)
    p.add_argument("--color", default="#c0392b", help="Marker colour.")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    lon, lat, n_site, attrs = load_points(args.selected_pixels)
    print(f"{n_site:,} sites  |  lat [{lat.min():.1f}, {lat.max():.1f}]  "
          f"lon [{lon.min():.1f}, {lon.max():.1f}]")

    fig = plt.figure(figsize=(14, 7))
    if HAVE_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4,
                          linestyle="--")
        gl.top_labels = gl.right_labels = False
        ax.scatter(lon, lat, s=args.markersize, c=args.color, alpha=0.6,
                   edgecolors="none", transform=ccrs.PlateCarree(), zorder=3)
    else:
        print("[warn] cartopy not installed — no coastlines "
              "(`pip install cartopy`). Plotting a plain lon/lat grid.")
        ax = plt.axes()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, linewidth=0.3, alpha=0.4, linestyle="--")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.scatter(lon, lat, s=args.markersize, c=args.color, alpha=0.6,
                   edgecolors="none", zorder=3)

    ax.set_title(_title(args.selected_pixels, n_site, attrs))
    fig.tight_layout()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
