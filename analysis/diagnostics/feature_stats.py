# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Audit channel distributions before choosing a normalisation policy.

Reads a seeded sample of the ERA5 feature cubes (plus the LAI target and the
PFT map) and reports, per channel, the statistics that decide between
``zscore``, ``log1p_zscore``, ``sqrt_zscore``, ``minmax`` and ``none``:
moments, quantiles, skew, kurtosis, zero fraction and mass at the bounds.

The script is torch-free. Outputs are a CSV per table, one summary Markdown
and diagnostic PNGs; nothing is written outside ``--output_dir``.

Examples
--------
::

    python -m analysis.diagnostics.feature_stats \\
        --features_dir pixelset1/era5_pixelset \\
        --target_dir pixelset1/LAI_pixelset \\
        --pft_dir pixelset1/PFT_pixelset \\
        --parent_map pixelset1/selected_pixels01_1.nc \\
        --years 1992-2009 --n_sites 1000 --stride 3 \\
        --output_dir runs/experiments/norm_audit
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from phenonn.utils.config import (  # noqa: E402
    CO2_FEATURES,
    DYNAMIC_FEATURES,
    FEATURES_FNAME,
    PFT_COLS,
    PFT_FNAME,
    PFT_NAMES,
    TARGETS_FNAME,
)

_BOUNDS = {
    "daylength": (0.0, 24.0),
}


# ── Statistics ──────────────────────────────────────────────────────────────


class Moments:
    """Streaming mean, std, skew and excess kurtosis (finite values only)."""

    __slots__ = ("n", "s1", "s2", "s3", "s4", "lo", "hi")

    def __init__(self) -> None:
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.lo = np.inf
        self.hi = -np.inf

    def update(self, values: np.ndarray) -> None:
        """Fold a finite 1-D array into the accumulator."""
        x = np.asarray(values, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        self.n += int(x.size)
        self.s1 += float(x.sum())
        self.s2 += float(np.square(x).sum())
        self.s3 += float(np.power(x, 3).sum())
        self.s4 += float(np.power(x, 4).sum())
        self.lo = min(self.lo, float(x.min()))
        self.hi = max(self.hi, float(x.max()))

    def summary(self) -> Dict[str, float]:
        """Return mean/std/skew/kurtosis/min/max."""
        if self.n == 0:
            return {"n": 0}
        n = float(self.n)
        mean = self.s1 / n
        var = max(self.s2 / n - mean * mean, 0.0)
        std = float(np.sqrt(var))
        if var <= 0.0:
            return {
                "n": self.n,
                "mean": mean,
                "std": 0.0,
                "skew": 0.0,
                "kurtosis": 0.0,
                "min": self.lo,
                "max": self.hi,
            }
        m3 = self.s3 / n - 3 * mean * self.s2 / n + 2 * mean**3
        m4 = (
            self.s4 / n
            - 4 * mean * self.s3 / n
            + 6 * mean**2 * self.s2 / n
            - 3 * mean**4
        )
        return {
            "n": self.n,
            "mean": mean,
            "std": std,
            "skew": float(m3 / var**1.5),
            "kurtosis": float(m4 / var**2 - 3.0),
            "min": self.lo,
            "max": self.hi,
        }


class ChannelAudit:
    """Accumulate raw/log1p/sqrt moments plus a quantile subsample."""

    def __init__(self, per_year: int = 4000) -> None:
        self.raw = Moments()
        self.log = Moments()
        self.sqrt = Moments()
        self.per_year = per_year
        self._samples: List[np.ndarray] = []

    def update(self, values: np.ndarray, rng: np.random.Generator) -> None:
        """Fold one year of a channel."""
        x = np.asarray(values, dtype=np.float32).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        self.raw.update(x)
        self.log.update(np.log1p(np.clip(x, 0.0, None)))
        self.sqrt.update(np.sqrt(np.clip(x, 0.0, None)))
        take = min(self.per_year, x.size)
        if take:
            idx = rng.choice(x.size, size=take, replace=False)
            self._samples.append(x[idx].astype(np.float64))

    def quantiles(self, qs: Sequence[float]) -> Dict[str, float]:
        """Quantiles from the pooled subsample (empty dict when no data)."""
        if not self._samples:
            return {}
        pooled = np.concatenate(self._samples)
        values = np.percentile(pooled, qs)
        return {f"p{q:g}": float(v) for q, v in zip(qs, values)}

    def zero_frac(self) -> float:
        """Fraction of exact zeros in the pooled subsample."""
        if not self._samples:
            return float("nan")
        pooled = np.concatenate(self._samples)
        return float(np.mean(pooled == 0.0))


# ── I/O helpers ─────────────────────────────────────────────────────────────


def parse_years(spec: str) -> List[int]:
    """Parse ``1992-2009`` or ``1992,1995,2000`` into a year list."""
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def site_blocks(
    n_total: int, n_sites: int, seed: int, n_blocks: int = 20
) -> List[Tuple[int, int]]:
    """Return contiguous site blocks to read instead of a scattered index.

    A scattered ``isel(site=index_array)`` forces netCDF4 to load the whole
    variable before subsetting (hundreds of MB per channel-year). Contiguous
    blocks are read as hyperslabs, which keeps the audit I/O small. The blocks
    are spread over the whole site axis, so the sample stays representative.

    Parameters
    ----------
    n_total : int
        Sites in the file.
    n_sites : int
        Requested sample size (``<= 0`` or ``>= n_total`` means all sites).
    seed : int
        RNG seed for the block positions.
    n_blocks : int
        Number of blocks; the block size is ``n_sites // n_blocks``.

    Returns
    -------
    list of (start, stop)
        Half-open site slices.
    """
    if n_sites <= 0 or n_sites >= n_total or n_total == 0:
        return [(0, n_total)]
    block = max(1, n_sites // n_blocks)
    n_slots = max(1, n_total // block)
    rng = np.random.RandomState(seed)
    slots = rng.choice(n_slots, size=min(n_blocks, n_slots), replace=False)
    return sorted(
        (int(s) * block, min(int(s) * block + block, n_total)) for s in slots
    )


def dominant_pft(
    parent_map: str, pft_path: str, n_parent: int
) -> np.ndarray:
    """Return the dominant PFT index of every 0.1-degree cell.

    The 0.05-degree PFT fractions of all children of a cell are summed and the
    largest fraction is selected, so the group reflects the cell, not one child.
    """
    with xr.open_dataset(parent_map, engine="netcdf4", decode_times=False) as ds:
        parent = np.asarray(ds["parent_site"].values)
    agg = np.zeros((n_parent, len(PFT_COLS)), dtype=np.float64)
    with xr.open_dataset(pft_path, engine="netcdf4", decode_times=False) as ds:
        frac = np.asarray(ds["pft_frac"].values, dtype=np.float64)
    for k in range(len(PFT_COLS)):
        agg[:, k] = np.bincount(parent, weights=frac[k], minlength=n_parent)
    return agg.argmax(axis=1)


def recommend(row: Dict[str, float]) -> str:
    """Rule-of-thumb transform recommendation from the audit row."""
    zero = row.get("zero_frac", 0.0)
    skew = row.get("skew", 0.0)
    p99 = row.get("p99", np.nan)
    hi = row.get("max", np.nan)
    lo = row.get("min", np.nan)
    if np.isfinite(zero) and zero > 0.4:
        return "zero-inflated: sqrt_zscore or log1p_zscore"
    if np.isfinite(skew) and skew > 1.5:
        return "heavy right tail: log1p_zscore or sqrt_zscore"
    if np.isfinite(hi) and np.isfinite(p99) and p99 > 0 and hi > 10 * p99:
        return "outliers: minmax on p1-p99 (clip)"
    if np.isfinite(lo) and np.isfinite(hi) and lo >= 0 and (hi - lo) < 30 and abs(skew) < 1.0:
        return "bounded: minmax"
    if abs(skew) < 0.5:
        return "near-symmetric: zscore"
    return "zscore; check the per-PFT split"


# ── Main scan ───────────────────────────────────────────────────────────────


def audit_features(
    args, years: Sequence[int], pft_group: Optional[np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Scan the feature cubes and return (channel table, per-PFT table, pools)."""
    rng = np.random.default_rng(args.seed)
    audits = {name: ChannelAudit(args.per_year) for name in DYNAMIC_FEATURES}
    group_audits: Dict[Tuple[str, int], Moments] = {}
    pools: Dict[str, np.ndarray] = {}
    blocks: Optional[List[Tuple[int, int]]] = None

    for i_year, year in enumerate(years):
        path = os.path.join(args.features_dir, FEATURES_FNAME.format(year=year))
        if not os.path.exists(path):
            print(f"  [warn] missing {path}")
            continue
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
        if blocks is None:
            blocks = site_blocks(int(ds.sizes["site"]), args.n_sites, args.seed)
        for name in DYNAMIC_FEATURES:
            if name not in ds:
                continue
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=slice(None, None, args.stride))
            for start, stop in blocks:
                values = np.asarray(
                    da.isel(site=slice(start, stop)).values, dtype=np.float32
                )
                audits[name].update(values, rng)
                if i_year == 0 and name not in pools:
                    pools[name] = values.ravel()[:200_000]

                if pft_group is not None:
                    labels = pft_group[start:stop]
                    for g in np.unique(labels):
                        cols = np.flatnonzero(labels == g)
                        if cols.size == 0:
                            continue
                        key = (name, int(g))
                        group_audits.setdefault(key, Moments())
                        group_audits[key].update(values[:, cols])
        ds.close()
        print(f"  {year}: {len(DYNAMIC_FEATURES)} channels  ({i_year + 1}/{len(years)})")

    qs = (1, 5, 50, 95, 99)
    rows = []
    for name, audit in audits.items():
        row = {"channel": name, **audit.raw.summary()}
        row.update(audit.quantiles(qs))
        row["zero_frac"] = audit.zero_frac()
        row["log1p_mean"] = audit.log.summary().get("mean", np.nan)
        row["log1p_std"] = audit.log.summary().get("std", np.nan)
        row["log1p_skew"] = audit.log.summary().get("skew", np.nan)
        row["sqrt_mean"] = audit.sqrt.summary().get("mean", np.nan)
        row["sqrt_std"] = audit.sqrt.summary().get("std", np.nan)
        row["sqrt_skew"] = audit.sqrt.summary().get("skew", np.nan)
        row["recommendation"] = recommend(row)
        rows.append(row)

    group_rows = []
    for (name, g), mom in group_audits.items():
        s = mom.summary()
        group_rows.append(
            {
                "channel": name,
                "pft": g,
                "pft_name": PFT_NAMES[g] if g < len(PFT_NAMES) else str(g),
                **{k: s.get(k, np.nan) for k in ("n", "mean", "std", "skew")},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(group_rows), pools


def audit_targets(args, years: Sequence[int]) -> pd.DataFrame:
    """Stats of the LAI target, including the quantization step."""
    acc = Moments()
    samples: List[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    blocks: Optional[List[Tuple[int, int]]] = None
    for year in years:
        path = os.path.join(args.target_dir, TARGETS_FNAME.format(year=year))
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
            if blocks is None:
                blocks = site_blocks(int(ds.sizes["site"]), args.n_sites, args.seed)
            da = ds["LAI"]
            parts = [
                np.asarray(da.isel(site=slice(a, b)).values, dtype=np.float32).ravel()
                for a, b in blocks
            ]
        values = np.concatenate(parts)
        acc.update(values)
        take = min(args.per_year, values.size)
        samples.append(values[rng.choice(values.size, take, replace=False)])
    s = acc.summary()
    pooled = np.concatenate(samples) if samples else np.array([], dtype=np.float32)
    row = {"channel": "LAI", **{k: s.get(k, np.nan) for k in ("n", "mean", "std", "skew", "min", "max")}}
    if pooled.size:
        for q, v in zip((1, 5, 50, 95, 99), np.percentile(pooled, (1, 5, 50, 95, 99))):
            row[f"p{q}"] = float(v)
        unique = np.unique(pooled)
        diffs = np.diff(unique)
        diffs = diffs[diffs > 1e-6]
        row["quantization_step"] = float(np.median(diffs)) if diffs.size else np.nan
    return pd.DataFrame([row])


def audit_pft_channels(args, year: int) -> pd.DataFrame:
    """Per-PFT fraction statistics, straight from the PFT map."""
    path = os.path.join(args.pft_dir, PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        frac = np.asarray(ds["pft_frac"].values, dtype=np.float32)
    for k, name in enumerate(PFT_COLS):
        x = frac[k]
        rows.append(
            {
                "channel": name,
                "n": int(np.isfinite(x).sum()),
                "mean": float(np.nanmean(x)),
                "std": float(np.nanstd(x)),
                "min": float(np.nanmin(x)),
                "max": float(np.nanmax(x)),
                "zero_frac": float(np.mean(x == 0.0)),
                "recommendation": "fraction in [0, 1]: none",
            }
        )
    return pd.DataFrame(rows)


# ── Figures ─────────────────────────────────────────────────────────────────


def plot_distributions(pools: Dict[str, np.ndarray], out: str) -> None:
    """One histogram per dynamic channel, log-x when the channel is skewed."""
    names = [n for n in DYNAMIC_FEATURES if n in pools]
    if not names:
        return
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        x = pools[name]
        x = x[np.isfinite(x)]
        if x.size == 0:
            ax.set_visible(False)
            continue
        lo, hi = float(np.min(x)), float(np.max(x))
        bins = 60
        if lo >= 0 and hi > 0 and (np.percentile(x, 99) > 5 * max(np.median(x), 1e-6)):
            edges = np.logspace(np.log10(max(lo, 1e-6)), np.log10(hi), bins)
            ax.set_xscale("log")
        else:
            edges = np.linspace(lo, hi, bins)
        ax.hist(x, bins=edges, color="#1f77b4", alpha=0.85)
        ax.axvline(float(np.mean(x)), color="#d62728", lw=1.2, label="mean")
        ax.axvline(float(np.median(x)), color="#2e7d32", lw=1.2, ls="--", label="median")
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_skew_bounds(table: pd.DataFrame, out: str) -> None:
    """Scatter of zero fraction versus skew, one label per channel."""
    d = table.dropna(subset=["skew", "zero_frac"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.0), layout="constrained")
    ax.scatter(d["zero_frac"], d["skew"], s=40, color="#1f77b4")
    for _, row in d.iterrows():
        ax.annotate(row["channel"], (row["zero_frac"], row["skew"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(1.5, color="grey", ls="--", lw=1.0)
    ax.axvline(0.4, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("fraction of exact zeros")
    ax.set_ylabel("skew")
    ax.set_title("Zero inflation versus skew")
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features_dir", required=True, help="ERA5_daily_pixelset_{Y}.nc")
    p.add_argument("--target_dir", default="", help="LAI_dekadal_{Y}.nc (optional)")
    p.add_argument("--pft_dir", default="", help="PFTmap_{Y}.nc (optional)")
    p.add_argument("--parent_map", default="", help="selected_pixels01_1.nc (active cells)")
    p.add_argument("--years", default="1992-2009")
    p.add_argument("--n_sites", type=int, default=1000)
    p.add_argument("--stride", type=int, default=3, help="Day stride in the scan.")
    p.add_argument("--per_year", type=int, default=4000, help="Quantile subsample per year.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pft_year", type=int, default=0, help="0 = first --years entry.")
    p.add_argument("--output_dir", default="runs/experiments/norm_audit")
    return p.parse_args()


def main() -> None:
    """Run the audit and write the tables, figures and summary."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    years = parse_years(args.years)

    pft_group = None
    if args.pft_dir and args.parent_map and args.pft_year is not None:
        pft_year = args.pft_year or years[0]
        path = os.path.join(args.pft_dir, PFT_FNAME.format(year=pft_year))
        if os.path.exists(path):
            with xr.open_dataset(args.parent_map, engine="netcdf4", decode_times=False) as ds:
                n_parent = int(ds.sizes.get("site", 0))
            pft_group = dominant_pft(args.parent_map, path, n_parent)
            print(f"PFT groups    : {np.unique(pft_group).size} dominant PFTs ({pft_year})")

    table, by_pft, pools = audit_features(args, years, pft_group)
    table.to_csv(os.path.join(args.output_dir, "feature_stats.csv"), index=False)
    if not by_pft.empty:
        by_pft.to_csv(os.path.join(args.output_dir, "feature_stats_by_pft.csv"), index=False)

    pft_table = audit_pft_channels(args, years[0]) if args.pft_dir else pd.DataFrame()
    if not pft_table.empty:
        pft_table.to_csv(os.path.join(args.output_dir, "pft_stats.csv"), index=False)

    target = audit_targets(args, years) if args.target_dir else pd.DataFrame()
    if not target.empty:
        target.to_csv(os.path.join(args.output_dir, "target_lai_stats.csv"), index=False)

    plot_distributions(pools, os.path.join(args.output_dir, "feature_distributions.png"))
    plot_skew_bounds(table, os.path.join(args.output_dir, "feature_skew_bounds.png"))

    lines = [
        "# Channel audit — normalisation policy evidence",
        "",
        f"Years: {years[0]}–{years[-1]}  |  sites/year: {args.n_sites}  |  "
        f"day stride: {args.stride}",
        "",
        "| channel | mean | std | min | p1 | p99 | max | skew | zeros | log1p skew | sqrt skew | recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            "| {channel} | {mean:.3f} | {std:.3f} | {min:.3f} | {p1:.3f} | "
            "{p99:.3f} | {max:.3f} | {skew:+.2f} | {zero_frac:.2f} | "
            "{log1p_skew:+.2f} | {sqrt_skew:+.2f} | {recommendation} |".format(**r)
        )
    if not pft_table.empty:
        lines += ["", "## PFT fractions", "",
                  "| channel | mean | std | zeros | recommendation |",
                  "|---|---:|---:|---:|---|"]
        for _, r in pft_table.iterrows():
            lines.append(
                f"| {r['channel']} | {r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['zero_frac']:.2f} | {r['recommendation']} |"
            )
    if not target.empty:
        t = target.iloc[0]
        lines += ["", "## Target LAI", "",
                  f"mean {t['mean']:.4f}, std {t['std']:.4f}, "
                  f"range [{t['min']:.3f}, {t['max']:.3f}], "
                  f"quantization step {t.get('quantization_step', float('nan')):.4f}"]
    with open(os.path.join(args.output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {args.output_dir}/summary.md and the CSV/PNG set")


if __name__ == "__main__":
    main()
