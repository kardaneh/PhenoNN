#!/usr/bin/env python3
"""
pft_amplitude_correlation.py
============================

Per-PFT diagnostic to decide WHY the model stays close to the mean / does not
beat climatology for some PFTs. It separates the two causes:

  (a) amplitude damping      -> pred/obs correlation is GOOD but the predicted
                                seasonal amplitude is much smaller than observed
                                (the MSE rewards hedging toward the mean).
                                => a spectral / amplitude loss term can help.
  (b) lack of predictability -> correlation is ~0 (the features do not explain
                                the signal, or too few pixels of that PFT).
                                => no loss change will fix this.

Input
-----
The CSV produced by `prediction.predict` (36 dekadal rows per site×year):
    site_id, year, month, day, doy, lai_pred, lai_obs, lai_pred_norm,
    lai_obs_norm, error

For each pixel's DOMINANT PFT it computes, per (site, year) annual curve:
  - seasonal amplitude of the annual harmonic |rfft[1]|, predicted vs observed
  - the amplitude ratio  pred / obs
  - the timing shift of the annual harmonic (phase of rfft[1]) in dekads
  - the Pearson correlation between the predicted and observed annual curve
then aggregates (median) per dominant PFT into a summary CSV + 2 figures.

Column choice
-------------
  --column real : lai_pred / lai_obs   (reconstructed LAI; default)
  --column norm : lai_pred_norm / lai_obs_norm

IMPORTANT for anomaly-mode runs: the *real* LAI curves all share the
climatological seasonal cycle, so their correlation stays high EVEN IF the
model only predicts the climatology. To judge "does it beat climatology?" use
`--column norm`, where the norm columns ARE the anomaly (deviation from
climatology): a collapsed model then shows ~0 amplitude and ~0 correlation =
case (b).

Usage
-----
    python -m plot_diagnositcs.pft_amplitude_correlation \\
        --predictions runs_final/exp/predictions.csv \\
        --pft_dir     /data/sbarbu/pft \\
        --output_dir  plot_diagnositcs/pft_diag
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phenonn.utils.config import PFT_FNAME, PFT_NAMES, N_PFT


# ── Inlined helpers (avoid importing phenonn.data.lai_dataset, which pulls torch) ───


def _load_pft_vector_array(ds: xr.Dataset) -> xr.DataArray:
    """Return a (pft, latitude, longitude) DataArray from a PFT file.

    Inlined copy of phenonn.data.lai_dataset._load_pft_vector_array so this script
    runs without torch. Supports the ORCHIDEE PFTmap (maxvegetfrac), 15
    pftK_frac variables, or a single 3-D pft variable.
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
        f"Could not detect a PFT layout in {list(ds.data_vars)}."
    )


# ── PFT assignment ────────────────────────────────────────────────────────────


def assign_dominant_pft(site_ids, pft_dir: Path, pft_year: int) -> Dict[str, int]:
    """Map each site_id to its dominant PFT index (argmax of the 15-vector).

    Reads the pixelset PFTmap (`pft_frac(pft, site)`) and aligns to `site_ids`
    by the file's `site_id` coord. Sites absent from the file (or all-NaN) are
    omitted from the result.
    """
    pft_path = pft_dir / PFT_FNAME.format(year=pft_year)
    if not pft_path.exists():
        raise FileNotFoundError(f"PFT file not found: {pft_path}")
    # Pixelset PFT: netcdf4 + decode_times=False (matches lai_dataset).
    ds = xr.open_dataset(pft_path, engine="netcdf4", decode_times=False)
    pft_da = _load_pft_vector_array(ds)
    all_sites = np.asarray(ds["site_id"].values).astype(str)
    arr = pft_da.transpose("pft", "site").values.astype(np.float32)  # (N_PFT, n_all)
    ds.close()

    idx_of = {s: i for i, s in enumerate(all_sites)}
    rows = np.array([idx_of.get(s, -1) for s in site_ids], dtype=np.int64)
    sub = np.full((arr.shape[0], len(site_ids)), np.nan, dtype=np.float32)
    ok = rows >= 0
    sub[:, ok] = arr[:, rows[ok]]

    dominant: Dict[str, int] = {}
    for i, sid in enumerate(site_ids):
        if np.isfinite(sub[:, i]).any():
            dominant[sid] = int(np.nanargmax(sub[:, i]))
    return dominant


# ── Curve metrics ─────────────────────────────────────────────────────────────


def curve_metrics(obs: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Per-row (site×year) amplitude / phase / correlation metrics.

    obs, pred : (M, N_dekad) arrays of the annual curves.
    """
    n = obs.shape[1]
    h_obs = np.fft.rfft(obs, axis=1)
    h_pred = np.fft.rfft(pred, axis=1)

    amp_obs = np.abs(h_obs[:, 1]) * 2.0 / n     # amplitude of the annual cosine
    amp_pred = np.abs(h_pred[:, 1]) * 2.0 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_ratio = np.where(amp_obs > 1e-9, amp_pred / amp_obs, np.nan)

    dphi = np.angle(h_pred[:, 1]) - np.angle(h_obs[:, 1])
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi   # wrap to (-pi, pi]
    # Harmonic-1 phase difference, expressed in dekads. Only its magnitude is
    # meaningful here (the summary reports the median |.|): it is the timing
    # offset of the annual peak between pred and obs.
    phase_shift_dekads = dphi * n / (2 * np.pi)

    o = obs - obs.mean(axis=1, keepdims=True)
    p = pred - pred.mean(axis=1, keepdims=True)
    num = (o * p).sum(axis=1)
    den = np.sqrt((o ** 2).sum(axis=1) * (p ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.where(den > 1e-12, num / den, np.nan)

    return pd.DataFrame({
        "amp_obs": amp_obs,
        "amp_pred": amp_pred,
        "amp_ratio": amp_ratio,
        "phase_shift_dekads": phase_shift_dekads,
        "pearson": pearson,
    })


def classify(r_med: float, ratio_med: float) -> str:
    """Heuristic verdict per PFT (thresholds are deliberately loose)."""
    if not np.isfinite(r_med):
        return "n/a"
    if r_med < 0.2:
        return "(b) faible skill"
    if ratio_med < 0.8:
        return "(a) amortissement"
    return "ok / mixte"


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_mean_curves(obs, pred, pft_of_row, doy, present, out_path):
    """Small multiples: mean observed vs mean predicted annual curve per PFT."""
    ncol = 4
    nrow = math.ceil(len(present) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, pft in enumerate(present):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        m = pft_of_row == pft
        ax.plot(doy, obs[m].mean(0), color="k", lw=2, label="obs")
        ax.plot(doy, pred[m].mean(0), color="C3", lw=2, ls="--", label="pred")
        ax.set_title(f"{pft}: {PFT_NAMES[pft]}  (n={int(m.sum())})", fontsize=9)
        ax.set_xlabel("doy")
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Mean annual LAI curve per dominant PFT — obs vs pred",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, out_path):
    """Per-PFT median Pearson r and median amplitude ratio."""
    labels = [f"{int(r.pft)}:{r.pft_name}" for r in summary.itertuples()]
    y = np.arange(len(summary))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(summary))))

    ax1.barh(y, summary["pearson_median"], color="C0")
    ax1.axvline(0.2, color="grey", ls=":", lw=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("median Pearson r (pred vs obs)")
    ax1.set_title("Skill (low => case b)")
    ax1.invert_yaxis()

    ax2.barh(y, summary["amp_ratio_median"], color="C1")
    ax2.axvline(1.0, color="k", ls="-", lw=1)
    ax2.axvline(0.8, color="grey", ls=":", lw=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("median amplitude ratio pred/obs")
    ax2.set_title("Amplitude (<1 => damping, case a)")
    ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions", required=True,
                   help="CSV produced by prediction.predict")
    p.add_argument("--pft_dir", required=True,
                   help="Folder of PFTmap_{year}.nc")
    p.add_argument("--pft_year", type=int, default=0,
                   help="Year of the PFT file used to assign dominant PFT "
                        "(default: smallest year present in the CSV).")
    p.add_argument("--column", choices=["real", "norm"], default="real",
                   help="real=lai_pred/lai_obs ; norm=lai_pred_norm/lai_obs_norm "
                        "(use norm for anomaly-mode runs to judge skill vs "
                        "climatology).")
    p.add_argument("--output_dir", default="plot_diagnositcs/pft_diag")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    col_pred = "lai_pred" if args.column == "real" else "lai_pred_norm"
    col_obs = "lai_obs" if args.column == "real" else "lai_obs_norm"

    df = pd.read_csv(args.predictions)
    for c in ("site_id", "year", "doy", col_pred, col_obs):
        if c not in df.columns:
            raise ValueError(f"Column {c!r} missing from {args.predictions}")
    pft_year = args.pft_year or int(df["year"].min())
    print(f"Predictions : {args.predictions}  ({len(df):,} rows)")
    print(f"Column      : {args.column}  ({col_pred} / {col_obs})")
    print(f"PFT year    : {pft_year}")

    # ── Pivot to (site, year) × doy ──
    obs_p = df.pivot_table(index=["site_id", "year"], columns="doy",
                           values=col_obs)
    pred_p = df.pivot_table(index=["site_id", "year"], columns="doy",
                            values=col_pred).reindex(obs_p.index)
    doy = obs_p.columns.values.astype(float)

    obs = obs_p.values
    pred = pred_p.values
    keep = ~(np.isnan(obs).any(1) | np.isnan(pred).any(1))
    obs, pred = obs[keep], pred[keep]
    index = obs_p.index[keep]
    site_of_row = index.get_level_values("site_id").to_numpy()
    print(f"Samples     : {len(obs):,} complete (site, year) curves "
          f"over {obs.shape[1]} dekads")
    if len(obs) == 0:
        raise RuntimeError("No complete annual curves to analyse.")

    # ── Per-row metrics + dominant PFT ──
    metrics = curve_metrics(obs, pred)
    site_to_pft = assign_dominant_pft(np.unique(site_of_row), Path(args.pft_dir),
                                      pft_year)
    pft_of_row = np.array([site_to_pft[s] for s in site_of_row])
    metrics["pft"] = pft_of_row

    # ── Aggregate per dominant PFT ──
    grp = metrics.groupby("pft")
    summary = pd.DataFrame({
        "pft": grp.size().index,
        "n_samples": grp.size().values,
        "pearson_median": grp["pearson"].median().values,
        "amp_ratio_median": grp["amp_ratio"].median().values,
        "amp_obs_median": grp["amp_obs"].median().values,
        "amp_pred_median": grp["amp_pred"].median().values,
        "phase_shift_abs_median_dekads": grp["phase_shift_dekads"]
            .apply(lambda s: np.nanmedian(np.abs(s))).values,
    })
    summary["pft_name"] = [PFT_NAMES[int(p)] for p in summary["pft"]]
    summary["verdict"] = [classify(r, a) for r, a in
                          zip(summary["pearson_median"],
                              summary["amp_ratio_median"])]
    summary = summary.sort_values("n_samples", ascending=False).reset_index(drop=True)

    csv_path = out_dir / "pft_diagnostic_summary.csv"
    summary.to_csv(csv_path, index=False)
    with pd.option_context("display.width", 140,
                           "display.max_columns", None):
        print("\n" + summary.to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))
    print(f"\nSummary CSV -> {csv_path}")

    # ── Figures ──
    present = list(summary["pft"].astype(int))
    plot_mean_curves(obs, pred, pft_of_row, doy, present,
                     out_dir / "mean_curves_per_pft.png")
    plot_summary_bars(summary, out_dir / "skill_amplitude_per_pft.png")
    print(f"Figures     -> {out_dir}/mean_curves_per_pft.png")
    print(f"               {out_dir}/skill_amplitude_per_pft.png")
    print("\nReading the verdict:")
    print("  (a) amortissement  : r OK mais amp_ratio < 1  -> loss spectrale utile")
    print("  (b) faible skill   : r ~ 0                    -> probleme features/donnees")


if __name__ == "__main__":
    main()
