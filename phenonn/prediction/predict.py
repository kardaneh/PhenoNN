#!/usr/bin/env python3
"""
predict.py
==========

LAI inference for PhenoNN checkpoints. Port of LaiNN/phenocam/prediction_big.py
adapted to the *pixelset* data pipeline: every input is a flat `site`-indexed
NetCDF keyed by `site_id` (the old full-grid lat/lon layout is no longer
accepted), exactly like `phenonn.training.train_global`:

  - features : ERA5_daily_pixelset_{Y}.nc  (per-site daily series)
  - targets  : LAI_dekadal_{Y}.nc          (pixelset LAI(dekad, site))
  - PFT      : PFTmap_{Y}.nc               (pixelset pft_frac(pft, site))

Loads `best_model.pth` (or any snapshot) produced by `phenonn.training.train_global`, restores
the model, builds an `LAIDataset` over the requested sites × years, runs
inference, recovers physical LAI (handling normalization + anomaly mode),
and writes a CSV compatible with the existing diagnostics scripts:

    site_id, year, month, day, doy, lai_pred, lai_obs,
    lai_pred_norm, lai_obs_norm, error

Site selection
--------------
  --selected_pixels P   : sites listed in a selected_pixels*.nc (e.g.
                          selected_pixels_PFT9.nc) — overrides the modes below
  --predict_sites val   : ckpt['val_site_ids']
  --predict_sites train : ckpt['train_site_ids']
  --predict_sites all   : union of train_site_ids ∪ val_site_ids
  --predict_sites grid  : every pixel id in the bbox `--row_min/max --col_min/max`
                          (falls back to the bbox from training args if -1). Only
                          the bbox sites present in the pixelset survive — the
                          others are silently dropped by LAIDataset.
  --predict_sites test  : grid \\ val_site_ids (held-out for overlap mode)

Usage
-----
    python -m phenonn.prediction.predict \\
        --checkpoint runs_final/exp/checkpoints/best_model.pth \\
        --features_dir /data/sbarbu/era5_features \\
        --target_dir   /data/sbarbu/targets \\
        --pft_dir      /data/sbarbu/pft \\
        --predict_sites val \\
        --output_csv runs_final/exp/predictions.csv
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phenonn.utils.config import ALL_FEATURES, N_DEKAD_YEAR, PFT_FNAME, PFT_NAMES
from phenonn.utils.wrappers import _OBS_POSITIONS
from phenonn.data.lai_dataset import (
    RamLAIDataset, encode_site_id, generate_site_ids_from_range, load_co2_lut,
    load_parent01_map, load_selected_pixels,
)
from phenonn.utils.model_loader import build_model, build_model_pft
from phenonn.utils.diagnostics import plot_pred_vs_obs, plot_gcc_curves, plot_gcc_curves_all
from phenonn.utils.utils import EasyDict


# 36 obs (month, day, doy) — non-leap year
_OBS_DATES = [
    (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
    for m in range(1, 13)
    for d in [5, 15, 25]
]


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="PhenoNN inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir",   default="")
    p.add_argument("--pft_dir",      default="")
    p.add_argument("--parent_map", default="",
                   help="Optional selected_pixels_01.nc (data_creation."
                        "make_selected_pixels_01). When set, ERA5 features are "
                        "read from the deduplicated 0.1° ERA5_daily_pixelset "
                        "(site_id 'E{lat}_{lon}') via each 0.05° site's parent "
                        "cell, instead of a 0.05°-indexed feature file. "
                        "Targets/PFT stay on the 0.05° sites. Must match the "
                        "--parent_map used at training.")
    p.add_argument("--predict_sites", default="val",
                   choices=["val", "train", "all", "grid", "test"])
    p.add_argument("--selected_pixels", default="",
                   help="Path to a selected_pixels*.nc (e.g. "
                        "selected_pixels_PFT9.nc). When set, predict only on "
                        "its sites — overrides --sites and --predict_sites.")
    p.add_argument("--sites", default="",
                   help="Comma-separated explicit site IDs — overrides "
                        "--predict_sites.")
    p.add_argument("--n_predict_sites", type=int, default=0,
                   help="If > 0, random subsample of this many sites.")
    p.add_argument("--predict_years", default="",
                   help="'2015-2018', '2015,2016' or 'all'. "
                        "Empty → val_years from the checkpoint.")
    p.add_argument("--row_min", type=int, default=-1)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=-1)
    p.add_argument("--col_max", type=int, default=-1)
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scatter_years", action="store_true",
                   help="One pred-vs-obs scatter per evaluated year, in a "
                        "`scatter_year/` sub-folder next to --output_csv.")
    p.add_argument("--n_curves", type=int, default=0,
                   help="Number of per-site curves to plot in *_lai_curves_all.png. "
                        "0 = plot every site that produced predictions (default). "
                        "Otherwise a random subset of this size is drawn.")
    p.add_argument("--ablate_features", default="",
                   help="Comma list of features the model was TRAINED without. "
                        "Normally read from the checkpoint; use this to force it "
                        "when the checkpoint predates that bookkeeping.")
    p.add_argument("--pft_min_frac", type=float, default=0.05,
                   help="Show PFTs with fraction ≥ this threshold above each curve "
                        "subplot (default: 0.05 = 5%%).")
    return p.parse_args()


def parse_year_spec(spec: str):
    if not spec or spec.lower() == "all":
        return None
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def _load_pft_fracs(pft_dir: str, year: int, site_ids) -> dict:
    """Return {site_id: (N_PFT,) fraction vector} from PFTmap_{year}.nc, aligned
    by site_id. Empty dict if the file is missing (PFT annotation then skipped)."""
    import xarray as xr
    path = os.path.join(pft_dir, PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        return {}
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
    all_sites = np.asarray(ds["site_id"].values).astype(str)
    arr = da.transpose("pft", "site").values.astype(np.float32)   # (N_PFT, n_all)
    ds.close()
    idx_of = {s: i for i, s in enumerate(all_sites)}
    out = {}
    for s in map(str, site_ids):
        j = idx_of.get(s, -1)
        if j >= 0:
            out[s] = arr[:, j]
    return out


def _resolve_sites(args, ckpt, train_args) -> list:
    if args.selected_pixels:
        return load_selected_pixels(args.selected_pixels)
    if args.sites:
        return [s.strip() for s in args.sites.split(",") if s.strip()]
    mode = args.predict_sites
    train_pool = list(ckpt.get("train_site_ids", []))
    val_pool   = list(ckpt.get("val_site_ids", []))
    if mode == "val":
        return val_pool
    if mode == "train":
        return train_pool
    if mode == "all":
        return sorted(set(train_pool) | set(val_pool))
    # grid / test need a row/col range
    rmin = args.row_min if args.row_min >= 0 else int(train_args.get("row_min", 0))
    rmax = args.row_max if args.row_max >= 0 else int(train_args.get("row_max", 0))
    cmin = args.col_min if args.col_min >= 0 else int(train_args.get("col_min", 0))
    cmax = args.col_max if args.col_max >= 0 else int(train_args.get("col_max", 0))
    grid = generate_site_ids_from_range((rmin, rmax), (cmin, cmax))
    if mode == "grid":
        return grid
    if mode == "test":
        if not val_pool:
            raise RuntimeError("'test' mode needs val_site_ids in the checkpoint.")
        return sorted(set(grid) - set(val_pool))
    raise ValueError(mode)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = EasyDict(ckpt["args"])
    norm_stats     = ckpt.get("norm_stats", None)
    co2_lut        = ckpt.get("co2_lut", None)
    anomaly_clim   = ckpt.get("anomaly_clim", None) if ckpt.get("anomaly_mode") else None
    is_anomaly     = bool(ckpt.get("anomaly_mode", False))
    is_normalized  = norm_stats is not None
    lai_normalized = bool(ckpt.get("normalize_lai", True))
    is_pft_mixing  = bool(ckpt.get("pft_mixing", False))

    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Epoch         : {ckpt['epoch']}")
    print(f"Val RMSE / R² : {ckpt.get('val_rmse', float('nan')):.5f} / "
          f"{ckpt.get('val_r2', float('nan')):.4f}")
    print(f"Flags         : norm={is_normalized}, lai_norm={lai_normalized}, "
          f"pft_mixing={is_pft_mixing}, anomaly={is_anomaly}, "
          f"co2={co2_lut is not None}")

    features_dir = args.features_dir or train_args.get("features_dir", "")
    target_dir   = args.target_dir   or train_args.get("target_dir",   "")
    pft_dir      = args.pft_dir      or train_args.get("pft_dir",      "")
    if not features_dir or not target_dir or not pft_dir:
        raise ValueError("Provide --features_dir, --target_dir, --pft_dir "
                         "or use a checkpoint that stored them.")

    # ── 0.05°→0.1° parent map (optional) ── falls back to the checkpoint's
    #    training --parent_map so inference reads features the same way.
    parent_map_path = args.parent_map or train_args.get("parent_map", "")
    parent_map = None
    if parent_map_path:
        if not os.path.exists(parent_map_path):
            raise FileNotFoundError(parent_map_path)
        parent_map = load_parent01_map(parent_map_path)
        print(f"Parent map    : {parent_map_path}  "
              f"({len(parent_map):,} 0.05°→0.1° links)")

    # ── Rebuild model ──
    if is_pft_mixing:
        model = build_model_pft(train_args, norm_stats).to(device)
        wrapper_label = "PFTMixingWrapper"
    else:
        model = build_model(train_args).to(device)
        wrapper_label = "Every10DaysWrapper"
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model         : {train_args.get('type')} + {wrapper_label}")

    # ── Daily-trained model: it outputs 365 days, but the targets here are the
    #    36 dekads → sample the prediction at the dekad positions. ──
    is_daily = bool(train_args.get("daily_lai", False))
    obs_pos = torch.tensor(_OBS_POSITIONS, dtype=torch.long)
    if is_daily:
        print("Daily model   : sampling the 365-day output at the 36 dekads")

    # ── Trained with ablated features → zero the SAME channels at inference,
    #    otherwise the model sees inputs it never saw during training. ──
    abl_spec = (args.ablate_features
                or str(train_args.get("ablate_features", "") or ""))
    abl_feats = [f.strip() for f in abl_spec.split(",") if f.strip()]
    if abl_feats:
        abl_ch = [ALL_FEATURES.index(f) for f in abl_feats]

        def _ablate_hook(_mod, inputs):
            t = inputs[0]
            if not torch.is_tensor(t):
                return None
            t = t.clone()
            for c in abl_ch:
                t[:, c, :] = 0.0
            return (t,) + tuple(inputs[1:])

        model.register_forward_pre_hook(_ablate_hook)
        print(f"Ablation      : re-applied {abl_feats} (channels {abl_ch})")

    # ── Resolve sites and years ──
    site_ids = _resolve_sites(args, ckpt, train_args)
    if not site_ids:
        raise RuntimeError("Empty site list.")
    if 0 < args.n_predict_sites < len(site_ids):
        rng = np.random.RandomState(args.seed)
        site_ids = rng.choice(site_ids, size=args.n_predict_sites,
                              replace=False).tolist()
    print(f"Sites         : {len(site_ids):,}")

    years = parse_year_spec(args.predict_years)
    if years is None:
        val_years_str = str(train_args.get("val_years", "") or "")
        years = parse_year_spec(val_years_str)
        if not years:
            raise ValueError("Could not infer predict_years.")
    print(f"Years         : {years}")

    # ── Dataset ── (RAM-resident: vectorised read, avoids the per-site disk
    #    thrashing of the on-disk LAIDataset; bit-identical features.)
    print("Building dataset …")
    dataset = RamLAIDataset(
        features_dir=features_dir,
        target_dir=target_dir,
        pft_dir=pft_dir,
        years=years,
        site_ids=site_ids,
        seq_length=int(train_args.get("seq_length", 720)),
        norm_stats=norm_stats,
        anomaly_clim=anomaly_clim,
        co2_lut=co2_lut,
        normalize_lai=lai_normalized,
        verbose=True,
        parent_map=parent_map,
    )
    if len(dataset) == 0:
        raise RuntimeError("No prediction samples produced.")
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    # ── Inference ──
    denorm_lai = is_normalized and lai_normalized and not is_anomaly
    if denorm_lai:
        lai_mean = float(norm_stats["LAI"]["mean"])
        lai_std  = float(norm_stats["LAI"]["std"])

    rows = []
    print("Running inference …")
    with torch.no_grad():
        for i_batch, (features, targets) in enumerate(loader):
            preds = model(features.to(device)).cpu()
            if is_daily:
                preds = preds[:, :, obs_pos]            # (B, 1, 365) → (B, 1, 36)
            batch_start = i_batch * args.batch_size
            for j in range(preds.size(0)):
                idx  = batch_start + j
                meta = dataset.get_site_info(idx)
                pred_v = preds[j, 0, :].numpy()
                tgt_v  = targets[j, 0, :].numpy()

                if is_anomaly:
                    clim_vec = anomaly_clim.get(meta["site_id"])
                    if clim_vec is None:
                        continue
                    pred_real = pred_v + clim_vec
                    tgt_real  = tgt_v  + clim_vec
                    pred_norm = pred_v
                    tgt_norm  = tgt_v
                elif denorm_lai:
                    pred_real = pred_v * lai_std + lai_mean
                    tgt_real  = tgt_v  * lai_std + lai_mean
                    pred_norm = pred_v
                    tgt_norm  = tgt_v
                else:
                    pred_real = pred_v
                    tgt_real  = tgt_v
                    pred_norm = pred_v
                    tgt_norm  = tgt_v

                for k, (month, day, doy) in enumerate(_OBS_DATES):
                    rows.append({
                        "site_id":      meta["site_id"],
                        "year":         meta["year"],
                        "month":        month,
                        "day":          day,
                        "doy":          doy,
                        "lai_pred":     float(pred_real[k]),
                        "lai_obs":      float(tgt_real[k]),
                        "lai_pred_norm": float(pred_norm[k]),
                        "lai_obs_norm":  float(tgt_norm[k]),
                    })

    df = pd.DataFrame(rows)
    df["error"] = df["lai_pred"] - df["lai_obs"]

    valid = df.dropna(subset=["lai_obs"]).copy()
    print(f"\nObservations  : {len(valid):,} valid / {len(df):,} total")
    summary_lines: list = []
    if not valid.empty:
        obs  = valid["lai_obs"].values.astype(float)
        pred = valid["lai_pred"].values.astype(float)
        err  = pred - obs
        overall_rmse = float(np.sqrt(np.mean(err ** 2)))

        # Global R² (NSE): SS_tot uses the single global obs mean.
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Centered R² (NSE on dynamics): remove each site's own mean from obs
        # AND pred, so only intra-site temporal variation is scored.
        oc = valid["lai_obs"]  - valid.groupby("site_id")["lai_obs"].transform("mean")
        pc = valid["lai_pred"] - valid.groupby("site_id")["lai_pred"].transform("mean")
        ss_res_c = float(np.sum((oc - pc) ** 2))
        ss_tot_c = float(np.sum(oc ** 2))
        r2_centered = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 0 else float("nan")

        # Pearson² (squared linear correlation; ignores bias/scale).
        if obs.std() > 1e-9 and pred.std() > 1e-9:
            r = float(np.corrcoef(obs, pred)[0, 1])
            r2_pearson = r * r
        else:
            r2_pearson = float("nan")

        # Per-site R² (NSE), sites with enough valid points and obs variance.
        site_r2s = []
        for _, g in valid.groupby("site_id"):
            o = g["lai_obs"].values
            p = g["lai_pred"].values
            sstot = float(np.sum((o - o.mean()) ** 2))
            if len(o) < 5 or sstot <= 0:
                continue
            site_r2s.append(1.0 - float(np.sum((p - o) ** 2)) / sstot)

        # ── R² decomposition (paper-style): overall / site-year / interannual ──
        #   overall     : one NSE over all (site, year, dekad) points (= r2_global).
        #   site-year   : NSE within each site-year (its 36 dekads) → mean ± std.
        #   interannual : NSE across site-years at a FIXED day-of-year → mean ± std.
        def _nse(o, p, nmin=5):
            o = np.asarray(o, float); p = np.asarray(p, float)
            sstot = float(np.sum((o - o.mean()) ** 2))
            if len(o) < nmin or sstot <= 0:
                return None
            return 1.0 - float(np.sum((p - o) ** 2)) / sstot

        siteyear_r2 = [r for _, g in valid.groupby(["site_id", "year"])
                       if (r := _nse(g["lai_obs"], g["lai_pred"])) is not None]
        interann_r2 = [r for _, g in valid.groupby("doy")
                       if (r := _nse(g["lai_obs"], g["lai_pred"])) is not None]

        # ── Build summary (printed and saved to <output>_metrics.txt) ──
        summary_lines.append("── Pooled metrics ──")
        summary_lines.append(f"  Global    R²  (NSE)         : {r2_global:+.4f}  "
                             f"(includes inter-site mean differences — generous)")
        summary_lines.append(f"  Centered  R²  (NSE on dyn.) : {r2_centered:+.4f}  "
                             f"(intra-site variance only — fair test of dynamics)")
        summary_lines.append(f"  Pearson²  R²  (correlation) : {r2_pearson:+.4f}  "
                             f"(linear alignment only — ignores bias / scale; matches papers)")
        summary_lines.append(f"  Overall RMSE                : {overall_rmse:.4f}")

        if site_r2s:
            arr = np.array(site_r2s)
            n_pos = int(np.sum(arr > 0))
            summary_lines.append("")
            summary_lines.append(
                f"── Per-site R² (NSE) distribution ({len(arr)} sites) ──")
            summary_lines.append(f"  Median         : {np.median(arr):+.4f}")
            summary_lines.append(f"  Mean ± std     : {np.mean(arr):+.4f} ± {np.std(arr):.4f}")
            summary_lines.append(f"  5th  percentile: {np.percentile(arr, 5):+.4f}")
            summary_lines.append(f"  95th percentile: {np.percentile(arr, 95):+.4f}")
            summary_lines.append(f"  Sites with R²>0: {n_pos:,} / {len(arr):,} "
                                 f"({100.0 * n_pos / len(arr):.1f}%)")

        # ── R² decomposition (overall / site-year / interannual) ──
        sy = np.array(siteyear_r2, dtype=float)
        ia = np.array(interann_r2, dtype=float)
        r2_items = [
            ("Overall      R²", r2_global, None, ""),
            ("Site-year    R²", float(sy.mean()) if sy.size else float("nan"),
             float(sy.std()) if sy.size else None, f" (n={sy.size} site-years)"),
            ("Interannual  R²", float(ia.mean()) if ia.size else float("nan"),
             float(ia.std()) if ia.size else None, f" (n={ia.size} DOYs)"),
        ]
        finite = [m for _, m, _, _ in r2_items if np.isfinite(m)]
        best = max(finite) if finite else None
        summary_lines.append("")
        summary_lines.append(
            "── R² decomposition (overall / site-year / interannual) ──")
        for label, m, s, note in r2_items:
            txt = f"{m:+.4f}" if s is None else f"{m:+.4f} ± {s:.4f}"
            if best is not None and np.isfinite(m) and m == best:
                txt = f"**{txt}**"                      # best value in bold
            summary_lines.append(f"  {label} : {txt}{note}")

        print("\n" + "\n".join(summary_lines))

    # ── Save CSV ──
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions → {args.output_csv}")

    base = os.path.splitext(args.output_csv)[0]
    if summary_lines:
        metrics_path = base + "_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"Metrics → {metrics_path}")
    if not valid.empty:
        plot_pred_vs_obs(
            valid["lai_pred"].values, valid["lai_obs"].values,
            filename=base + "_pred_vs_obs.png",
            title=f"PhenoNN — Predicted vs observed LAI "
                  f"({len(valid):,} pts)",
        )
        if is_anomaly:
            plot_pred_vs_obs(
                valid["lai_pred_norm"].values, valid["lai_obs_norm"].values,
                filename=base + "_pred_vs_obs_anomaly.png",
                title=f"ΔLAI anomaly ({len(valid):,} pts)",
                xlabel="Observed ΔLAI", ylabel="Predicted ΔLAI",
            )
        elif denorm_lai:
            plot_pred_vs_obs(
                valid["lai_pred_norm"].values, valid["lai_obs_norm"].values,
                filename=base + "_pred_vs_obs_norm.png",
                title=f"Normalized LAI ({len(valid):,} pts)",
                xlabel="Observed LAI (normalized)",
                ylabel="Predicted LAI (normalized)",
            )
        if valid["site_id"].nunique() >= 3:
            plot_gcc_curves(
                valid, filename=base + "_lai_curves.png",
                site_col="site_id", year_col="year", doy_col="doy",
            )

        # ── Grid of per-site curves: all sites (--n_curves 0) or a random
        #    subset of --n_curves sites. Each subplot title lists its PFTs with
        #    fraction ≥ --pft_min_frac (read from PFTmap of the first year). ──
        sites_all = valid["site_id"].unique()
        if args.n_curves and 0 < args.n_curves < len(sites_all):
            rng_c = np.random.RandomState(args.seed)
            keep = set(rng_c.choice(sites_all, size=args.n_curves, replace=False))
            curve_df = valid[valid["site_id"].isin(keep)]
        else:
            curve_df = valid
        n_curve_sites = curve_df["site_id"].nunique()
        if n_curve_sites > 400:
            print(f"[warn] plotting {n_curve_sites:,} site curves in one grid — "
                  f"use --n_curves to subsample.")
        pft_by_site = _load_pft_fracs(pft_dir, years[0],
                                      curve_df["site_id"].unique())
        plot_gcc_curves_all(
            curve_df, filename=base + "_lai_curves_all.png",
            site_col="site_id", year_col="year", doy_col="doy",
            pft_by_site=pft_by_site, pft_names=PFT_NAMES,
            pft_min_frac=args.pft_min_frac,
        )

    if args.scatter_years and not valid.empty:
        year_dir = os.path.join(os.path.dirname(base) or ".", "scatter_year")
        os.makedirs(year_dir, exist_ok=True)
        for yr in sorted(int(y) for y in valid["year"].unique()):
            sub = valid[valid["year"] == yr]
            if len(sub) < 2:
                continue
            ss_res = float(np.sum((sub["lai_obs"] - sub["lai_pred"]) ** 2))
            ss_tot = float(np.sum((sub["lai_obs"] - sub["lai_obs"].mean()) ** 2))
            r2_y   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse_y = float(np.sqrt(np.mean((sub["lai_pred"] - sub["lai_obs"]) ** 2)))
            plot_pred_vs_obs(
                sub["lai_pred"].values, sub["lai_obs"].values,
                filename=os.path.join(year_dir, f"year_{yr}_pred_vs_obs.png"),
                title=f"Year {yr} — R²={r2_y:+.4f} RMSE={rmse_y:.4f} "
                      f"n={len(sub):,}",
            )

    print("Done.")


if __name__ == "__main__":
    main()
