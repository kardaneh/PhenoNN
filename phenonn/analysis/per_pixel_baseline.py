#!/usr/bin/env python3
"""
per_pixel_baseline.py
=====================

Per-pixel baseline: fit an INDEPENDENT small neural net on each site, using only
that site's own time series (one sample per year), and evaluate it on held-out
years. This is the opposite of the global PhenoNN model (one net shared across
all sites) — here every pixel gets its own weights.

Rationale / caveats
-------------------
Per site there are only ~n_train_years samples (one (features→36-dekad LAI) pair
per year), so the per-pixel net MUST be low-capacity and will still tend to
overfit. This is meant as a BASELINE to compare against the global model, not a
production approach. The PFT / cyclic channels are constant within a single site
so they carry no per-site information (the net simply ignores them); CO2 is the
only non-weather channel that varies across a site's years.

Pipeline
--------
  1. One RamLAIDataset over ALL selected sites × (train ∪ val) years  → RAM once.
  2. For each site: gather its train-year and val-year sample indices.
  3. Build a fresh small model (build_model, --type), train on the site's train
     samples, predict its val samples.
  4. Denormalize LAI, accumulate per (site, year, dekad) rows, report per-site
     and pooled metrics (same definitions as prediction/predict.py).

Usage
-----
    python -m prediction.per_pixel_baseline \\
        --features_dir /data/sbarbu/PhenoNN/data/era5_1992_2019 \\
        --target_dir   /data/sbarbu/PhenoNN/data/pixelset_10%_1992_2019/LAI_pixelset \\
        --pft_dir      /data/sbarbu/PhenoNN/data/pixelset_10%_1992_2019/PFT_pixelset \\
        --stats_path   /data/sbarbu/PhenoNN/data/norm_stats_1992_2019.json \\
        --co2_path     /data/co2_1700_2023_TRENDYv2024.txt \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_10%.nc \\
        --n_sites 200 --type lstm --hidden_size 32 --num_layers 1 \\
        --num_epochs 200 --output_csv runs/per_pixel_baseline.csv
"""

import argparse
import datetime
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from phenonn.data.lai_dataset import RamLAIDataset, load_co2_lut, load_selected_pixels
from phenonn.utils.model_loader import build_model


# 36 obs (month, day, doy) — non-leap year, same convention as predict.py
_OBS_DATES = [
    (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
    for m in range(1, 13)
    for d in [5, 15, 25]
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_dir", required=True)
    p.add_argument("--target_dir",   required=True)
    p.add_argument("--pft_dir",      required=True)
    p.add_argument("--stats_path",   required=True,
                   help="norm_stats.json (z-score stats).")
    p.add_argument("--co2_path", default="")
    p.add_argument("--selected_pixels", default="",
                   help="selected_pixels*.nc — sites to run on.")
    p.add_argument("--sites", default="",
                   help="Comma-separated explicit site_ids (overrides "
                        "--selected_pixels).")
    p.add_argument("--n_sites", type=int, default=0,
                   help="If >0, random subsample of this many sites.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_years", default="1992-2009")
    p.add_argument("--val_years",   default="2010-2018")
    # small per-site model
    p.add_argument("--type", default="lstm")
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--seq_length", type=int, default=720)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--normalize_lai", type=int, default=1)
    p.add_argument("--output_csv", default="per_pixel_baseline.csv")
    return p.parse_args()


def parse_year_spec(spec: str):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def _model_args(args) -> SimpleNamespace:
    """Namespace with every field build_model may read, so any --type works."""
    return SimpleNamespace(
        type=args.type, seq_length=args.seq_length,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout2=args.dropout,
        # defaults for the other architectures (unused by lstm/fcn/linear*)
        embed_size=64, nhead=4, forward_expansion=4, d_model=None,
        dropout1=0.05, dropout_att=0.0,
        feed_forward_trans1=4, feed_forward_trans2=4,
        n_attn_blocks=2, stress_dim=8, num_layers1=2, d_layers=2,
    )


def train_site(model, feats, tgts, device, epochs, lr, delta):
    """Full-batch train on a single site's samples. `tgts` may contain NaN
    (missing dekads) → masked out of the loss."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.HuberLoss(delta=delta)
    mask = torch.isfinite(tgts)
    tgts0 = torch.nan_to_num(tgts, nan=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(feats)                          # (n, 1, 36)
        loss = lossf(pred[mask], tgts0[mask])
        loss.backward()
        opt.step()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    with open(args.stats_path) as f:
        norm_stats = json.load(f)
    co2_lut = load_co2_lut(args.co2_path) if args.co2_path else None
    lai_normalized = bool(args.normalize_lai)

    train_years = parse_year_spec(args.train_years)
    val_years   = parse_year_spec(args.val_years)
    all_years   = sorted(set(train_years) | set(val_years))

    # ── Resolve sites ──
    if args.sites:
        site_ids = [s.strip() for s in args.sites.split(",") if s.strip()]
    elif args.selected_pixels:
        site_ids = load_selected_pixels(args.selected_pixels)
    else:
        raise ValueError("Provide --selected_pixels or --sites.")
    if 0 < args.n_sites < len(site_ids):
        rng = np.random.RandomState(args.seed)
        site_ids = rng.choice(site_ids, size=args.n_sites, replace=False).tolist()
    print(f"Sites        : {len(site_ids):,}")
    print(f"Train years  : {train_years}")
    print(f"Val years    : {val_years}")
    print(f"Model/pixel  : {args.type} (hidden={args.hidden_size}, "
          f"layers={args.num_layers})")

    # ── One RAM load for everything ──
    print("Building dataset (one load for all sites × years) …")
    ds = RamLAIDataset(
        features_dir=args.features_dir, target_dir=args.target_dir,
        pft_dir=args.pft_dir, years=all_years, site_ids=site_ids,
        seq_length=args.seq_length, norm_stats=norm_stats,
        co2_lut=co2_lut, normalize_lai=lai_normalized, verbose=True,
    )
    if len(ds) == 0:
        raise RuntimeError("No samples produced.")

    denorm = lai_normalized and "LAI" in norm_stats
    lai_mean = float(norm_stats["LAI"]["mean"]) if denorm else 0.0
    lai_std  = float(norm_stats["LAI"]["std"])  if denorm else 1.0
    margs = _model_args(args)

    # ── Per-site train + eval ──
    rows = []
    n_done = n_skipped = 0
    for si, site in enumerate(site_ids):
        train_idx = [ds.sample_index[(site, y)] for y in train_years
                     if (site, y) in ds.sample_index]
        val_pairs = [(y, ds.sample_index[(site, y)]) for y in val_years
                     if (site, y) in ds.sample_index]
        if not train_idx or not val_pairs:
            n_skipped += 1
            continue

        feats = torch.stack([ds[i][0] for i in train_idx]).to(device)   # (nt,C,L)
        tgts  = torch.stack([ds[i][1] for i in train_idx]).to(device)   # (nt,1,36)
        if not torch.isfinite(tgts).any():          # no observed LAI to fit on
            n_skipped += 1
            continue

        model = build_model(margs)
        train_site(model, feats, tgts, device,
                   args.num_epochs, args.lr, args.huber_delta)

        vfeats = torch.stack([ds[i][0] for _, i in val_pairs]).to(device)
        model.eval()
        with torch.no_grad():
            vpred = model(vfeats).cpu().numpy()[:, 0, :]                # (nv, 36)
        for (year, idx), pred_v in zip(val_pairs, vpred):
            obs_v = ds[idx][1].numpy()[0, :]                           # (36,)
            pred_real = pred_v * lai_std + lai_mean
            obs_real  = obs_v * lai_std + lai_mean
            for k, (month, day, doy) in enumerate(_OBS_DATES):
                rows.append({
                    "site_id": site, "year": int(year),
                    "month": month, "day": day, "doy": doy,
                    "lai_pred": float(pred_real[k]),
                    "lai_obs":  float(obs_real[k]),
                })
        n_done += 1
        if (si + 1) % 25 == 0 or si == len(site_ids) - 1:
            print(f"  trained {n_done:,} sites  ({si + 1}/{len(site_ids)})")

    print(f"\nTrained {n_done:,} models | skipped {n_skipped:,} "
          f"(no train or no val sample).")
    if not rows:
        raise RuntimeError("No predictions produced.")

    df = pd.DataFrame(rows)
    df["error"] = df["lai_pred"] - df["lai_obs"]
    valid = df.dropna(subset=["lai_obs"]).copy()

    # ── Metrics (same definitions as prediction/predict.py) ──
    obs  = valid["lai_obs"].values.astype(float)
    pred = valid["lai_pred"].values.astype(float)
    err  = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    oc = valid["lai_obs"]  - valid.groupby("site_id")["lai_obs"].transform("mean")
    pc = valid["lai_pred"] - valid.groupby("site_id")["lai_pred"].transform("mean")
    ss_res_c = float(np.sum((oc - pc) ** 2))
    ss_tot_c = float(np.sum(oc ** 2))
    r2_centered = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 0 else float("nan")
    if obs.std() > 1e-9 and pred.std() > 1e-9:
        r2_pearson = float(np.corrcoef(obs, pred)[0, 1]) ** 2
    else:
        r2_pearson = float("nan")

    site_r2s = []
    for _, g in valid.groupby("site_id"):
        o, p = g["lai_obs"].values, g["lai_pred"].values
        sstot = float(np.sum((o - o.mean()) ** 2))
        if len(o) >= 5 and sstot > 0:
            site_r2s.append(1.0 - float(np.sum((p - o) ** 2)) / sstot)

    lines = ["── Per-pixel baseline (pooled over sites) ──",
             f"  Overall RMSE               : {rmse:.4f}",
             f"  Centered  R² (NSE on dyn.) : {r2_centered:+.4f}",
             f"  Pearson²  R² (correlation) : {r2_pearson:+.4f}",
             f"  Valid points               : {len(valid):,} / {len(df):,}"]
    if site_r2s:
        arr = np.array(site_r2s)
        lines += ["", f"── Per-site R² (NSE) over {len(arr)} sites ──",
                  f"  Median          : {np.median(arr):+.4f}",
                  f"  Mean            : {np.mean(arr):+.4f}",
                  f"  5th percentile  : {np.percentile(arr, 5):+.4f}",
                  f"  95th percentile : {np.percentile(arr, 95):+.4f}",
                  f"  Sites with R²>0 : {int(np.sum(arr > 0)):,} / {len(arr):,}"]
    print("\n" + "\n".join(lines))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    with open(os.path.splitext(args.output_csv)[0] + "_metrics.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nPredictions → {args.output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
