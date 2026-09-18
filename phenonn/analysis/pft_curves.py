#!/usr/bin/env python3
"""
PFT decomposition visualization (PhenoNN pixelset pipeline).

Loads a model trained with `--pft_mixing` (phenonn.training.train_global / phenonn.training.train_full_ram,
`model_kind='phenon_big'`, `pft_mixing=True`) and plots the 15 "pure" LAI curves
— one per Plant Functional Type — for each selected site and year.

Each subplot shows:
  - 15 colored lines : pure LAI per PFT (LAI if that PFT covered 100% of the
    cell), denormalized to physical units
  - thick black line : the area-weighted prediction (∑_k frac_k · pure_k)
  - red markers      : observed LAI at the 36 obs days

PFTs whose fraction in the cell is below --min_frac are drawn in thin grey so
they stay visible but do not crowd the legend.

Data come from the *pixelset* files (same as training), read via RamLAIDataset.

Usage
-----
    python -m prediction.pft_curves \\
        --checkpoint runs/Base1/checkpoints/best_model.pth \\
        --features_dir /data/sbarbu/PhenoNN/data/era5_10% \\
        --target_dir   /data/sbarbu/PhenoNN/data/pixelset_10%/LAI_pixelset \\
        --pft_dir      /data/sbarbu/PhenoNN/data/pixelset_10%/PFT_pixelset \\
        --n_sites 6 --year 2005 \\
        --output runs/Base1/pft_curves.png

    # explicit sites, or a selected_pixels*.nc subset:
    #   --sites pix_0176_00563,pix_0177_02847
    #   --selected_pixels /data/.../selected_pixels_PFT9.nc
"""

import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phenonn.utils.config import N_PFT, PFT_NAMES
from phenonn.data.lai_dataset import RamLAIDataset, load_selected_pixels
from phenonn.utils.model_loader import build_model_pft
from phenonn.utils.utils import EasyDict


# 36 obs DOYs (5th, 15th, 25th of every month in a non-leap year)
_OBS_DOYS = np.array([
    datetime.date(2001, m, d).timetuple().tm_yday
    for m in range(1, 13) for d in [5, 15, 25]
])
_MONTH_MID_DOYS = [datetime.date(2001, m, 15).timetuple().tm_yday
                   for m in range(1, 13)]
_MONTH_ABBR = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# Saturated palette (ColorBrewer Set1 + Dark2 + extras) so every PFT gets a
# vivid, distinguishable colour.
PFT_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3",
    "#A65628", "#F781BF", "#1B9E77", "#D95F02", "#7570B3",
    "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#000080",
]


def parse_args():
    p = argparse.ArgumentParser(description="PFT decomposition visualization (PhenoNN)")
    p.add_argument("--checkpoint", required=True,
                   help="best_model.pth from a --pft_mixing training run.")
    p.add_argument("--features_dir", default="",
                   help="ERA5 pixelset dir (default: path stored in the checkpoint).")
    p.add_argument("--target_dir", default="",
                   help="LAI pixelset dir (default: from checkpoint).")
    p.add_argument("--pft_dir", default="",
                   help="PFT pixelset dir (default: from checkpoint).")
    p.add_argument("--selected_pixels", default="",
                   help="Pick sites from a selected_pixels*.nc (subsampled to "
                        "--n_sites). Overrides --n_sites' default val pool.")
    p.add_argument("--sites", default="",
                   help="Comma-separated site IDs. Overrides --selected_pixels.")
    p.add_argument("--n_sites", type=int, default=6,
                   help="Random sites from the checkpoint val split when neither "
                        "--sites nor --selected_pixels is given (default 6).")
    p.add_argument("--year", type=int, default=None,
                   help="Year to plot (default: val_years from the checkpoint).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_frac", type=float, default=0.02,
                   help="PFTs with fraction < min_frac drawn in grey (no legend).")
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--output", default="pft_curves.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = EasyDict(ckpt["args"])
    norm_stats = ckpt.get("norm_stats")

    if not bool(ckpt.get("pft_mixing", False)):
        raise ValueError(
            f"This script needs a --pft_mixing checkpoint. Got "
            f"model_kind={ckpt.get('model_kind')!r}, "
            f"pft_mixing={ckpt.get('pft_mixing')}."
        )
    if bool(ckpt.get("anomaly_mode", False)):
        print("[warn] checkpoint is anomaly_mode — LAI curves will be in anomaly "
              "space (obs − climatology), not physical LAI.")

    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Epoch       : {ckpt.get('epoch')}    "
          f"val_R²: {ckpt.get('val_r2', float('nan')):.4f}    "
          f"val_RMSE: {ckpt.get('val_rmse', float('nan')):.5f}")

    # ── Rebuild model ──
    model = build_model_pft(train_args, norm_stats).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Resolve data dirs (CLI > checkpoint args) ──
    features_dir = args.features_dir or train_args.get("features_dir", "")
    target_dir   = args.target_dir   or train_args.get("target_dir", "")
    pft_dir      = args.pft_dir       or train_args.get("pft_dir", "")
    if not (features_dir and target_dir and pft_dir):
        raise ValueError("Provide --features_dir, --target_dir and --pft_dir "
                         "(or use a checkpoint that stored them).")

    # ── Years ──
    if args.year:
        years = [args.year]
    else:
        vy = str(train_args.get("val_years", "") or "")
        if "-" in vy and "," not in vy:
            a, b = vy.split("-")
            years = list(range(int(a), int(b) + 1))
        elif vy:
            years = [int(y) for y in vy.split(",")]
        else:
            raise ValueError("Could not infer year to plot; pass --year.")

    # ── Sites ──
    rng = np.random.RandomState(args.seed)
    if args.sites:
        site_ids = [s.strip() for s in args.sites.split(",") if s.strip()]
    else:
        if args.selected_pixels:
            pool = load_selected_pixels(args.selected_pixels)
        else:
            pool = list(ckpt.get("val_site_ids") or [])
            if not pool:
                raise ValueError("No val_site_ids in checkpoint; pass --sites or "
                                 "--selected_pixels.")
        n = min(args.n_sites, len(pool))
        site_ids = rng.choice(pool, size=n, replace=False).tolist()
    print(f"Sites       : {len(site_ids)}  |  years: {years}")

    # ── Dataset (pixelset, RAM) ──
    ds = RamLAIDataset(
        features_dir=features_dir, target_dir=target_dir, pft_dir=pft_dir,
        years=years, site_ids=site_ids,
        seq_length=int(train_args.get("seq_length", 720)),
        norm_stats=norm_stats,
        anomaly_clim=ckpt.get("anomaly_clim"),
        co2_lut=ckpt.get("co2_lut"),
        normalize_lai=bool(ckpt.get("normalize_lai", True)),
        verbose=True,
    )
    if len(ds) == 0:
        raise RuntimeError("No samples for the selected sites/year.")
    print(f"Samples     : {len(ds)}")
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)

    # ── Denormalization of LAI back to physical units ──
    lai_normalized = bool(ckpt.get("normalize_lai", True))
    if lai_normalized and norm_stats is not None and "LAI" in norm_stats:
        lai_mean = float(norm_stats["LAI"]["mean"])
        lai_std  = float(norm_stats["LAI"]["std"])
    else:
        lai_mean, lai_std = 0.0, 1.0

    # ── Inference: replicate PFTMixingWrapper to expose the 15 pure-LAI channels ──
    with torch.no_grad():
        features, targets = next(iter(loader))
        features = features.to(device)

        # In meteo_only mode the base model was built to see only the meteo/
        # cyclic/co2 channels; strip the PFT block exactly like the wrapper does.
        base_in = (features[:, : model.pft_start, :]
                   if getattr(model, "meteo_only", False) else features)
        raw = model.base_model(base_in)                          # (B, 15, L)
        if getattr(model, "nonneg", False):
            # Mirror the wrapper's soft floor so the shown pure LAI matches the
            # values the model actually mixes.
            raw = model.nonneg_z0 + torch.nn.functional.softplus(
                raw - model.nonneg_z0)
        if getattr(model, "sparse_output", False):
            lai_pure_norm = raw                                  # (B, 15, 36)
        else:
            lai_pure_norm = raw[:, :, -365:][:, :, model.obs_positions]
        # Mirror the wrapper's per-PFT override (PFT-1 bare soil → 0).
        lai_pure_norm = lai_pure_norm * model.lai_pft_mask + model.lai_pft_bias

        lai_pred_norm = model(features)                          # (B, 1, 36)

    lai_pure_real = lai_pure_norm.cpu().numpy() * lai_std + lai_mean      # (B,15,36)
    lai_pred_real = (lai_pred_norm.cpu().numpy() * lai_std + lai_mean).squeeze(1)
    lai_obs_real  = (targets.numpy() * lai_std + lai_mean).squeeze(1)

    # PFT physical fractions — recovered exactly as the wrapper does.
    pf0 = int(model.pft_start)
    pft_means = model.pft_means.cpu().numpy().reshape(1, N_PFT)
    pft_stds  = model.pft_stds.cpu().numpy().reshape(1, N_PFT)
    pft_norm  = features[:, pf0:pf0 + N_PFT, -1].cpu().numpy()
    pft_real  = np.clip(pft_norm * pft_stds + pft_means, 0.0, None)
    pft_real  = pft_real / pft_real.sum(axis=1, keepdims=True).clip(1e-6)

    # ── Plot ──
    n_samples = len(ds)
    ncols = min(args.ncols, n_samples)
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows),
        squeeze=False, constrained_layout=True,
    )

    def cmap(k):
        return PFT_PALETTE[k % len(PFT_PALETTE)]

    for i in range(n_samples):
        ax = axes[i // ncols, i % ncols]
        meta = ds.get_site_info(i)
        site, yr = meta["site_id"], meta["year"]
        fracs = pft_real[i]

        # Weighted-sum prediction skill vs observation (NaN-aware, 36 obs days)
        o, p = lai_obs_real[i], lai_pred_real[i]
        m = np.isfinite(o) & np.isfinite(p)
        if m.any():
            skill = (f"MAE={np.mean(np.abs(p[m] - o[m])):.2f}  "
                     f"MSE={np.mean((p[m] - o[m]) ** 2):.2f}")
        else:
            skill = "MAE=N/A  MSE=N/A"

        # Background: PFTs below threshold in thin grey.
        for k in range(N_PFT):
            if fracs[k] < args.min_frac:
                ax.plot(_OBS_DOYS, lai_pure_real[i, k],
                        color="lightgrey", linewidth=0.6, alpha=0.35, zorder=1)

        # Foreground: significant PFTs, sorted by descending fraction.
        significant = sorted(
            [k for k in range(N_PFT) if fracs[k] >= args.min_frac],
            key=lambda k: -fracs[k],
        )
        for k in significant:
            ax.plot(_OBS_DOYS, lai_pure_real[i, k],
                    color=cmap(k), linewidth=2.0, alpha=1.0,
                    label=f"{PFT_NAMES[k]}  ({fracs[k] * 100:.0f}%)", zorder=2)

        ax.plot(_OBS_DOYS, lai_pred_real[i], color="black", linewidth=2.2,
                label="Weighted sum (pred)", zorder=3)
        ax.plot(_OBS_DOYS, lai_obs_real[i], "o", color="red", markersize=4.5,
                label="Observed", zorder=4)

        ax.set_title(f"{site}  —  {yr}\n{skill}", fontsize=10)
        ax.set_xticks(_MONTH_MID_DOYS)
        ax.set_xticklabels(_MONTH_ABBR, fontsize=8)
        ax.set_xlim(0, 365)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("LAI", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5, loc="upper left",
                  frameon=True, framealpha=0.85, handlelength=1.6)

    for i in range(n_samples, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle(
        "Per-PFT pure-LAI curves  vs  weighted-sum prediction  vs  observation",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
