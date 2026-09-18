#!/usr/bin/env python3
"""
compare_mixing.py — is a NO-mixing model linear in the PFT fractions?

From a trained NO-pft_mixing checkpoint (the model ingests the 15 PFT fractions
as input channels and predicts the cell's total LAI directly), for every eval
cell we compute two LAI predictions:

  • direct  = model(x)                          — the cell with its real fractions
  • mix     = Σ_i p_i · model(x | PFT := pure i) — re-run the SAME weather with the
              PFT vector forced to pure PFT i (one-hot, re-normalized), weighted by
              the cell's real fractions p_i (denormalized, clamped ≥0, renormalized
              to sum 1 so the linear combination is exact in normalized-LAI space).

If direct ≈ mix, the model has learned an ~linear PFT mixing → validates the
additive assumption behind PFTMixingWrapper / the greedy unmixing. It also reports
per-PFT skill on ~pure cells (dominant fraction ≥ --purity) vs the observations.

No change to phenonn/. Heavy: 15 extra forward passes per sample — use
--n_predict_sites to subsample.

    python -m phenonn.analysis.pft_linearity.compare_mixing \\
        --checkpoint runs_sweep/<nomix_run>/checkpoints/best_model.pth \\
        --predict_sites val --n_predict_sites 5000 --output_dir runs_pft_linearity
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from phenonn.utils.config import N_PFT, PFT_START
from phenonn.data.lai_dataset import RamLAIDataset, load_co2_lut, load_parent01_map
from phenonn.utils.model_loader import build_model, build_model_pft
from phenonn.utils.utils import EasyDict
from phenonn.prediction.predict import parse_year_spec, _resolve_sites

_OBS_DOY = [datetime.date(2001, m, d).timetuple().tm_yday
            for m in range(1, 13) for d in (5, 15, 25)]


def _pft_norm_arrays(norm_stats):
    """(mean, std, has) arrays over the 15 PFT fraction channels."""
    mean = np.zeros(N_PFT, np.float32); std = np.ones(N_PFT, np.float32)
    has = np.zeros(N_PFT, bool)
    if norm_stats is not None:
        for k in range(1, N_PFT + 1):
            key = f"pft{k}_frac"
            if key in norm_stats:
                mean[k - 1] = float(norm_stats[key]["mean"])
                std[k - 1] = max(float(norm_stats[key]["std"]), 1e-8)
                has[k - 1] = True
    return mean, std, has


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="NO-mixing checkpoint (gives `direct` and `mix`).")
    p.add_argument("--checkpoint_mixing", default="",
                   help="Optional PFT-mixing checkpoint, evaluated on the SAME "
                        "cells → extra column `lai_pftmix`. Must share the same "
                        "seq_length and feature norm_stats as --checkpoint.")
    p.add_argument("--features_dir", default=""); p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default=""); p.add_argument("--parent_map", default="")
    p.add_argument("--predict_sites", default="val",
                   choices=["val", "train", "all", "grid", "test"])
    p.add_argument("--selected_pixels", default=""); p.add_argument("--sites", default="")
    p.add_argument("--n_predict_sites", type=int, default=5000)
    p.add_argument("--predict_years", default="")
    p.add_argument("--row_min", type=int, default=-1); p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=-1); p.add_argument("--col_max", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=64); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--purity", type=float, default=0.9,
                   help="Dominant-PFT fraction to call a cell ~pure (default 0.9).")
    p.add_argument("--output_dir", default="runs_pft_linearity")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = EasyDict(ckpt["args"])
    norm_stats = ckpt.get("norm_stats", None)
    co2_lut = ckpt.get("co2_lut", None)
    lai_normalized = bool(ckpt.get("normalize_lai", True))
    is_pft_mixing = bool(ckpt.get("pft_mixing", False))
    if is_pft_mixing:
        print("[warn] checkpoint IS pft_mixing → mix == direct by construction; "
              "this test is meant for a NO-mixing model.")

    features_dir = args.features_dir or train_args.get("features_dir", "")
    target_dir = args.target_dir or train_args.get("target_dir", "")
    pft_dir = args.pft_dir or train_args.get("pft_dir", "")
    parent_map_path = args.parent_map or train_args.get("parent_map", "")
    parent_map = load_parent01_map(parent_map_path) if parent_map_path else None

    model = (build_model_pft(train_args, norm_stats) if is_pft_mixing
             else build_model(train_args)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    # ── Optional PFT-mixing model, evaluated on the SAME cells / same inputs ──
    model_mix, lm2, ls2 = None, 0.0, 1.0
    if args.checkpoint_mixing:
        ck2 = torch.load(args.checkpoint_mixing, map_location=device, weights_only=False)
        ta2 = EasyDict(ck2["args"]); ns2 = ck2.get("norm_stats", None)
        mix2 = bool(ck2.get("pft_mixing", False))
        if not mix2:
            print("[warn] --checkpoint_mixing is NOT a pft_mixing model.")
        # One shared dataset ⇒ both models must see identically-scaled inputs.
        if int(ta2.get("seq_length", 720)) != int(train_args.get("seq_length", 720)):
            raise SystemExit("seq_length differs between the two checkpoints — "
                             "they cannot share one dataset.")
        if (ns2 is None) != (norm_stats is None):
            raise SystemExit("feature normalization differs between checkpoints.")
        if ns2 is not None and norm_stats is not None:
            bad = [k for k in norm_stats if k in ns2 and (
                abs(float(ns2[k]["mean"]) - float(norm_stats[k]["mean"])) > 1e-9 or
                abs(float(ns2[k]["std"]) - float(norm_stats[k]["std"])) > 1e-9)]
            if bad:
                raise SystemExit(f"norm_stats differ for {bad[:5]}… — the two "
                                 f"checkpoints used different scaling.")
        model_mix = (build_model_pft(ta2, ns2) if mix2 else build_model(ta2)).to(device)
        model_mix.load_state_dict(ck2["model_state_dict"]); model_mix.eval()
        if bool(ck2.get("normalize_lai", True)) and ns2 is not None and "LAI" in ns2:
            lm2, ls2 = float(ns2["LAI"]["mean"]), float(ns2["LAI"]["std"])
        print(f"Mixing model  : {args.checkpoint_mixing}  ({ta2.get('type')}, "
              f"pft_mixing={mix2})")

    site_ids = _resolve_sites(args, ckpt, train_args)
    if 0 < args.n_predict_sites < len(site_ids):
        rng = np.random.RandomState(args.seed)
        site_ids = rng.choice(site_ids, size=args.n_predict_sites, replace=False).tolist()
    years = parse_year_spec(args.predict_years) or parse_year_spec(
        str(train_args.get("val_years", "") or ""))
    print(f"Sites {len(site_ids):,} | years {years}")

    dataset = RamLAIDataset(
        features_dir=features_dir, target_dir=target_dir, pft_dir=pft_dir,
        years=years, site_ids=site_ids,
        seq_length=int(train_args.get("seq_length", 720)), norm_stats=norm_stats,
        anomaly_clim=None, co2_lut=co2_lut, normalize_lai=lai_normalized,
        verbose=True, parent_map=parent_map)
    if len(dataset) == 0:
        raise RuntimeError("No samples.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mean, std, has = _pft_norm_arrays(norm_stats)
    # Normalized pure-PFT input vectors: (15 pfts, 15 channels).
    pure_z = np.zeros((N_PFT, N_PFT), np.float32)
    for i in range(N_PFT):
        phys = np.zeros(N_PFT, np.float32); phys[i] = 1.0
        pure_z[i] = np.where(has, (phys - mean) / std, phys)
    pure_z_t = torch.from_numpy(pure_z).to(device)          # (15, 15)

    denorm = lai_normalized and norm_stats is not None and "LAI" in norm_stats
    lm = float(norm_stats["LAI"]["mean"]) if denorm else 0.0
    ls = float(norm_stats["LAI"]["std"]) if denorm else 1.0

    rows = []
    idx0 = 0
    with torch.no_grad():
        for feats, tgts in loader:
            x = feats.to(device)                            # (B, 28, L)
            B, _, L = x.shape
            direct = model(x)[:, 0, :]                      # (B, 36)

            # Real fractions p (denorm from last timestep), clamp, renormalize.
            z_last = x[:, PFT_START:PFT_START + N_PFT, -1]  # (B, 15)
            p = z_last * torch.tensor(std, device=device) + torch.tensor(mean, device=device)
            p = p.clamp(min=0.0)
            p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-6)

            mix = torch.zeros_like(direct)
            for i in range(N_PFT):
                xi = x.clone()
                xi[:, PFT_START:PFT_START + N_PFT, :] = pure_z_t[i].view(1, N_PFT, 1)
                pred_i = model(xi)[:, 0, :]                 # (B, 36)
                mix += p[:, i:i + 1] * pred_i

            pftmix = None
            if model_mix is not None:
                pftmix = (model_mix(x)[:, 0, :] * ls2 + lm2).cpu().numpy()

            direct = (direct * ls + lm).cpu().numpy()
            mix = (mix * ls + lm).cpu().numpy()
            obs = (tgts[:, 0, :].numpy() * ls + lm) if denorm else tgts[:, 0, :].numpy()
            pnp = p.cpu().numpy()
            for j in range(B):
                info = dataset.get_site_info(idx0 + j)
                dom = int(np.argmax(pnp[j])) + 1
                for k in range(36):
                    r = {"site_id": info["site_id"], "year": info["year"],
                         "doy": _OBS_DOY[k], "dom_pft": dom,
                         "dom_frac": float(pnp[j].max()),
                         "lai_obs": float(obs[j, k]),
                         "lai_direct": float(direct[j, k]),
                         "lai_mix": float(mix[j, k])}
                    if pftmix is not None:
                        r["lai_pftmix"] = float(pftmix[j, k])
                    rows.append(r)
            idx0 += B

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    csv = os.path.join(args.output_dir, "mixing_compare.csv")
    df.to_csv(csv, index=False)
    print(f"\nCSV → {csv}")
    _summary(df, args.output_dir, args.purity)


def _r2_rmse(a, b):
    """R² of b vs reference a (NSE) and RMSE, on finite pairs."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    sstot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum((a - b) ** 2)) / sstot if sstot > 0 else float("nan")
    return r2, rmse


def _summary(df, out_dir, purity):
    o = df["lai_obs"].values; d = df["lai_direct"].values; mx = df["lai_mix"].values
    lines = ["── PFT-linearity: direct vs linear-mix reconstruction ──"]
    r2, rmse = _r2_rmse(d, mx)
    lines.append(f"  Agreement direct↔mix : R²={r2:+.4f}  RMSE={rmse:.4f}  "
                 f"(1 → model is linear in PFT fractions)")
    r2, rmse = _r2_rmse(o, d); lines.append(f"  direct vs obs        : R²={r2:+.4f}  RMSE={rmse:.4f}")
    r2, rmse = _r2_rmse(o, mx); lines.append(f"  mix    vs obs        : R²={r2:+.4f}  RMSE={rmse:.4f}")

    if "lai_pftmix" in df.columns:
        pm = df["lai_pftmix"].values
        lines.append("")
        lines.append("── PFT-mixing model (same cells) ──")
        r2, rmse = _r2_rmse(o, pm)
        lines.append(f"  pftmix vs obs        : R²={r2:+.4f}  RMSE={rmse:.4f}")
        r2, rmse = _r2_rmse(pm, mx)
        lines.append(f"  pftmix ↔ mix         : R²={r2:+.4f}  RMSE={rmse:.4f}  "
                     f"(does the mixing model match the linear reconstruction?)")
        r2, rmse = _r2_rmse(pm, d)
        lines.append(f"  pftmix ↔ direct      : R²={r2:+.4f}  RMSE={rmse:.4f}")

    pure = df[df["dom_frac"] >= purity]
    lines.append("")
    lines.append(f"── Per-PFT skill on ~pure cells (dom_frac ≥ {purity}) "
                 f"[{pure['site_id'].nunique() if not pure.empty else 0} sites] ──")
    if pure.empty:
        lines.append("  (no cell above the purity threshold)")
    else:
        for pft, g in pure.groupby("dom_pft"):
            r2, rmse = _r2_rmse(g["lai_obs"].values, g["lai_direct"].values)
            extra = ""
            if "lai_pftmix" in g.columns:
                r2m, rmsem = _r2_rmse(g["lai_obs"].values, g["lai_pftmix"].values)
                extra = f"  | pftmix R²={r2m:+.4f} RMSE={rmsem:.4f}"
            lines.append(f"  PFT {pft:2d}: direct R²={r2:+.4f} RMSE={rmse:.4f}{extra}  "
                         f"(n={g['site_id'].nunique()} sites)")

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Summary → {path}")

    _plot(df, out_dir)


def _plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(df["lai_direct"], df["lai_mix"], s=2, alpha=0.2)
    lo = float(np.nanmin(df[["lai_direct", "lai_mix"]].values))
    hi = float(np.nanmax(df[["lai_direct", "lai_mix"]].values))
    ax[0].plot([lo, hi], [lo, hi], "r-", lw=1)
    ax[0].set_xlabel("direct LAI"); ax[0].set_ylabel("Σ pᵢ·pure_i LAI")
    ax[0].set_title("direct vs linear-mix")
    curves = [("lai_obs", "obs"), ("lai_direct", "direct"), ("lai_mix", "mix")]
    if "lai_pftmix" in df.columns:
        curves.append(("lai_pftmix", "pftmix model"))
    for col, lab in curves:
        ax[1].plot(df.groupby("doy")[col].mean().sort_index(), marker="o", ms=3, label=lab)
    ax[1].set_xlabel("day of year"); ax[1].set_ylabel("mean LAI")
    ax[1].set_title("mean seasonal curves"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mixing_compare.png"), dpi=130)
    plt.close(fig)
    print(f"Plot → {out_dir}/mixing_compare.png")


if __name__ == "__main__":
    main()
