#!/usr/bin/env python3
"""
aggregate.py — collect the feature-ablation runs into a ΔR² table + bar plot.

Reads runs/<experiment>/checkpoints/best_model.pth for the baseline and every
ablation, and reports how much validation R² each feature was worth:

    delta_r2  = R2(ablated) - R2(baseline)      # < 0  → the feature was helping
    importance = -delta_r2                        # R² lost when the feature is removed

Outputs a printed table, a CSV, and a horizontal bar plot sorted by importance.

Usage
-----
    python -m phenonn.analysis.feature_ablation.aggregate \\
        --runs_dir runs \\
        --output study/feature_ablation/feature_ablation.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from phenonn.analysis.feature_ablation._common import NON_PFT_FEATURES, BASELINE_EXP, exp_name


def _read_metrics(runs_dir: str, experiment: str):
    """Best-epoch val metrics from a run's best_model.pth, or None if missing."""
    ckpt = Path(runs_dir) / experiment / "checkpoints" / "best_model.pth"
    if not ckpt.exists():
        return None
    snap = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {
        "val_r2":     snap.get("val_r2"),
        "val_rmse":   snap.get("val_rmse"),
        "val_loss":   snap.get("val_loss"),
        "best_epoch": snap.get("best_epoch"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", default="runs",
                    help="Folder holding <experiment>/checkpoints/best_model.pth.")
    ap.add_argument("--output", default="study/feature_ablation/feature_ablation.png",
                    help="Bar-plot path; the CSV is written alongside it.")
    args = ap.parse_args()

    base = _read_metrics(args.runs_dir, BASELINE_EXP)
    if base is None or base["val_r2"] is None:
        raise SystemExit(
            f"Baseline run {BASELINE_EXP!r} not found under {args.runs_dir}/ "
            f"(expected {BASELINE_EXP}/checkpoints/best_model.pth). Run it first."
        )
    r2_base = float(base["val_r2"])
    rmse_base = float(base["val_rmse"])

    rows = []
    for feat in NON_PFT_FEATURES:
        m = _read_metrics(args.runs_dir, exp_name(feat))
        if m is None or m["val_r2"] is None:
            print(f"  [skip] {feat}: run missing")
            continue
        r2 = float(m["val_r2"])
        rmse = float(m["val_rmse"])
        rows.append({
            "feature":    feat,
            "val_r2":     r2,
            "delta_r2":   r2 - r2_base,
            "val_rmse":   rmse,
            "delta_rmse": rmse - rmse_base,
        })

    if not rows:
        raise SystemExit("No ablation runs found. Submit the jobs first.")

    # Most important feature = biggest R² drop when removed = most negative ΔR².
    rows.sort(key=lambda r: r["delta_r2"])

    print(f"\nBaseline ({BASELINE_EXP}): R2={r2_base:.4f}  RMSE={rmse_base:.4f}\n")
    print(f"{'feature':<14}{'val_R2':>9}{'dR2':>10}{'dRMSE':>10}")
    print("-" * 43)
    for r in rows:
        print(f"{r['feature']:<14}{r['val_r2']:>9.4f}"
              f"{r['delta_r2']:>10.4f}{r['delta_rmse']:>10.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("feature,val_r2,delta_r2,val_rmse,delta_rmse\n")
        f.write(f"{BASELINE_EXP},{r2_base},0.0,{rmse_base},0.0\n")
        for r in rows:
            f.write(f"{r['feature']},{r['val_r2']},{r['delta_r2']},"
                    f"{r['val_rmse']},{r['delta_rmse']}\n")

    feats = [r["feature"] for r in rows]
    importance = [-r["delta_r2"] for r in rows]      # positive = feature matters
    plt.figure(figsize=(8, 0.45 * len(feats) + 1.6))
    colors = ["#c0392b" if v >= 0 else "#7f8c8d" for v in importance]
    plt.barh(range(len(feats)), importance, color=colors)
    plt.yticks(range(len(feats)), feats)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("R2 lost when feature is removed  (-dR2)")
    plt.title(f"Feature ablation vs baseline (R2={r2_base:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"\n[ok] table -> {csv_path}")
    print(f"[ok] plot  -> {out_path}")


if __name__ == "__main__":
    main()
