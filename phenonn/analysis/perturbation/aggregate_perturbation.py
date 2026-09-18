#!/usr/bin/env python3
"""
aggregate_perturbation.py — compare the perturbation scenarios of one model.

Reads every `<output_dir>/<scenario>/deltas.csv` produced by the scenario scripts
and writes a single comparison: a table (CSV + text) and a figure overlaying the
four seasonal ΔLAI curves.

    python -m phenonn.analysis.perturbation.aggregate_perturbation \\
        --output_dir runs_perturb/<experiment>
"""

import argparse
import os

import numpy as np
import pandas as pd

SCENARIOS = ["drought", "heatwave", "warming_co2", "winter_frost"]


def _stats(df):
    d = df["delta"].values
    summer = df.loc[df["doy"].between(152, 243), "delta"].values
    base = df["lai_base"].values
    rel = (np.nanmean(d) / np.nanmean(base) * 100.0
           if np.nanmean(base) not in (0.0, np.nan) else np.nan)
    return {
        "mean_dLAI":     float(np.nanmean(d)),
        "median_dLAI":   float(np.nanmedian(d)),
        "std_dLAI":      float(np.nanstd(d)),
        "mean_abs_dLAI": float(np.nanmean(np.abs(d))),
        "min_dLAI":      float(np.nanmin(d)),
        "max_dLAI":      float(np.nanmax(d)),
        "summer_dLAI":   float(np.nanmean(summer)) if summer.size else np.nan,
        "rel_change_%":  float(rel),
        "frac_|d|>0.05_%": float(100.0 * np.mean(np.abs(d) > 0.05)),
        "n_points":      int(len(df)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True,
                   help="Folder holding the <scenario>/ sub-folders.")
    args = p.parse_args()

    rows, curves = {}, {}
    for s in SCENARIOS:
        path = os.path.join(args.output_dir, s, "deltas.csv")
        if not os.path.exists(path):
            print(f"  ✗ {s}: {path} missing — skipped")
            continue
        df = pd.read_csv(path)
        rows[s] = _stats(df)
        curves[s] = df.groupby("doy")["delta"].mean().sort_index()
        print(f"  ✓ {s}: {len(df):,} points")

    if not rows:
        raise SystemExit("No scenario results found.")

    table = pd.DataFrame(rows).T
    csv = os.path.join(args.output_dir, "comparison.csv")
    table.to_csv(csv)

    lines = [f"── Perturbation scenarios — {os.path.basename(args.output_dir)} ──",
             "", table.to_string(float_format=lambda v: f"{v:+.4f}"), "",
             "ΔLAI = LAI(perturbed) − LAI(baseline), same cells, ceteris paribus.",
             "Most impactful scenario (|mean ΔLAI|): "
             f"{max(rows, key=lambda k: abs(rows[k]['mean_dLAI']))}"]
    txt = os.path.join(args.output_dir, "comparison.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0.0, color="grey", lw=0.8)
    for s, c in curves.items():
        ax.plot(c.index, c.values, marker="o", ms=3, label=s)
    ax.set_xlabel("day of year")
    ax.set_ylabel("mean ΔLAI (perturbed − baseline)")
    ax.set_title(f"Perturbation scenarios — {os.path.basename(args.output_dir)}")
    ax.legend()
    fig.tight_layout()
    png = os.path.join(args.output_dir, "comparison.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    print(f"\nTable → {csv}\nText  → {txt}\nPlot  → {png}")


if __name__ == "__main__":
    main()
