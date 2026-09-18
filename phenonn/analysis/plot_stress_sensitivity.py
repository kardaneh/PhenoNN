#!/usr/bin/env python3
"""
plot_stress_sensitivity.py
==========================

From a sensitivity matrix produced by `prediction.stress_sensitivity`
(`<prefix>.npy`, shape (L, 365) or (L, 365, 15) with --per_pft), plot for each
output day j of the last year two magnitudes on the same axes:

  • instantané : |M[i_same, j]|            — impact of the SAME day's stress
  • passé      : Σ_{i < i_same} |M[i, j]|  — summed |impact| of all earlier days

where i_same = (L - 365) + j is the input position of the same calendar day as
output day j. (Input days i > i_same are the causal "future" region and should
be ≈ 0; drawn as a faint dashed line as a sanity check.)

This answers "for the prediction at day j, how much comes from today's stress
vs. the accumulated stress of the preceding days".

Usage
-----
    python -m prediction.plot_stress_sensitivity \\
        sensitivity_pix_0453_01987.npy \\
        --output sensitivity_pix_0453_01987_inst_vs_past.png
"""

import argparse
import os

import numpy as np

from phenonn.utils.config import PFT_NAMES


def _decompose(M2d: np.ndarray):
    """M2d: (L, out_len) → (inst, past, future) each (out_len,)."""
    L, out_len = M2d.shape
    offset = L - out_len
    j = np.arange(out_len)
    i_same = offset + j                     # input index of the same day as j
    A = np.abs(M2d)

    inst = A[i_same, j]

    cum = np.cumsum(A, axis=0)              # cum[i, j] = Σ_{k<=i} A[k, j]
    total = cum[-1, j]
    prev = np.where(i_same >= 1, cum[np.clip(i_same - 1, 0, L - 1), j], 0.0)
    past = prev                            # Σ_{i < i_same}
    future = total - cum[i_same, j]        # Σ_{i > i_same}
    return inst, past, future


def _plot_pair(ax, inst, past, future, title):
    x = np.arange(len(inst)) + 1           # day-of-year 1..365
    ax.plot(x, past, color="tab:blue", lw=1.6,
            label="passé  Σ|M[i<j, j]|")
    ax.plot(x, inst, color="tab:red", lw=1.6,
            label="instantané  |M[j, j]|")
    if np.nanmax(np.abs(future)) > 0:
        ax.plot(x, future, color="0.6", lw=0.8, ls="--",
                label="fuite future  Σ|M[i>j, j]|")
    ax.set_xlabel("Jour de l'année (dernière année)")
    ax.set_ylabel("|ΔLAI| par unité de perturbation")
    ax.set_xlim(1, len(inst))
    ax.margins(y=0.05)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matrix", help="Path to <prefix>.npy from stress_sensitivity.")
    p.add_argument("--output", default="",
                   help="Output PNG (default: <matrix>_inst_vs_past.png).")
    p.add_argument("--log", action="store_true",
                   help="Log-scale the y-axis (past can dwarf instantané).")
    p.add_argument("--title", default="",
                   help="Extra title text (e.g. the site/year).")
    args = p.parse_args()

    M = np.load(args.matrix)
    if M.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D matrix, got shape {M.shape}.")

    out = args.output or (os.path.splitext(args.matrix)[0] + "_inst_vs_past.png")
    stem = os.path.basename(os.path.splitext(args.matrix)[0])
    suffix = f" — {args.title}" if args.title else f" — {stem}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if M.ndim == 2:
        inst, past, future = _decompose(M)
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_pair(ax, inst, past, future,
                   f"Sensibilité instantanée vs passée{suffix}")
        if args.log:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"instantané: max={np.nanmax(inst):.3e}  "
              f"passé: max={np.nanmax(past):.3e}  "
              f"ratio médian passé/instantané="
              f"{np.nanmedian(past / np.where(inst > 0, inst, np.nan)):.2f}")
    else:
        n = M.shape[2]
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(5 * ncols, 3.2 * nrows))
        for k in range(nrows * ncols):
            ax = axes[k // ncols][k % ncols]
            if k >= n:
                ax.axis("off")
                continue
            inst, past, future = _decompose(M[:, :, k])
            name = PFT_NAMES[k] if k < len(PFT_NAMES) else ""
            _plot_pair(ax, inst, past, future, f"PFT{k + 1} {name}")
            if args.log:
                ax.set_yscale("log")
        fig.suptitle(f"Sensibilité instantanée vs passée par PFT{suffix}")
        fig.tight_layout()
        fig.savefig(out, dpi=130)

    print(f"Figure → {out}")


if __name__ == "__main__":
    main()
