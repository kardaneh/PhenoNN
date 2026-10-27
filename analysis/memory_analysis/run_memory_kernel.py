#!/usr/bin/env python3
"""Memory kernel of a trained PhenoNN model.

The kernel measures how far back in time the model's seasonal prediction
depends on a channel.  For each lag ``k`` (days before the last day of the
input window) a small impulse is added to one channel on a single day and the
mean absolute change in the 36-dekad LAI prediction is recorded.

The script also runs the controls required by ``PROTOCOL.md`` (F13):

* ``noop``       — a zero-amplitude impulse must produce exactly zero response.
* ``random_day`` — impulses on random days give the null distribution.
* ``linearity``  — doubling the amplitude must double the response (within
  tolerance) if the model is locally linear.

This is an inference-only experiment; no training is performed.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from analysis.perturbation import run_perturbation as R
from analysis.perturbation._perturb import ChannelPerturbation, Perturbation

SCENARIO = "memory_kernel"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    R.add_common_args(p)
    p.add_argument("--channel", default="Tmean", help="Channel to perturb.")
    p.add_argument("--delta", type=float, default=2.0, help="Impulse amplitude (physical units).")
    p.add_argument(
        "--lags",
        default="0,5,10,15,30,45,60,90,120,180,270,365",
        help="Comma-separated lags in days (0 = last day of the window).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=1,
        help="Impulse width in days (1 = single day).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=512,
        help="Number of (site, year) samples used for the kernel.",
    )
    p.add_argument(
        "--control_channel",
        default="strd_sum",
        help="Channel used for the random-day control.",
    )
    p.add_argument(
        "--n_random_controls",
        type=int,
        default=50,
        help="Number of random-day control impulses.",
    )
    p.add_argument(
        "--linearity_amplitude",
        type=float,
        default=None,
        help="Second amplitude for the linearity check (default 2× --delta).",
    )
    p.add_argument(
        "--linearity_tolerance",
        type=float,
        default=0.25,
        help="Allowed relative deviation from exact linear scaling.",
    )
    return p.parse_args()


def _parse_lags(spec: str) -> list:
    return [int(x) for x in str(spec).split(",") if str(x).strip()]


def _impulse_mask(seq_length: int, lag: int, width: int = 1) -> np.ndarray:
    """Boolean mask with a window ending ``lag`` days before the last day."""
    mask = np.zeros(seq_length, dtype=bool)
    end = seq_length - 1 - int(lag)
    start = end - max(1, int(width)) + 1
    if end < 0:
        return mask
    mask[max(0, start) : end + 1] = True
    return mask


def _subset_loader(ctx, max_samples: int, seed: int):
    n = len(ctx.dataset)
    if max_samples <= 0 or max_samples >= n:
        indices = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n, size=max_samples, replace=False))
    loader = DataLoader(
        Subset(ctx.dataset, indices.tolist()),
        batch_size=min(64, len(indices)),
        shuffle=False,
        num_workers=0,
    )
    lats = (
        None
        if ctx.lat_by_sample is None
        else np.asarray(ctx.lat_by_sample)[indices]
    )
    return loader, indices, lats


def _response(base: np.ndarray, pert: np.ndarray) -> np.ndarray:
    """Per-sample mean and mean-absolute response over the 36 dekads."""
    d = pert - base
    return np.nanmean(d, axis=1), np.nanmean(np.abs(d), axis=1)


def _kernel_for_mask(ctx, loader, lats, channel, delta, mask, site_ids) -> np.ndarray:
    pert = Perturbation(
        [ChannelPerturbation(channel, add=delta)],
        ctx.norm_stats,
        position_mask=mask,
    )
    raw = R._infer_perturbed(
        ctx.model, loader, pert, lats, ctx.device, ablate_ch=ctx.ablate_ch
    )
    if ctx.is_daily:
        pos = np.asarray(R._OBS_POSITIONS, dtype=np.int64)
        raw = raw[:, pos]
    return R._to_physical(raw, ctx, site_ids)


def main():
    args = parse_args()
    ctx = R.load_model_and_data(args)
    loader, indices, lats = _subset_loader(ctx, args.max_samples, args.seed)
    n = len(indices)
    print(f"Kernel samples : {n:,} of {len(ctx.dataset):,}")
    print(f"Channel        : {args.channel}  Δ={args.delta:g}  width={args.window}")

    # Baseline on the same subset.
    base_raw = R._infer(ctx.model, loader, ctx.device)
    if ctx.is_daily:
        pos = np.asarray(R._OBS_POSITIONS, dtype=np.int64)
        base_raw = base_raw[:, pos]
    site_ids = [ctx.dataset.get_site_info(int(i))["site_id"] for i in indices]
    base = R._to_physical(base_raw, ctx, site_ids)

    seq_length = int(ctx.dataset[0][0].shape[-1])
    lags = [lag for lag in _parse_lags(args.lags) if 0 <= lag < seq_length]
    print(f"Window length  : {seq_length} days; {len(lags)} lags")

    rows = []
    for lag in lags:
        mask = _impulse_mask(seq_length, lag, args.window)
        if not mask.any():
            continue
        pert = _kernel_for_mask(
            ctx, loader, lats, args.channel, args.delta, mask, site_ids
        )
        mean_d, abs_d = _response(base, pert)
        abs_d = abs_d[np.isfinite(abs_d)]
        lo, hi = (
            np.percentile(abs_d, [2.5, 97.5]) if abs_d.size else (np.nan, np.nan)
        )
        rows.append(
            {
                "lag_days": int(lag),
                "channel": args.channel,
                "amplitude": float(args.delta),
                "width_days": int(args.window),
                "n_samples": int(abs_d.size),
                "mean_delta": float(np.nanmean(mean_d)),
                "mean_abs_delta": float(np.nanmean(abs_d)),
                "std_abs_delta": float(np.nanstd(abs_d)),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
        print(
            f"  lag {lag:4d} d : mean|ΔLAI| = {np.nanmean(abs_d):.5f} "
            f"[{lo:.5f}, {hi:.5f}]"
        )

    out_dir = os.path.join(args.output_dir, SCENARIO)
    os.makedirs(out_dir, exist_ok=True)
    kernel = pd.DataFrame(rows)
    kernel.to_csv(os.path.join(out_dir, "memory_kernel.csv"), index=False)
    print(f"\nKernel → {out_dir}/memory_kernel.csv")

    # ── Controls ──
    controls = []
    noop = _kernel_for_mask(
        ctx, loader, lats, args.channel, 0.0, _impulse_mask(seq_length, 0), site_ids
    )
    controls.append(
        {
            "control": "noop",
            "value": float(np.nanmax(np.abs(noop - base))),
            "description": "zero-amplitude impulse (should be ~0)",
        }
    )

    rng = np.random.default_rng(args.seed)
    null = []
    for _ in range(args.n_random_controls):
        lag = int(rng.integers(0, seq_length))
        mask = _impulse_mask(seq_length, lag, args.window)
        pert = _kernel_for_mask(
            ctx, loader, lats, args.control_channel, args.delta, mask, site_ids
        )
        _, abs_d = _response(base, pert)
        null.append(float(np.nanmean(abs_d)))
    null = np.asarray(null)
    controls.append(
        {
            "control": "random_day",
            "value": float(np.nanmean(null)),
            "description": (
                f"mean |ΔLAI| over {len(null)} random-day impulses on "
                f"{args.control_channel}"
            ),
        }
    )
    controls.append(
        {
            "control": "random_day_p95",
            "value": float(np.nanpercentile(null, 95)),
            "description": "95th percentile of the random-day null",
        }
    )

    amp2 = (
        args.linearity_amplitude
        if args.linearity_amplitude is not None
        else 2.0 * args.delta
    )
    pert2 = _kernel_for_mask(
        ctx, loader, lats, args.channel, amp2, _impulse_mask(seq_length, 0), site_ids
    )
    _, abs2 = _response(base, pert2)
    if kernel.size:
        ref = float(kernel.iloc[0]["mean_abs_delta"])
        ratio = float(np.nanmean(abs2) / max(ref, 1e-12))
        deviation = abs(ratio - amp2 / max(args.delta, 1e-12))
        controls.append(
            {
                "control": "linearity",
                "value": ratio,
                "description": (
                    f"response ratio at {amp2:g} vs {args.delta:g}; "
                    f"expected {amp2 / max(args.delta, 1e-12):.2f}, "
                    f"deviation {deviation:.2f} (tol {args.linearity_tolerance})"
                ),
            }
        )
        print(
            f"Linearity      : ratio {ratio:.3f} "
            f"(expected {amp2 / max(args.delta, 1e-12):.2f})"
        )

    pd.DataFrame(controls).to_csv(
        os.path.join(out_dir, "memory_controls.csv"), index=False
    )
    print(f"Controls → {out_dir}/memory_controls.csv")

    _plot(kernel, out_dir)


def _plot(kernel: pd.DataFrame, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if kernel.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        kernel["lag_days"],
        kernel["mean_abs_delta"],
        marker="o",
        color="#1f4e79",
        lw=2,
        label="mean |ΔLAI|",
    )
    ax.fill_between(
        kernel["lag_days"],
        kernel["ci_low"],
        kernel["ci_high"],
        color="#1f4e79",
        alpha=0.2,
        lw=0,
    )
    ax.set_xlabel("Lag before the last day (days)")
    ax.set_ylabel("Mean |ΔLAI|")
    ax.set_title("Memory kernel")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_kernel.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "memory_kernel.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
