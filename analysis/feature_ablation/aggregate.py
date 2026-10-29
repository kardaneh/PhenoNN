#!/usr/bin/env python3
"""
aggregate.py — collect feature-ablation runs into ΔR² tables and bar plots.

Supports the three families produced by this study:

    features  abl_baseline + abl_<feature>
    groups    ablate_<model>_baseline + ablate_<model>_<group>
    combo     ablate_combo_<config>_baseline + ablate_combo_<config>

Each checkpoint's stored ``args.ablate_features`` is validated against the mask
expected for that run, so a renamed or copied run cannot silently contaminate
the table. Mismatches are skipped unless ``--allow-mask-mismatch`` is given;
missing masks (legacy checkpoints) are included unless ``--strict-mask``.

Usage
-----
    python -m analysis.feature_ablation.aggregate \\
        --kind features --runs_dir runs/feature_ablation

    python -m analysis.feature_ablation.aggregate \\
        --kind groups --model aelstm --runs_dir runs/feature_ablation \\
        --output runs/feature_ablation/feature_ablation_groups.png

    python -m analysis.feature_ablation.aggregate \\
        --kind combo --runs_dir runs/feature_ablation
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analysis._checkpoint_utils import find_best_checkpoint
from analysis.feature_ablation._common import (
    BASELINE_EXP,
    COMBO_CONFIGS,
    COMBO_FEATURES,
    GROUPS,
    NON_PFT_FEATURES,
    combo_baseline_exp_name,
    combo_exp_name,
    exp_name,
    group_exp_name,
)

KINDS = ("features", "groups", "combo")


def normalize_mask(value):
    """Return a sorted tuple of feature names from a checkpoint mask value."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    return tuple(sorted(p.strip() for p in parts if str(p).strip()))


def masks_match(actual, expected):
    """Compare two masks ignoring order; ``None`` means "unknown"."""
    if actual is None or expected is None:
        return actual is expected
    return normalize_mask(actual) == normalize_mask(expected)


def combo_configs():
    """Stable ``[(label, model, hidden_size, stress_dim), ...]``."""
    out = []
    for model, hidden, stress in COMBO_CONFIGS:
        label = f"{model}_h{hidden}" + (f"_s{stress}" if stress else "")
        out.append((label, model, hidden, stress))
    return out


def _read_run(runs_dir, experiment):
    """Read a run's best checkpoint metrics and stored mask, or ``None``."""
    ckpt = find_best_checkpoint(Path(runs_dir) / experiment)
    if ckpt is None:
        return None
    try:
        snap = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  [skip] {experiment}: unreadable checkpoint ({exc})")
        return None
    if not isinstance(snap, dict):
        print(f"  [skip] {experiment}: checkpoint is not a mapping")
        return None
    args = snap.get("args", {}) or {}
    if not isinstance(args, dict):
        print(f"  [skip] {experiment}: checkpoint args are not a mapping")
        return None
    has_mask = "ablate_features" in args
    return {
        "experiment": experiment,
        "checkpoint": str(ckpt),
        "val_r2": snap.get("val_r2"),
        "val_rmse": snap.get("val_rmse"),
        "val_loss": snap.get("val_loss"),
        "best_epoch": snap.get("best_epoch"),
        "mask": normalize_mask(args.get("ablate_features")) if has_mask else None,
        "has_mask": has_mask,
    }


def _validate(run, expected, strict_mask, allow_mismatch):
    """Return True when the run should be used."""
    try:
        val_r2 = float(run["val_r2"])
    except (TypeError, ValueError):
        val_r2 = None
    if val_r2 is None or not math.isfinite(val_r2):
        print(f"  [skip] {run['experiment']}: no finite val_r2 in checkpoint")
        return False
    if not run["has_mask"]:
        if strict_mask:
            print(f"  [skip] {run['experiment']}: checkpoint has no ablation mask")
            return False
        print(
            f"  [warn] {run['experiment']}: checkpoint has no ablation mask "
            "(legacy run); including it"
        )
        return True
    if masks_match(run["mask"], expected):
        return True
    msg = (
        f"{run['experiment']}: stored mask={list(run['mask'] or ())} "
        f"expected={list(normalize_mask(expected) or ())}"
    )
    if allow_mismatch:
        print(f"  [warn] mask mismatch, including: {msg}")
        return True
    print(f"  [skip] mask mismatch: {msg}")
    return False


def _metrics(run):
    return (
        float(run["val_r2"]),
        float(run["val_rmse"]) if run["val_rmse"] is not None else float("nan"),
    )


def _row(label, run, r2_base, rmse_base):
    r2, rmse = _metrics(run)
    return {
        "label": label,
        "group": label,
        "experiment": run["experiment"],
        "ablate_features": ",".join(run["mask"] or ()),
        "val_r2": r2,
        "delta_r2": r2 - r2_base,
        "val_rmse": rmse,
        "delta_rmse": rmse - rmse_base,
        "best_epoch": run["best_epoch"],
    }


def _baseline(runs_dir, experiment, expected, strict_mask, allow_mismatch):
    run = _read_run(runs_dir, experiment)
    if run is None:
        return None
    if not _validate(run, expected, strict_mask, allow_mismatch):
        return None
    return run


def _feature_inputs(features):
    if features:
        chosen = [f.strip() for f in features.split(",") if f.strip()]
    else:
        chosen = list(NON_PFT_FEATURES)
    return [(feat, exp_name(feat), (feat,)) for feat in chosen]


def _group_inputs(model, groups):
    if groups:
        wanted = {g.strip() for g in groups.split(",") if g.strip()}
    else:
        wanted = {name for name, _ in GROUPS if name != "baseline"}
    out = []
    for name, feats in GROUPS:
        if name == "baseline" or name not in wanted:
            continue
        out.append((name, group_exp_name(model, name), tuple(feats)))
    return out


def _combo_inputs():
    out = []
    for label, model, hidden, stress in combo_configs():
        out.append(
            (
                label,
                combo_exp_name(model, hidden, stress),
                combo_baseline_exp_name(model, hidden, stress),
                tuple(COMBO_FEATURES),
            )
        )
    return out


def _collect_features(runs_dir, baseline_exp, inputs, strict_mask, allow_mismatch):
    base = _baseline(runs_dir, baseline_exp, (), strict_mask, allow_mismatch)
    if base is None:
        raise SystemExit(
            f"Baseline run {baseline_exp!r} not found (or failed validation) "
            f"under {runs_dir}/. Run it first."
        )
    r2_base, rmse_base = _metrics(base)
    rows = [
        {
            "label": "baseline",
            "experiment": baseline_exp,
            "ablate_features": "",
            "val_r2": r2_base,
            "delta_r2": 0.0,
            "val_rmse": rmse_base,
            "delta_rmse": 0.0,
            "best_epoch": base["best_epoch"],
        }
    ]
    for label, experiment, expected in inputs:
        run = _read_run(runs_dir, experiment)
        if run is None:
            print(f"  [skip] {label}: run missing ({experiment})")
            continue
        if not _validate(run, expected, strict_mask, allow_mismatch):
            continue
        rows.append(_row(label, run, r2_base, rmse_base))
    return rows, baseline_exp


def _collect_combo(runs_dir, inputs, strict_mask, allow_mismatch):
    rows = []
    baselines = {}
    for label, experiment, baseline_exp, expected in inputs:
        base = _baseline(runs_dir, baseline_exp, (), strict_mask, allow_mismatch)
        if base is None:
            print(
                f"  [skip] {label}: baseline missing ({baseline_exp}); "
                "rerun without --skip-baseline"
            )
            continue
        r2_base, rmse_base = _metrics(base)
        baselines[label] = (baseline_exp, r2_base, rmse_base)
        run = _read_run(runs_dir, experiment)
        if run is None:
            print(f"  [skip] {label}: run missing ({experiment})")
            continue
        if not _validate(run, expected, strict_mask, allow_mismatch):
            continue
        rows.append(_row(label, run, r2_base, rmse_base))
    return rows, baselines


def _print_table(rows, baseline_label, title):
    print(f"\n{title} (baseline: {baseline_label})\n")
    print(f"{'label':<28}{'val_R2':>9}{'dR2':>10}{'dRMSE':>10}")
    print("-" * 57)
    for row in rows:
        print(
            f"{row['label']:<28}{row['val_r2']:>9.4f}"
            f"{row['delta_r2']:>10.4f}{row['delta_rmse']:>10.4f}"
        )


def _write_outputs(rows, out_path, title):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "label",
                "experiment",
                "ablate_features",
                "val_r2",
                "delta_r2",
                "val_rmse",
                "delta_rmse",
                "best_epoch",
            )
        )
        writer.writerows(
            (
                row["label"],
                row["experiment"],
                row["ablate_features"],
                row["val_r2"],
                row["delta_r2"],
                row["val_rmse"],
                row["delta_rmse"],
                row["best_epoch"],
            )
            for row in rows
        )

    labels = [row["label"] for row in rows]
    importance = [-row["delta_r2"] for row in rows]
    plt.figure(figsize=(8, 0.45 * len(labels) + 1.6))
    colors = ["#c0392b" if v >= 0 else "#7f8c8d" for v in importance]
    plt.barh(range(len(labels)), importance, color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("R2 lost when removed  (-dR2)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\n[ok] table -> {csv_path}")
    print(f"[ok] plot  -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kind", default="features", choices=KINDS)
    ap.add_argument(
        "--runs_dir",
        default="runs/feature_ablation",
        help="Folder holding <experiment>/checkpoints/.",
    )
    ap.add_argument(
        "--output",
        default="",
        help="Plot path; the CSV is written alongside it (default: per kind).",
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="Explicit baseline experiment (features/groups only).",
    )
    ap.add_argument(
        "--model",
        default="aelstm",
        help="Model prefix for --kind groups (default: aelstm).",
    )
    ap.add_argument("--features", default="", help="Per-feature subset.")
    ap.add_argument("--groups", default="", help="Group subset.")
    ap.add_argument("--strict-mask", action="store_true")
    ap.add_argument("--allow-mask-mismatch", action="store_true")
    args = ap.parse_args()

    if args.allow_mask_mismatch and args.strict_mask:
        ap.error("--strict-mask and --allow-mask-mismatch are exclusive")

    title = f"Feature ablation — {args.kind}"
    if args.kind == "features":
        baseline_exp = args.baseline or BASELINE_EXP
        inputs = _feature_inputs(args.features)
        rows, baseline_label = _collect_features(
            args.runs_dir,
            baseline_exp,
            inputs,
            args.strict_mask,
            args.allow_mask_mismatch,
        )
        default = Path(args.runs_dir) / "feature_ablation.png"
    elif args.kind == "groups":
        baseline_exp = args.baseline or group_exp_name(args.model, "baseline")
        inputs = _group_inputs(args.model, args.groups)
        rows, baseline_label = _collect_features(
            args.runs_dir,
            baseline_exp,
            inputs,
            args.strict_mask,
            args.allow_mask_mismatch,
        )
        default = Path(args.runs_dir) / "feature_ablation_groups.png"
    else:
        inputs = _combo_inputs()
        rows, baselines = _collect_combo(
            args.runs_dir,
            inputs,
            args.strict_mask,
            args.allow_mask_mismatch,
        )
        baseline_label = ", ".join(sorted({b[0] for b in baselines.values()})) or "—"
        default = Path(args.runs_dir) / "feature_ablation_combo.png"

    if not rows:
        raise SystemExit("No runs found. Generate and submit the jobs first.")

    rows.sort(key=lambda r: r["delta_r2"])
    _print_table(rows, baseline_label, title)
    _write_outputs(rows, Path(args.output) if args.output else default, title)


if __name__ == "__main__":
    main()
