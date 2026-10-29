#!/usr/bin/env python3
"""Generate every paper figure and table from run bundles.

Usage
-----
    python -m analysis.figures.make_all --runs_dir runs/protocol_B --out_dir paper/figures

Each figure is skipped with a message when its input artefact is absent, so the
command is safe to run while some experiments are still pending.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from analysis.figures import figures as F
from analysis.figures import tables as T
from analysis.figures.registry import Registry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_dir", default="runs", help="Tree to scan for metrics.json.")
    p.add_argument("--out_dir", default="runs/figures")
    p.add_argument("--run", default="", help="Run name or substring for F1–F4.")
    p.add_argument("--figures", default="all", help="'all' or comma list F1,F2,...")
    p.add_argument(
        "--ablation_csv", default="", help="CSV with group, delta_r2, ci_low, ci_high."
    )
    p.add_argument(
        "--perturbation_dir", default="", help="Parent of <scenario>/deltas.csv."
    )
    p.add_argument(
        "--warming_csv", default="", help="CSV for F10 (delta_t, co2_mode, ...)."
    )
    p.add_argument("--pft_csv", default="", help="Long CSV from pft_curves.py.")
    p.add_argument("--memory_dir", default="", help="Directory with memory_kernel.csv.")
    p.add_argument(
        "--choices",
        default="",
        help="Optional JSON with deterministic F2/F11 site-year choices.",
    )
    p.add_argument(
        "--loss_runs",
        default="",
        help="Comma-separated run names for F12; default: one run per architecture.",
    )
    return p.parse_args()


def _load_choices(path: str) -> dict:
    """Load optional ``figure_choices.json``; return an empty mapping if absent."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _wanted(spec: str) -> set:
    if spec.strip().lower() == "all":
        return {f"F{i}" for i in range(1, 14)}
    return {s.strip().upper() for s in spec.split(",") if s.strip()}


def _skip(name, why):
    print(f"[skip] {name}: {why}")


def _loss_histories(registry: Registry, loss_runs: str):
    """Return ``({architecture: [history]}, {architecture: stop_epoch})``.

    By default one run per paper architecture is selected from the registry.
    ``--loss_runs`` overrides that with explicit run names.
    """
    if loss_runs:
        runs = [
            registry.get(name.strip()) for name in loss_runs.split(",") if name.strip()
        ]
    else:
        wanted = ("lstm", "bitransformer", "aelstm")
        runs = []
        for label in wanted:
            for run in registry:
                if run.model_label == label and run.has("history.csv"):
                    runs.append(run)
                    break

    groups = {}
    stops = {}
    for run in runs:
        if not run.has("history.csv"):
            print(f"[warn] F12: {run.name} has no history.csv; skipped")
            continue
        groups.setdefault(run.model_label, []).append(run.history())
        stop = run.metadata.get("early_stop_epoch")
        if stop is not None:
            stops[run.model_label] = int(stop)
    return groups, stops


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = _wanted(args.figures)
    registry = Registry.from_dir(args.runs_dir)
    print(f"Registry: {len(registry)} run(s) under {args.runs_dir}")
    if len(registry):
        run = registry.get(args.run) if args.run else registry.runs[0]
        print(f"Primary run: {run.name} ({run.directory})")
    else:
        run = None
        print("No metrics.json found — only optional inputs can be processed.")

    def out(name):
        return os.path.join(args.out_dir, name)

    choices = _load_choices(args.choices)
    pft_df = (
        pd.read_csv(args.pft_csv)
        if args.pft_csv and os.path.exists(args.pft_csv)
        else None
    )

    # ── F1–F4: prediction-level figures ──
    if run is None:
        for name in ("F1", "F2", "F3", "F4"):
            if name in wanted:
                _skip(name, "no metrics.json in --runs_dir")
    else:
        if wanted & {"F1", "F2", "F3", "F4"}:
            sites = run.prediction_sites
            if sites is None:
                print(
                    "[warn] F1–F4: no predict_metadata.json; "
                    "the prediction split is unknown"
                )
            elif sites in ("train", "all", "grid"):
                print(
                    f"[warn] F1–F4: predictions come from "
                    f"--predict_sites={sites}, not a held-out split"
                )
    if run is not None and wanted & {"F1", "F2", "F3"}:
        try:
            pred = run.predictions()
        except FileNotFoundError:
            _skip("F1–F3", "predictions.csv missing")
            pred = None
        if pred is not None:
            if "F1" in wanted:
                F.plot_paper_pred_vs_obs(pred, run.metrics, out("F1_pred_vs_obs"))
            if "F2" in wanted:
                F.plot_paper_site_year_curves(
                    pred,
                    out("F2_site_years"),
                    choices=choices.get("F2"),
                    pft_df=pft_df,
                )
            if "F3" in wanted and {"obs_anom", "pred_anom"}.issubset(pred.columns):
                F.plot_paper_anomaly_scatter(pred, run.metrics, out("F3_anomaly"))
            elif "F3" in wanted:
                _skip("F3", "obs_anom/pred_anom missing")

    if run is not None and "F4" in wanted:
        path = run.file("rmse_by_doy.csv")
        if os.path.exists(path):
            F.plot_paper_rmse_by_doy(pd.read_csv(path), out("F4_rmse_by_doy"))
        else:
            _skip("F4", "rmse_by_doy.csv missing")

    # ── F5–F7: tables ──
    if run is None:
        for name in ("F5", "F6", "F7"):
            if name in wanted:
                _skip(name, "no metrics.json in --runs_dir")
    else:
        if "F5" in wanted:
            T.architecture_table(registry, out("F5_architecture.tex"))
        if "F6" in wanted:
            T.model_vs_climatology_table(registry, out("F6_climatology.tex"))
        if "F7" in wanted:
            T.anomaly_table(registry, out("F7_anomaly.tex"))

    # ── F8: ablation ──
    if "F8" in wanted:
        if args.ablation_csv and os.path.exists(args.ablation_csv):
            df = pd.read_csv(args.ablation_csv)
            F.plot_paper_ablation(df, out("F8_ablation"))
            T.ablation_table(df, out("F8_ablation.tex"))
        else:
            _skip("F8", "provide --ablation_csv")

    # ── F9: perturbation ──
    if "F9" in wanted:
        if args.perturbation_dir and os.path.isdir(args.perturbation_dir):
            frames = {}
            for name in sorted(os.listdir(args.perturbation_dir)):
                path = os.path.join(args.perturbation_dir, name, "deltas.csv")
                if os.path.exists(path):
                    frames[name] = pd.read_csv(path)
            if frames:
                F.plot_paper_perturbation_curves(frames, out("F9_perturbation"))
            else:
                _skip("F9", "no <scenario>/deltas.csv")
        else:
            _skip("F9", "provide --perturbation_dir")

    # ── F10: warming + CO2 ──
    if "F10" in wanted:
        if args.warming_csv and os.path.exists(args.warming_csv):
            df = pd.read_csv(args.warming_csv)
            F.plot_paper_warming_co2(df, out("F10_warming_co2"))
            T.warming_co2_table(df, out("F10_warming_co2.tex"))
        else:
            _skip("F10", "provide --warming_csv")

    # ── F11: per-PFT ──
    if "F11" in wanted:
        if pft_df is not None:
            f11_choice = choices.get("F11") or {}
            F.plot_paper_pft_curves(
                pft_df,
                out("F11_pft_curves"),
                site_id=f11_choice.get("site_id"),
                year=f11_choice.get("year"),
            )
        else:
            _skip("F11", "provide --pft_csv")

    # ── F12: loss curves ──
    if "F12" in wanted:
        histories, stops = _loss_histories(registry, args.loss_runs)
        if histories:
            if len(histories) < 3:
                print(
                    "[warn] F12: only "
                    f"{len(histories)} architecture(s) found; expected "
                    "lstm, bitransformer and aelstm"
                )
            F.plot_paper_loss_curves(
                histories, out("F12_loss_curves"), early_stop_epochs=stops
            )
        else:
            _skip("F12", "no history.csv for LSTM / BiTransformer / AELSTM")

    # ── F13: memory kernel ──
    if "F13" in wanted:
        if args.memory_dir and os.path.isdir(args.memory_dir):
            kpath = os.path.join(args.memory_dir, "memory_kernel.csv")
            cpath = os.path.join(args.memory_dir, "memory_controls.csv")
            if os.path.exists(kpath):
                F.plot_paper_memory_kernel(
                    pd.read_csv(kpath),
                    pd.read_csv(cpath) if os.path.exists(cpath) else None,
                    out("F13_memory_kernel"),
                )
            else:
                _skip("F13", "memory_kernel.csv missing")
        else:
            _skip("F13", "provide --memory_dir")

    print(f"\nDone → {args.out_dir}")


if __name__ == "__main__":
    main()
