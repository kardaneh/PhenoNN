"""
run_perturbation.py: shared driver for the climate-perturbation baselines.

From an already-trained checkpoint it runs two inferences over the same sites
and years: one baseline, and one with a `Perturbation` forward-pre-hook
installed. It reports the causal impact of the perturbation on the model:

    Delta LAI = LAI(perturbed prediction) - LAI(baseline prediction)

(ceteris paribus: only the targeted channels move; see `_perturb.py`). It writes
a per-(site, year, dekad) CSV, a text summary, and two plots (mean Delta LAI seasonal
curve; histogram of per-sample mean Delta LAI) under `<output_dir>/<scenario>/`.

This module is imported by the four scenario scripts (drought.py, heatwave.py,
warming_co2.py, winter_frost.py); it is not meant to be run directly. It reuses
`phenonn.prediction.predict` for site/year resolution so nothing drifts.
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analysis._checkpoint_utils import load_state_dict
from phenonn.data.lai_dataset import (
    RamLAIDataset,
    load_parent01_map,
)
from phenonn.prediction.predict import _parse_year_spec as parse_year_spec
from phenonn.prediction.predict import _resolve_sites
from phenonn.utils.model_loader import build_model, build_model_pft
from phenonn.utils.utils import EasyDict
from phenonn.utils.wrappers import _OBS_POSITIONS

# 36 obs (month, day, doy): non-leap year (same convention as predict.py).
_OBS_DATES = [
    (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
    for m in range(1, 13)
    for d in [5, 15, 25]
]


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the command-line options shared by every scenario.

    Parameters
    ----------
    p : argparse.ArgumentParser
        Parser for one scenario script.
    """
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Trained run_best_model.pth.tar / best_model.pth.tar checkpoint.",
    )
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument(
        "--parent_map",
        default="",
        help="selected_pixels01_1.nc: falls back to the checkpoint's "
        "training --parent_map so features are read identically.",
    )
    p.add_argument(
        "--predict_sites",
        default="val",
        choices=["val", "train", "all", "grid", "test"],
    )
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Predict only on a selected_pixels*.nc site list.",
    )
    p.add_argument("--sites", default="", help="Comma-separated explicit site IDs.")
    p.add_argument(
        "--n_predict_sites",
        type=int,
        default=0,
        help="If > 0, random subsample of this many sites.",
    )
    p.add_argument(
        "--predict_years",
        default="",
        help="'2015-2018', '2015,2016' or 'all'. "
        "Empty -> val_years from the checkpoint.",
    )
    p.add_argument("--row_min", type=int, default=-1)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=-1)
    p.add_argument("--col_max", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hemisphere",
        default="auto",
        choices=["auto", "north", "south"],
        help="Season convention. 'auto' shifts the window per sample using "
        "the site latitude (recommended for global runs).",
    )
    p.add_argument(
        "--output_dir",
        default="runs/perturbation",
        help="Results go to <output_dir>/<scenario>/.",
    )


def load_site_latitudes(pft_dir: str, year: int, site_ids) -> dict:
    """Return ``{site_id: latitude}`` from ``PFTmap_{year}.nc``.

    The pixelset PFT files carry per-site ``latitude`` / ``longitude``
    coordinates, so the perturbation study can pick the correct hemisphere
    without a second metadata file.
    """
    import numpy as np

    path = os.path.join(pft_dir, f"PFTmap_{int(year)}.nc")
    if not os.path.exists(path):
        return {}
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        if "latitude" not in ds.coords and "latitude" not in ds.variables:
            return {}
        sites = np.asarray(ds["site_id"].values).astype(str)
        lats = np.asarray(ds["latitude"].values).astype(float)
    idx = {s: i for i, s in enumerate(sites)}
    return {str(s): float(lats[idx[s]]) for s in site_ids if str(s) in idx}


def load_model_and_data(args):
    """Load a checkpoint and build the matching evaluation dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed scenario arguments. The data flags fall back to the paths stored
        in the checkpoint.

    Returns
    -------
    EasyDict
        Inference context with the model, dataset, loader, normalization
        statistics, anomaly climatology, and the optional ablation hook.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = EasyDict(ckpt["args"])
    norm_stats = ckpt.get("norm_stats", None)
    co2_lut = ckpt.get("co2_lut", None)
    anomaly_clim = ckpt.get("anomaly_clim", None) if ckpt.get("anomaly_mode") else None
    is_anomaly = bool(ckpt.get("anomaly_mode", False))
    is_normalized = norm_stats is not None
    lai_normalized = bool(ckpt.get("normalize_lai", True))
    is_pft_mixing = bool(ckpt.get("pft_mixing", False))

    print(f"Checkpoint    : {args.checkpoint}  (epoch {ckpt.get('epoch')})")
    print(
        f"Flags         : norm={is_normalized}, lai_norm={lai_normalized}, "
        f"pft_mixing={is_pft_mixing}, anomaly={is_anomaly}, "
        f"co2={co2_lut is not None}"
    )

    features_dir = args.features_dir or train_args.get("features_dir", "")
    target_dir = args.target_dir or train_args.get("target_dir", "")
    pft_dir = args.pft_dir or train_args.get("pft_dir", "")
    if not features_dir or not target_dir or not pft_dir:
        raise ValueError(
            "Provide --features_dir/--target_dir/--pft_dir "
            "or use a checkpoint that stored them."
        )

    parent_map_path = args.parent_map or train_args.get("parent_map", "")
    parent_map = None
    if parent_map_path:
        if not os.path.exists(parent_map_path):
            raise FileNotFoundError(parent_map_path)
        parent_map = load_parent01_map(parent_map_path)
        print(f"Parent map    : {parent_map_path}  ({len(parent_map):,} links)")

    if is_pft_mixing:
        model = build_model_pft(train_args, norm_stats).to(device)
    else:
        model = build_model(train_args).to(device)
    model.load_state_dict(load_state_dict(ckpt))
    model.eval()
    print(
        f"Model         : {train_args.get('type')} "
        f"({'PFTMixing' if is_pft_mixing else 'Every10Days'})"
    )

    # -- If the model was TRAINED with ablated features, the same channels must
    #    be zeroed at inference: otherwise it sees inputs it never saw. The hook
    #    is returned (not registered here) so run() can order it AFTER the
    #    perturbation hook: perturb physical values, THEN neutralise the ablated
    #    channels, exactly as during training.
    abl_feats = [
        f.strip()
        for f in str(train_args.get("ablate_features", "") or "").split(",")
        if f.strip()
    ]
    ablate_hook = None
    if abl_feats:
        from analysis.feature_ablation.run_ablate_inproc import _channels

        abl_ch = _channels(abl_feats)

        def ablate_hook(_mod, inputs):
            t = inputs[0]
            if not torch.is_tensor(t):
                return None
            t = t.clone()
            for c in abl_ch:
                t[:, c, :] = 0.0
            return (t,) + tuple(inputs[1:])

        print(
            f"Ablation      : re-applied at inference {abl_feats} (channels {abl_ch})"
        )

    site_ids = _resolve_sites(args, ckpt, train_args)
    if not site_ids:
        raise RuntimeError("Empty site list.")
    if 0 < args.n_predict_sites < len(site_ids):
        rng = np.random.RandomState(args.seed)
        site_ids = rng.choice(
            site_ids, size=args.n_predict_sites, replace=False
        ).tolist()
    print(f"Sites         : {len(site_ids):,}")

    years = parse_year_spec(args.predict_years)
    if years is None:
        years = parse_year_spec(str(train_args.get("val_years", "") or ""))
        if not years:
            raise ValueError("Could not infer --predict_years.")
    print(f"Years         : {years}")

    print("Building dataset ...")
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
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Per-sample latitude for hemisphere-aware season windows.
    lat_map = load_site_latitudes(pft_dir, years[0], site_ids)
    lat_by_sample = np.array(
        [
            lat_map.get(dataset.get_site_info(i)["site_id"], 0.0)
            for i in range(len(dataset))
        ],
        dtype=np.float32,
    )
    if lat_map:
        print(
            f"Latitudes     : {len(lat_map):,} sites "
            f"({np.mean(lat_by_sample < 0) * 100:.0f}% southern hemisphere)"
        )

    return EasyDict(
        {
            "device": device,
            "model": model,
            "dataset": dataset,
            "loader": loader,
            "norm_stats": norm_stats,
            "anomaly_clim": anomaly_clim,
            "is_anomaly": is_anomaly,
            "lai_normalized": lai_normalized,
            "is_normalized": is_normalized,
            "ablate_hook": ablate_hook,
            "ablate_feats": abl_feats,
            "ablate_ch": abl_ch if abl_feats else [],
            "lat_by_sample": lat_by_sample,
            "is_daily": bool(train_args.get("daily_lai", False)),
            "co2_lut": co2_lut,
            "train_args": train_args,
        }
    )


def _infer(model, loader, device) -> np.ndarray:
    """Run one inference pass and stack the raw model outputs.

    Parameters
    ----------
    model : torch.nn.Module
        Model in evaluation mode.
    loader : torch.utils.data.DataLoader
        Data loader for the selected sites and years.
    device : torch.device
        Device used for inference.

    Returns
    -------
    numpy.ndarray
        Raw model outputs with shape ``(N, 36)``.
    """
    out = []
    with torch.no_grad():
        for features, _targets in loader:
            preds = model(features.to(device)).cpu().numpy()  # (B, 1, 36)
            out.append(preds[:, 0, :])
    return np.concatenate(out, axis=0)


def _infer_perturbed(
    model,
    loader,
    perturbation,
    lats,
    device,
    ablate_ch=None,
) -> np.ndarray:
    """Run one perturbed inference pass, optionally hemisphere-aware.

    Unlike a forward hook, this path knows the per-sample latitude, which is
    required for the ``hemisphere="auto"`` season convention on a global
    dataset.  The ablation channels are zeroed after the perturbation so the
    order matches training-time feature ablation.
    """
    out = []
    start = 0
    with torch.no_grad():
        for features, _targets in loader:
            b = features.size(0)
            x = features.to(device)
            batch_lats = None if lats is None else lats[start : start + b]
            x = perturbation.apply(x, lats=batch_lats)
            if ablate_ch:
                x = x.clone()
                for c in ablate_ch:
                    x[:, c, :] = 0.0
            preds = model(x).cpu().numpy()  # (B, 1, 36)
            out.append(preds[:, 0, :])
            start += b
    return np.concatenate(out, axis=0)


def _to_physical(pred_raw: np.ndarray, ctx, site_ids) -> np.ndarray:
    """Convert raw model outputs to physical LAI.

    Parameters
    ----------
    pred_raw : numpy.ndarray
        Raw model outputs with shape ``(N, 36)``.
    ctx : EasyDict
        Inference context returned by :func:`load_model_and_data`.
    site_ids : sequence
        Site ID for each row of ``pred_raw``.

    Returns
    -------
    numpy.ndarray
        LAI values in physical units, with anomaly climatology added when the
        checkpoint uses anomaly mode.
    """
    if ctx.is_anomaly:
        real = pred_raw.copy()
        for i, s in enumerate(site_ids):
            clim = ctx.anomaly_clim.get(s)
            if clim is not None:
                real[i] = pred_raw[i] + clim
        return real
    if ctx.is_normalized and ctx.lai_normalized:
        mu = float(ctx.norm_stats["LAI"]["mean"])
        sd = float(ctx.norm_stats["LAI"]["std"])
        return pred_raw * sd + mu
    return pred_raw


def run_from_context(
    ctx,
    args,
    perturbation,
    scenario: str,
    description: str = "",
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Run baseline and perturbed inference with an already loaded context.

    Parameters
    ----------
    ctx : EasyDict
        Output of :func:`load_model_and_data`: model, dataset, loader,
        normalisation statistics and ablation hooks.
    args : argparse.Namespace
        Parsed scenario arguments.  ``output_dir`` is only used when
        ``write_outputs`` is true.
    perturbation : Perturbation
        Perturbation to install on the model input.
    scenario : str
        Scenario name used for the output subdirectory and summary.
    description : str, optional
        Human-readable description written to the summary.
    write_outputs : bool
        When true, write ``deltas.csv``, ``summary.txt`` and the two plots
        under ``<output_dir>/<scenario>/``.  Set false for sweeps that only
        need the returned table.

    Returns
    -------
    pandas.DataFrame
        Per-site, per-year, per-dekad baseline and perturbed LAI.
    """
    print(f"\nScenario      : {scenario}")
    print(f"Perturbation  : {perturbation.describe()}")

    n = len(ctx.dataset)
    site_ids = [ctx.dataset.get_site_info(i)["site_id"] for i in range(n)]
    years = [ctx.dataset.get_site_info(i)["year"] for i in range(n)]

    # Hooks fire in registration order -> register the perturbation FIRST and the
    # ablation LAST, so ablated channels stay neutralised as during training.
    print("Baseline inference ...")
    h_abl = (
        ctx.model.register_forward_pre_hook(ctx.ablate_hook)
        if ctx.ablate_hook is not None
        else None
    )
    try:
        base_raw = _infer(ctx.model, ctx.loader, ctx.device)
    finally:
        if h_abl is not None:
            h_abl.remove()

    print("Perturbed inference ...")
    pert_raw = _infer_perturbed(
        ctx.model,
        ctx.loader,
        perturbation,
        ctx.lat_by_sample,
        ctx.device,
        ablate_ch=ctx.ablate_ch,
    )

    # A daily-trained model outputs 365 days -> sample the 36 dekad positions so
    # everything downstream (36-dekad tables, obs comparison) stays aligned.
    if ctx.is_daily:
        pos = np.asarray(_OBS_POSITIONS, dtype=np.int64)
        print("Daily model   : sampling 365-day output at the 36 dekads")
        base_raw, pert_raw = base_raw[:, pos], pert_raw[:, pos]

    base = _to_physical(base_raw, ctx, site_ids)
    pert = _to_physical(pert_raw, ctx, site_ids)
    delta = pert - base  # (N, 36)

    rows = []
    for i in range(n):
        for k, (month, day, doy) in enumerate(_OBS_DATES):
            rows.append(
                {
                    "site_id": site_ids[i],
                    "year": years[i],
                    "month": month,
                    "day": day,
                    "doy": doy,
                    "lai_base": float(base[i, k]),
                    "lai_pert": float(pert[i, k]),
                    "delta": float(delta[i, k]),
                }
            )
    df = pd.DataFrame(rows)

    if not write_outputs:
        return df

    out_dir = os.path.join(args.output_dir, scenario)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "deltas.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDelta LAI table    -> {csv_path}")

    _write_summary(df, out_dir, scenario, description, perturbation)
    _plot(df, out_dir, scenario)
    print("Done.")
    return df


def run(args, make_perturbation, scenario: str, description: str = "") -> None:
    """Run baseline and perturbed inference and write the result files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed scenario arguments.
    make_perturbation : callable
        Factory that receives ``norm_stats`` and returns a ``Perturbation``. The
        factory is needed because normalization statistics are loaded with the
        checkpoint.
    scenario : str
        Scenario name used for the output subdirectory.
    description : str, optional
        Human-readable description written to the summary.
    """
    ctx = load_model_and_data(args)
    perturbation = make_perturbation(ctx.norm_stats)
    run_from_context(ctx, args, perturbation, scenario, description)


def _write_summary(df, out_dir, scenario, description, perturbation) -> None:
    """Write aggregate Delta LAI statistics to ``summary.txt``."""
    d = df["delta"].values
    peak = df.loc[df["doy"].between(152, 243), "delta"].values  # summer peak
    lines = [
        f"-- Perturbation scenario: {scenario} --",
        description,
        f"Perturbation  : {perturbation.describe()}",
        "",
        f"Samples (site/year/dekad) : {len(df):,}",
        f"Mean   Delta LAI               : {np.nanmean(d):+.4f}",
        f"Median Delta LAI               : {np.nanmedian(d):+.4f}",
        f"Std    Delta LAI               : {np.nanstd(d):.4f}",
        f"Min / Max Delta LAI            : {np.nanmin(d):+.4f} / {np.nanmax(d):+.4f}",
        f"Mean |Delta LAI|               : {np.nanmean(np.abs(d)):.4f}",
        f"Mean Delta LAI (summer JJA)    : {np.nanmean(peak):+.4f}"
        if peak.size
        else "Mean Delta LAI (summer JJA)    : n/a",
        (f"Fraction of dekads |Delta|>0.05: {100.0 * np.mean(np.abs(d) > 0.05):.1f}%"),
    ]
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Summary       -> {path}")


def _plot(df, out_dir, scenario) -> None:
    """Write the seasonal curve and per-site-year Delta LAI histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mean Delta LAI seasonal curve (over the 36 dekad DOYs).
    by_doy = df.groupby("doy")["delta"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.plot(by_doy.index, by_doy.values, marker="o", ms=3)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Mean Delta LAI (perturbed - baseline)")
    ax.set_title(f"{scenario}: seasonal mean Delta LAI")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_seasonal.png"), dpi=130)
    plt.close(fig)

    # Histogram of per-sample mean Delta LAI (one value per site/year).
    per_sample = df.groupby(["site_id", "year"])["delta"].mean().values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.hist(per_sample[np.isfinite(per_sample)], bins=50)
    ax.set_xlabel("Mean Delta LAI per site/year")
    ax.set_ylabel("Count")
    ax.set_title(f"{scenario}: Delta LAI distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_hist.png"), dpi=130)
    plt.close(fig)
    print(f"Plots         -> {out_dir}/delta_seasonal.png, delta_hist.png")
