#!/usr/bin/env python3
# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
LAI inference for PhenoNN checkpoints.

Reads the pixelset NetCDF layout and writes a per-``(site, year, dekad)``
CSV of predictions and observations, along with diagnostic plots.

Checkpoint schema
-----------------
The predictor reads what :meth:`ModelUtils.save_training_checkpoint` writes.
The following keys are expected at the top level of the snapshot:

``state_dict``
    Model weights (older checkpoints may carry ``model_state_dict``).
``args``
    The training arguments, as a plain dict.
``epoch``, ``val_loss``, ``best_val_loss``, ``best_epoch``
    Training state.
``val_rmse``, ``val_r2``
    Latest validation metrics.
``norm_stats``, ``co2_lut``, ``anomaly_clim``
    The auxiliary tables (may be ``None``).
``pft_mixing``
    Whether the PFT mixing wrapper is enabled.
``pft_meteo_only``
    Whether the model is forced to use meteorological inputs only.
``normalize_lai``
    Whether the LAI target is normalised.
``daily_lai``
    Whether daily LAI is used.
``anomaly_mode``
    The anomaly mode used to build the target.
``train_site_ids``, ``val_site_ids``
    The site pools.

Missing keys fall back to the same value read from ``args``; older
snapshots (written before the top-level fields existed) therefore keep
loading.

Module overview
---------------
The inference pipeline is a chain of small functions:

1.  ``parse_args``                     — CLI
2.  ``load_checkpoint``                — read the snapshot via ModelUtils
3.  ``log_configuration``              — dump the flags
4.  ``resolve_data_directories``       — CLI vs checkpoint-stored paths
5.  ``load_parent_map_if_requested``   — optional 0.05°→0.1° mapping
6.  ``initialize_model``               — build + load weights
7.  ``install_ablation_hook``          — replay the training-time ablation
8.  ``_resolve_sites``                 — CLI flag → list of site ids
9.  ``resolve_years``                  — CLI flag → list of years
10. ``create_dataset_and_loader``      — RAM-resident dataset + DataLoader
11. ``run_inference``                  — forward pass, denormalise, CSV rows
12. ``compute_metrics``                — pooled / per-site / decomposition
13. ``write_outputs``                  — CSV, metrics text, plots

Site selection
--------------
The site pool is chosen by the first of the following flags that is set:

``--selected_pixels P``
    Sites listed in a ``selected_pixels*.nc`` file. Overrides ``--sites``
    and ``--predict_sites``.
``--sites A,B,C``
    Explicit comma-separated site IDs. Overrides ``--predict_sites``.
``--predict_sites val``
    The checkpoint's ``val_site_ids``.
``--predict_sites train``
    The checkpoint's ``train_site_ids``.
``--predict_sites all``
    Union of ``train_site_ids`` and ``val_site_ids``.
``--predict_sites grid``
    Every pixel ID in the bbox ``--row_min/--row_max/--col_min/--col_max``,
    falling back to the training bbox when a value is -1.
``--predict_sites test``
    ``grid`` minus ``val_site_ids``.

Usage
-----

.. code-block:: bash

    python -m phenonn.prediction.predict \\
        --checkpoint runs/exp/checkpoints/best_model.pth.tar \\
        --predict_sites val \\
        --output_csv runs/exp/predictions.csv
"""

import argparse
import datetime
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phenonn.utils.config import ALL_FEATURES, PFT_FNAME, PFT_NAMES
from phenonn.utils.logger import Logger
from phenonn.utils.wrappers import _OBS_POSITIONS
from phenonn.evaluation import bundle as B
from phenonn.evaluation import metrics as M
from phenonn.evaluation.leakage import (
    LeakageError,
    parse_years,
    validate_climatology_years,
)
from phenonn.evaluation.reproducibility import collect_run_metadata
from phenonn.data.lai_dataset import (
    RamLAIDataset,
    generate_site_ids_from_range,
    load_parent01_map,
    load_selected_pixels,
)
from phenonn.utils.model_loader import build_model, build_model_pft
from phenonn.utils.diagnostics import (
    plot_pred_vs_obs,
    plot_gcc_curves,
    plot_gcc_curves_all,
)
from phenonn.utils.utils import EasyDict


# 36 obs (month, day, doy) — non-leap year
_OBS_DATES = [
    (m, d, datetime.date(2001, m, d).timetuple().tm_yday)
    for m in range(1, 13)
    for d in [5, 15, 25]
]


# ── Phase 1 — CLI ───────────────────────────────────────────────────────────


def parse_args():
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="PhenoNN inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument(
        "--parent_map",
        default="",
        help="Optional selected_pixels01_1.nc. When set, ERA5 features are "
        "read from the deduplicated 0.1° pixelset via each 0.05° site's "
        "parent cell, instead of a 0.05°-indexed feature file. "
        "Must match the --parent_map used at training.",
    )
    p.add_argument(
        "--predict_sites",
        default="val",
        choices=["val", "train", "all", "grid", "test"],
    )
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Path to a selected_pixels*.nc. Overrides --sites and " "--predict_sites.",
    )
    p.add_argument(
        "--sites",
        default="",
        help="Comma-separated explicit site IDs — overrides --predict_sites.",
    )
    p.add_argument(
        "--n_predict_sites",
        type=int,
        default=0,
        help="If > 0, random subsample of this many sites.",
    )
    p.add_argument(
        "--predict_years",
        default="",
        help="'2015-2018', '2015,2016' or 'all'. Empty → val_years from the ckpt.",
    )
    p.add_argument("--row_min", type=int, default=-1)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=-1)
    p.add_argument("--col_max", type=int, default=-1)
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scatter_years",
        action="store_true",
        help="One pred-vs-obs scatter per evaluated year, in a "
        "`scatter_year/` sub-folder next to --output_csv.",
    )
    p.add_argument(
        "--n_curves",
        type=int,
        default=0,
        help="Number of per-site curves to plot in *_lai_curves_all.png. "
        "0 = plot every site that produced predictions.",
    )
    p.add_argument(
        "--ablate_features",
        default="",
        help="Comma list of features the model was TRAINED without. "
        "Normally read from the checkpoint.",
    )
    p.add_argument(
        "--pft_min_frac",
        type=float,
        default=0.05,
        help="Show PFTs with fraction ≥ this threshold above each curve "
        "subplot (default: 0.05 = 5%%).",
    )
    # ── Paper-protocol bundle options ──
    p.add_argument(
        "--clim_years",
        default="",
        help="Years used for the anomaly climatology, e.g. '1992-2009'. "
        "Empty -> train_years stored in the checkpoint.",
    )
    p.add_argument(
        "--no_climatology",
        action="store_true",
        help="Skip the anomaly climatology and the anomaly metrics "
        "(quick diagnostics only; paper runs keep it on).",
    )
    p.add_argument(
        "--clim_target_dir",
        default="",
        help="Folder of LAI_dekadal_{Y}.nc used to build the climatology. "
        "Empty -> --target_dir / checkpoint target_dir.",
    )
    p.add_argument(
        "--bundle_dir",
        default="",
        help="Directory for metrics.json / per-site tables. "
        "Empty -> directory of --output_csv.",
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap draws over sites for the 95%% CIs (0 disables).",
    )
    p.add_argument(
        "--protocol",
        default="none",
        choices=["A", "B", "none"],
        help="Split protocol used for leakage checks.",
    )
    p.add_argument(
        "--test_years",
        default="",
        help="Test years, e.g. '2010-2019'. Used by the leakage checks.",
    )
    p.add_argument(
        "--allow_leakage",
        action="store_true",
        help="Downgrade leakage errors to warnings (not for paper runs).",
    )
    p.add_argument(
        "--no_pft_columns",
        action="store_true",
        help="Do not attach pft_dom / bare_frac / purity to the predictions.",
    )
    p.add_argument(
        "--no_phenology_metrics",
        action="store_true",
        help="Skip pheno_metrics.csv (slow on >200k site-years).",
    )
    p.add_argument(
        "--pheno_max_siteyears",
        type=int,
        default=0,
        help="If > 0, compute phenology on a deterministic subsample of "
        "site-years larger than this.",
    )
    return p.parse_args()


# ── Phase 2 — checkpoint ────────────────────────────────────────────────────


def _parse_year_spec(spec) -> Optional[List[int]]:
    """
    Parse a year string such as ``"2010-2019"`` or ``"2010,2015"``.

    Parameters
    ----------
    spec : str or None
        The year specification. ``"all"`` and empty strings return ``None``.

    Returns
    -------
    list of int or None
    """
    if not spec:
        return None
    spec = str(spec)
    if spec.lower() == "all":
        return None
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def _reconstruct_auxiliary_tables(
    ckpt_raw: Dict, train_args: Dict, logger: Logger
) -> None:
    """
    Rebuild the auxiliary tables on disk if they are missing from the snapshot.

    The predictor needs three tables: the normalisation statistics, the CO2
    lookup, and (in anomaly mode) the per-site climatology. Modern
    snapshots carry them; older ones do not. When a table is missing but
    its source path is stored in ``args``, the loader is re-run.

    Parameters
    ----------
    ckpt_raw : dict
        The raw snapshot. Modified in place.
    train_args : dict
        The training arguments stored in the snapshot.
    logger : Logger
    """
    import json

    # Normalisation statistics.
    if ckpt_raw.get("norm_stats") is None and train_args.get("stats_path"):
        path = train_args["stats_path"]
        if os.path.exists(path):
            with open(path) as f:
                ckpt_raw["norm_stats"] = json.load(f)
            logger.info(f"Norm stats    : rebuilt from {path}")

    # CO2 lookup.
    if ckpt_raw.get("co2_lut") is None and train_args.get("co2_path"):
        from phenonn.data.lai_dataset import load_co2_lut

        path = train_args["co2_path"]
        if os.path.exists(path):
            ckpt_raw["co2_lut"] = load_co2_lut(path)
            logger.info(f"CO2 LUT       : rebuilt from {path}")

    # Per-site climatology (anomaly mode only).
    if ckpt_raw.get("anomaly_clim") is None and bool(
        ckpt_raw.get("anomaly_mode", train_args.get("anomaly_mode", False))
    ):
        from phenonn.data.lai_dataset import compute_climatology_lookup

        clim_target_dir = train_args.get("clim_target_dir") or train_args.get(
            "target_dir", ""
        )
        clim_years = _parse_year_spec(train_args.get("clim_years", "1992-2010")) or []
        site_pool = sorted(
            set(ckpt_raw.get("train_site_ids", []))
            | set(ckpt_raw.get("val_site_ids", []))
        )
        if clim_years and site_pool:
            logger.start_task(
                "Rebuilding climatology",
                f"{len(clim_years)} years × {len(site_pool):,} sites",
            )
            ckpt_raw["anomaly_clim"] = compute_climatology_lookup(
                target_dir=clim_target_dir,
                clim_years=clim_years,
                wanted_sites=site_pool,
                verbose=False,
            )
            logger.success(
                f"  {len(ckpt_raw['anomaly_clim']):,} site climatologies built"
            )


def load_checkpoint(args, device: torch.device, logger: Logger) -> Dict:
    """
    Load a training snapshot and normalise its schema.

    Reads the raw checkpoint with :func:`torch.load`, then unifies the
    schema across the two generations of writers: modern snapshots have
    the auxiliary tables and mode flags at the top level, older ones only
    in ``args``. Returns a dictionary with a uniform shape regardless.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI. Reads ``checkpoint``.
    device : torch.device
        Where the state dict is mapped.
    logger : Logger

    Returns
    -------
    dict
        Uniform checkpoint with keys ``state_dict``, ``args``, ``epoch``,
        ``val_loss``, ``val_rmse``, ``val_r2``, ``norm_stats``,
        ``co2_lut``, ``anomaly_clim``, ``pft_mixing``, ``pft_meteo_only``,
        ``normalize_lai``, ``daily_lai``, ``anomaly_mode``,
        ``train_site_ids``, ``val_site_ids``.

    Raises
    ------
    FileNotFoundError
        If ``args.checkpoint`` does not exist.
    KeyError
        If the snapshot carries neither ``state_dict`` nor
        ``model_state_dict``.
    """
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    logger.start_task("Loading checkpoint", args.checkpoint)
    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # ── State dict: current name, then legacy name ──
    state_dict = raw.get("state_dict") or raw.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            "Checkpoint has neither 'state_dict' nor 'model_state_dict'. "
            f"Keys present: {sorted(raw.keys())}"
        )

    # ── Training args: plain dict with every flag the trainer saw ──
    train_args = dict(raw.get("args", {}))

    # ── Rebuild any auxiliary table the snapshot does not carry ──
    _reconstruct_auxiliary_tables(raw, train_args, logger)

    # ── Mode flags: top level first, then args, then a sensible default ──
    pft_mixing = bool(raw.get("pft_mixing", train_args.get("pft_mixing", False)))
    pft_meteo_only = bool(
        raw.get("pft_meteo_only", train_args.get("pft_meteo_only", False))
    )
    normalize_lai = bool(
        raw.get("normalize_lai", train_args.get("normalize_lai", True))
    )
    daily_lai = bool(raw.get("daily_lai", train_args.get("daily_lai", False)))
    anomaly_mode = bool(raw.get("anomaly_mode", train_args.get("anomaly_mode", False)))

    # ── Site pools: top level first, then args ──
    train_site_ids = list(
        raw.get("train_site_ids", train_args.get("train_site_ids", []))
    )
    val_site_ids = list(raw.get("val_site_ids", train_args.get("val_site_ids", [])))

    ckpt = {
        "state_dict": state_dict,
        "args": train_args,
        "epoch": raw.get("epoch", 0),
        "val_loss": raw.get("val_loss", float("nan")),
        "val_rmse": raw.get("val_rmse", float("nan")),
        "val_r2": raw.get("val_r2", float("nan")),
        "norm_stats": raw.get("norm_stats", None),
        "co2_lut": raw.get("co2_lut", None),
        "anomaly_clim": raw.get("anomaly_clim", None),
        "anomaly_mode": anomaly_mode,
        "pft_mixing": pft_mixing,
        "pft_meteo_only": pft_meteo_only,
        "normalize_lai": normalize_lai,
        "daily_lai": daily_lai,
        "train_site_ids": train_site_ids,
        "val_site_ids": val_site_ids,
    }

    logger.success(
        f"  epoch={ckpt['epoch']}, "
        f"val_loss={ckpt['val_loss']:.6f}, "
        f"pft_mixing={ckpt['pft_mixing']}, "
        f"normalize_lai={ckpt['normalize_lai']}, "
        f"anomaly={ckpt['anomaly_mode']}"
    )
    return ckpt


def log_configuration(ckpt: Dict, args, logger: Logger) -> None:
    """
    Log the checkpoint's stored flags and the inference-time flags.

    Parameters
    ----------
    ckpt : dict
        The dictionary returned by :func:`load_checkpoint`.
    args : argparse.Namespace
        Parsed CLI arguments.
    logger : Logger
    """
    logger.info(f"Checkpoint    : {args.checkpoint}")
    logger.info(f"Epoch         : {ckpt['epoch']}")
    logger.info(f"Val RMSE / R² : {ckpt['val_rmse']:.5f} / {ckpt['val_r2']:.4f}")
    logger.info(
        f"Flags         : norm={ckpt['norm_stats'] is not None}, "
        f"lai_norm={ckpt['normalize_lai']}, "
        f"pft_mixing={ckpt['pft_mixing']}, "
        f"anomaly={ckpt['anomaly_mode']}, "
        f"co2={ckpt['co2_lut'] is not None}, "
        f"daily_lai={ckpt['daily_lai']}"
    )


# ── Phase 4 — data directories ──────────────────────────────────────────────


def resolve_data_directories(args, train_args) -> Tuple[str, str, str]:
    """
    Resolve the three data directories, CLI overriding the checkpoint.

    Parameters
    ----------
    args : argparse.Namespace
    train_args : dict
        The checkpoint's stored arguments.

    Returns
    -------
    features_dir, target_dir, pft_dir : tuple of str

    Raises
    ------
    ValueError
        If any of the three directories is still empty after the merge.
    """
    features_dir = args.features_dir or train_args.get("features_dir", "")
    target_dir = args.target_dir or train_args.get("target_dir", "")
    pft_dir = args.pft_dir or train_args.get("pft_dir", "")
    if not features_dir or not target_dir or not pft_dir:
        raise ValueError(
            "Provide --features_dir, --target_dir, --pft_dir "
            "or use a checkpoint that stored them."
        )
    return features_dir, target_dir, pft_dir


# ── Phase 5 — parent map ────────────────────────────────────────────────────


def load_parent_map_if_requested(
    args,
    train_args,
    logger: Logger,
) -> Optional[Dict[str, str]]:
    """
    Load the optional 0.05°→0.1° parent map.

    Parameters
    ----------
    args : argparse.Namespace
    train_args : dict
    logger : Logger

    Returns
    -------
    dict or None
    """
    parent_map_path = args.parent_map or train_args.get("parent_map", "")
    if not parent_map_path:
        return None
    if not os.path.exists(parent_map_path):
        raise FileNotFoundError(parent_map_path)
    parent_map = load_parent01_map(parent_map_path)
    logger.info(
        f"Parent map    : {parent_map_path}  " f"({len(parent_map):,} 0.05°→0.1° links)"
    )
    return parent_map


# ── Phase 6 — model ─────────────────────────────────────────────────────────


def initialize_model(
    ckpt: Dict,
    train_args,
    device,
    logger: Logger,
) -> Tuple[nn.Module, str]:
    """
    Rebuild the model and load its weights.

    Parameters
    ----------
    ckpt : dict
        The dictionary returned by :func:`load_checkpoint`.
    train_args : dict
    device : torch.device
    logger : Logger

    Returns
    -------
    model : nn.Module
    wrapper_label : str
    """
    if ckpt["pft_mixing"]:
        model = build_model_pft(EasyDict(train_args), ckpt["norm_stats"]).to(device)
        wrapper_label = "PFTMixingWrapper"
    else:
        model = build_model(EasyDict(train_args)).to(device)
        wrapper_label = "Every10DaysWrapper"
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"Model         : {train_args.get('type')} + {wrapper_label}")
    return model, wrapper_label


# ── Phase 7 — ablation replay ───────────────────────────────────────────────


def install_ablation_hook(
    model: nn.Module,
    args,
    train_args,
    logger: Logger,
) -> None:
    """
    Zero the same channels the model was trained without.

    Parameters
    ----------
    model : nn.Module
    args : argparse.Namespace
    train_args : dict
    logger : Logger
    """
    abl_spec = args.ablate_features or str(train_args.get("ablate_features", "") or "")
    abl_feats = [f.strip() for f in abl_spec.split(",") if f.strip()]
    if not abl_feats:
        return
    abl_ch = [ALL_FEATURES.index(f) for f in abl_feats]

    def _ablate_hook(_mod, inputs):
        t = inputs[0]
        if not torch.is_tensor(t):
            return None
        t = t.clone()
        for c in abl_ch:
            t[:, c, :] = 0.0
        return (t,) + tuple(inputs[1:])

    model.register_forward_pre_hook(_ablate_hook)
    logger.info(f"Ablation      : re-applied {abl_feats} (channels {abl_ch})")


# ── Phase 8 — site resolution ───────────────────────────────────────────────


def _resolve_sites(args, ckpt: Dict, train_args) -> List[str]:
    """
    Resolve the CLI flags into a concrete list of site ids.

    Parameters
    ----------
    args : argparse.Namespace
    ckpt : dict
        The dictionary returned by :func:`load_checkpoint`.
    train_args : dict

    Returns
    -------
    list of str
    """
    if args.selected_pixels:
        return load_selected_pixels(args.selected_pixels)
    if args.sites:
        return [s.strip() for s in args.sites.split(",") if s.strip()]

    mode = args.predict_sites
    train_pool = ckpt["train_site_ids"]
    val_pool = ckpt["val_site_ids"]

    if mode == "val":
        return val_pool
    if mode == "train":
        return train_pool
    if mode == "all":
        return sorted(set(train_pool) | set(val_pool))

    rmin = args.row_min if args.row_min >= 0 else int(train_args.get("row_min", 0))
    rmax = args.row_max if args.row_max >= 0 else int(train_args.get("row_max", 0))
    cmin = args.col_min if args.col_min >= 0 else int(train_args.get("col_min", 0))
    cmax = args.col_max if args.col_max >= 0 else int(train_args.get("col_max", 0))
    grid = generate_site_ids_from_range((rmin, rmax), (cmin, cmax))

    if mode == "grid":
        return grid
    if mode == "test":
        if not val_pool:
            raise RuntimeError("'test' mode needs val_site_ids in the checkpoint.")
        return sorted(set(grid) - set(val_pool))
    raise ValueError(mode)


# ── Phase 9 — year resolution ───────────────────────────────────────────────


def resolve_years(args, train_args) -> List[int]:
    """
    Resolve the prediction years.

    Parameters
    ----------
    args : argparse.Namespace
    train_args : dict

    Returns
    -------
    list of int

    Raises
    ------
    ValueError
        If no year can be inferred from either the CLI or the checkpoint.
    """
    years = _parse_year_spec(args.predict_years)
    if years is not None:
        return years
    years = _parse_year_spec(str(train_args.get("val_years", "") or ""))
    if not years:
        raise ValueError("Could not infer predict_years.")
    return years


# ── Phase 9b — climatology and leakage ─────────────────────────────────────


def resolve_climatology(
    args,
    ckpt: Dict,
    train_args: Dict,
    target_dir: str,
    site_ids: List[str],
    years: List[int],
    logger: Logger,
) -> Tuple[Optional[Dict], List[int]]:
    """Build (or reuse) the per-site climatology used for anomaly metrics.

    The years are taken from ``--clim_years`` first, then from the training
    checkpoint's ``train_years``.  They must be a subset of the training years;
    otherwise :class:`LeakageError` is raised unless ``--allow_leakage`` is set.
    """
    clim_years = parse_years(args.clim_years) or parse_years(
        train_args.get("train_years", "")
    )
    train_years = parse_years(train_args.get("train_years", ""))

    if getattr(args, "no_climatology", False):
        logger.info("Climatology   : disabled by --no_climatology")
        return None, []

    clim = ckpt.get("anomaly_clim")
    if clim:
        # The checkpoint's climatology is authoritative: validate the years it
        # was actually built on, not the ones requested on the command line.
        ckpt_clim_years = parse_years(ckpt.get("clim_years") or [])
        if ckpt_clim_years:
            if clim_years and clim_years != ckpt_clim_years:
                logger.warning(
                    "Checkpoint climatology was built on "
                    f"{ckpt_clim_years[0]}–{ckpt_clim_years[-1]}; ignoring "
                    f"--clim_years={args.clim_years!r}."
                )
            clim_years = ckpt_clim_years
        logger.info(f"Climatology   : {len(clim):,} sites from checkpoint")
    elif clim_years:
        clim_dir = (
            args.clim_target_dir
            or train_args.get("clim_target_dir", "")
            or target_dir
        )
        logger.start_task(
            "Building climatology",
            f"{len(clim_years)} years × {len(site_ids):,} sites ({clim_dir})",
        )
        from phenonn.data.lai_dataset import compute_climatology_lookup

        clim = compute_climatology_lookup(
            target_dir=clim_dir,
            clim_years=clim_years,
            wanted_sites=site_ids,
            verbose=False,
        )
        logger.success(f"  {len(clim):,} site climatologies built")
    else:
        logger.warning(
            "No climatology years available — anomaly metrics will be skipped. "
            "Pass --clim_years explicitly."
        )
        return None, []

    if train_years:
        try:
            for w in validate_climatology_years(
                clim_years,
                train_years,
                predict_years=years,
                strict=not args.allow_leakage,
            ):
                logger.warning(w)
        except LeakageError as exc:
            raise LeakageError(
                f"{exc}\nSet --allow_leakage only for a documented diagnostic run."
            ) from exc
    if args.protocol in ("A", "B"):
        test_years = parse_years(args.test_years) or (
            years if args.protocol == "B" else []
        )
        overlap = sorted(set(clim_years) & set(test_years))
        if overlap and not args.allow_leakage:
            raise LeakageError(
                "anomaly climatology overlaps test years "
                f"{overlap[0]}–{overlap[-1]} under protocol {args.protocol}"
            )
    return clim, clim_years


# ── Phase 10 — dataset and loader ───────────────────────────────────────────


def create_dataset_and_loader(
    args,
    features_dir,
    target_dir,
    pft_dir,
    years,
    site_ids,
    train_args,
    norm_stats,
    anomaly_clim,
    co2_lut,
    parent_map,
    logger: Logger,
) -> Tuple[RamLAIDataset, DataLoader]:
    """
    Build the RAM-resident dataset and its DataLoader.

    Parameters
    ----------
    args : argparse.Namespace
    features_dir, target_dir, pft_dir : str
    years : list of int
    site_ids : list of str
    train_args : dict
    norm_stats, anomaly_clim, co2_lut, parent_map : dict or None
    logger : Logger

    Returns
    -------
    dataset : RamLAIDataset
    loader : DataLoader

    Raises
    ------
    RuntimeError
        If the dataset is empty.
    """
    logger.start_task(
        "Building dataset",
        f"{len(site_ids):,} sites × {len(years)} years",
    )
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
        normalize_lai=bool(train_args.get("normalize_lai", True)),
        verbose=True,
        parent_map=parent_map,
    )
    if len(dataset) == 0:
        raise RuntimeError("No prediction samples produced.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    logger.success(f"  {len(dataset):,} samples to predict")
    return dataset, loader


# ── Phase 11 — inference ────────────────────────────────────────────────────


def _denormalise_row(
    pred_v,
    tgt_v,
    meta,
    is_anomaly,
    denorm_lai,
    anomaly_clim,
    norm_stats,
    lai_mean,
    lai_std,
):
    """
    Return the ``(pred_real, tgt_real, pred_norm, tgt_norm)`` tuple for one row.

    Handles the three flavours of training target: anomaly, z-scored LAI,
    raw LAI.

    Parameters
    ----------
    pred_v, tgt_v : np.ndarray
        Predicted and observed values on the model's native scale.
    meta : dict
        Sample metadata (from ``dataset.get_site_info``).
    is_anomaly, denorm_lai : bool
    anomaly_clim : dict or None
    norm_stats : dict or None
    lai_mean, lai_std : float
        LAI z-scoring statistics, used only when ``denorm_lai`` is True.

    Returns
    -------
    pred_real, tgt_real, pred_norm, tgt_norm : tuple of np.ndarray

    Raises
    ------
    RuntimeError
        If anomaly mode is on and the sample's site is missing from the
        climatology lookup.
    """
    if is_anomaly:
        clim_vec = anomaly_clim.get(meta["site_id"]) if anomaly_clim else None
        if clim_vec is None:
            raise RuntimeError(
                f"Anomaly mode is on but {meta['site_id']!r} has no climatology."
            )
        return pred_v + clim_vec, tgt_v + clim_vec, pred_v, tgt_v
    if denorm_lai:
        return (
            pred_v * lai_std + lai_mean,
            tgt_v * lai_std + lai_mean,
            pred_v,
            tgt_v,
        )
    return pred_v, tgt_v, pred_v, tgt_v


def run_inference(
    model,
    loader,
    dataset,
    is_daily,
    is_anomaly,
    denorm_lai,
    anomaly_clim,
    norm_stats,
    device,
    logger: Logger,
) -> pd.DataFrame:
    """
    Run the forward pass and collect one row per (site, year, dekad).

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    dataset : RamLAIDataset
    is_daily : bool
        The checkpoint was trained with ``--daily_lai`` — the model outputs
        365 values, sample the 36 dekad positions.
    is_anomaly : bool
    denorm_lai : bool
        Whether to invert the LAI z-scoring on the way out.
    anomaly_clim : dict or None
    norm_stats : dict or None
    device : torch.device
    logger : Logger

    Returns
    -------
    pandas.DataFrame
        Columns: ``site_id, year, month, day, doy, lai_pred, lai_obs,
        lai_pred_norm, lai_obs_norm, error``.
    """
    obs_pos = torch.tensor(_OBS_POSITIONS, dtype=torch.long)
    if is_daily:
        logger.info("Daily model   : sampling the 365-day output at the 36 dekads")

    if denorm_lai:
        lai_mean = float(norm_stats["LAI"]["mean"])
        lai_std = float(norm_stats["LAI"]["std"])

    logger.start_task("Running inference", f"{len(dataset):,} samples")
    rows = []
    with torch.no_grad():
        for i_batch, (features, targets) in enumerate(loader):
            preds = model(features.to(device)).cpu()
            if is_daily:
                preds = preds[:, :, obs_pos]
            batch_start = i_batch * loader.batch_size
            for j in range(preds.size(0)):
                idx = batch_start + j
                meta = dataset.get_site_info(idx)
                pred_v = preds[j, 0, :].numpy()
                tgt_v = targets[j, 0, :].numpy()
                pred_real, tgt_real, pred_norm, tgt_norm = _denormalise_row(
                    pred_v,
                    tgt_v,
                    meta,
                    is_anomaly,
                    denorm_lai,
                    anomaly_clim,
                    norm_stats,
                    lai_mean if denorm_lai else 0.0,
                    lai_std if denorm_lai else 1.0,
                )
                for k, (month, day, doy) in enumerate(_OBS_DATES):
                    rows.append(
                        {
                            "site_id": meta["site_id"],
                            "year": meta["year"],
                            "month": month,
                            "day": day,
                            "doy": doy,
                            "lai_pred": float(pred_real[k]),
                            "lai_obs": float(tgt_real[k]),
                            "lai_pred_norm": float(pred_norm[k]),
                            "lai_obs_norm": float(tgt_norm[k]),
                        }
                    )
            if (i_batch + 1) % max(1, len(loader) // 10) == 0:
                logger.info(f"  batch {i_batch + 1}/{len(loader)}")

    df = pd.DataFrame(rows)
    df["error"] = df["lai_pred"] - df["lai_obs"]
    logger.success(f"  {len(df):,} predictions collected")
    return df


# ── Phase 12 — metrics ──────────────────────────────────────────────────────


def _nse(o, p, nmin: int = 5) -> Optional[float]:
    """
    Nash–Sutcliffe efficiency, or ``None`` if it is ill-defined.

    Parameters
    ----------
    o, p : array-like
    nmin : int, optional

    Returns
    -------
    float or None
    """
    o = np.asarray(o, float)
    p = np.asarray(p, float)
    sstot = float(np.sum((o - o.mean()) ** 2))
    if len(o) < nmin or sstot <= 0:
        return None
    return 1.0 - float(np.sum((p - o) ** 2)) / sstot


def compute_metrics(
    df: pd.DataFrame,
    logger: Logger,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Compute the pooled and per-site metrics.

    Parameters
    ----------
    df : pandas.DataFrame
    logger : Logger

    Returns
    -------
    summary_lines : list of str
    valid : pandas.DataFrame
    """
    valid = df.dropna(subset=["lai_obs"]).copy()
    summary: List[str] = []
    if valid.empty:
        logger.warning("No valid observations to compute metrics on.")
        return summary, valid

    obs = valid["lai_obs"].values.astype(float)
    pred = valid["lai_pred"].values.astype(float)
    err = pred - obs
    overall_rmse = float(np.sqrt(np.mean(err**2)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    oc = valid["lai_obs"] - valid.groupby("site_id")["lai_obs"].transform("mean")
    pc = valid["lai_pred"] - valid.groupby("site_id")["lai_pred"].transform("mean")
    ss_res_c = float(np.sum((oc - pc) ** 2))
    ss_tot_c = float(np.sum(oc**2))
    r2_centered = 1.0 - ss_res_c / ss_tot_c if ss_tot_c > 0 else float("nan")

    if obs.std() > 1e-9 and pred.std() > 1e-9:
        r = float(np.corrcoef(obs, pred)[0, 1])
        r2_pearson = r * r
    else:
        r2_pearson = float("nan")

    site_r2s = []
    for _, g in valid.groupby("site_id"):
        o = g["lai_obs"].values
        p = g["lai_pred"].values
        sstot = float(np.sum((o - o.mean()) ** 2))
        if len(o) < 5 or sstot <= 0:
            continue
        site_r2s.append(1.0 - float(np.sum((p - o) ** 2)) / sstot)

    siteyear_r2 = [
        r
        for _, g in valid.groupby(["site_id", "year"])
        if (r := _nse(g["lai_obs"], g["lai_pred"])) is not None
    ]
    interann_r2 = [
        r
        for _, g in valid.groupby("doy")
        if (r := _nse(g["lai_obs"], g["lai_pred"])) is not None
    ]

    summary.append("── Pooled metrics ──")
    summary.append(
        f"  Global    R²  (NSE)         : {r2_global:+.4f}  "
        f"(includes inter-site mean differences — generous)"
    )
    summary.append(
        f"  Centered  R²  (NSE on dyn.) : {r2_centered:+.4f}  "
        f"(intra-site variance only — fair test of dynamics)"
    )
    summary.append(
        f"  Pearson²  R²  (correlation) : {r2_pearson:+.4f}  "
        f"(linear alignment only — ignores bias / scale; matches papers)"
    )
    summary.append(f"  Overall RMSE                : {overall_rmse:.4f}")

    if "lai_clim" in valid.columns and valid["lai_clim"].notna().any():
        an = M.evaluate_predictions(
            valid,
            obs_col="lai_obs",
            pred_col="lai_pred",
            clim_col="lai_clim",
            n_boot=0,
        ).get("anomaly")
        if an:
            summary.append("")
            summary.append("── Anomaly metrics (vs training-year climatology) ──")
            summary.append(f"  Anomaly   R²                : {an['overall_r2']:+.4f}")
            summary.append(f"  Anomaly RMSE                : {an['rmse']:.4f}")
            summary.append(
                f"  Sites beating climatology   : "
                f"{an['per_site']['pct_gt0']:.1f}%"
            )

    if site_r2s:
        arr = np.array(site_r2s)
        n_pos = int(np.sum(arr > 0))
        summary.append("")
        summary.append(f"── Per-site R² (NSE) distribution ({len(arr)} sites) ──")
        summary.append(f"  Median         : {np.median(arr):+.4f}")
        summary.append(f"  Mean ± std     : {np.mean(arr):+.4f} ± {np.std(arr):.4f}")
        summary.append(f"  5th  percentile: {np.percentile(arr, 5):+.4f}")
        summary.append(f"  95th percentile: {np.percentile(arr, 95):+.4f}")
        summary.append(
            f"  Sites with R²>0: {n_pos:,} / {len(arr):,} "
            f"({100.0 * n_pos / len(arr):.1f}%)"
        )

    sy = np.array(siteyear_r2, dtype=float)
    ia = np.array(interann_r2, dtype=float)
    r2_items = [
        ("Overall      R²", r2_global, None, ""),
        (
            "Site-year    R²",
            float(sy.mean()) if sy.size else float("nan"),
            float(sy.std()) if sy.size else None,
            f" (n={sy.size} site-years)",
        ),
        (
            "Interannual  R²",
            float(ia.mean()) if ia.size else float("nan"),
            float(ia.std()) if ia.size else None,
            f" (n={ia.size} DOYs)",
        ),
    ]
    finite = [m for _, m, _, _ in r2_items if np.isfinite(m)]
    best = max(finite) if finite else None
    summary.append("")
    summary.append("── R² decomposition (overall / site-year / interannual) ──")
    for label, m, s, note in r2_items:
        txt = f"{m:+.4f}" if s is None else f"{m:+.4f} ± {s:.4f}"
        if best is not None and np.isfinite(m) and m == best:
            txt = f"**{txt}**"
        summary.append(f"  {label} : {txt}{note}")

    logger.info(f"Metrics       : {len(valid):,} valid / {len(df):,} total rows")
    return summary, valid


# ── Phase 13 — outputs ──────────────────────────────────────────────────────


def _load_pft_fracs(pft_dir: str, year: int, site_ids) -> Dict:
    """
    Return ``{site_id: (N_PFT,) fraction vector}`` from ``PFTmap_{year}.nc``.

    Empty dict if the file is missing or the site is absent.

    Parameters
    ----------
    pft_dir : str
    year : int
    site_ids : iterable of str

    Returns
    -------
    dict
    """
    import xarray as xr

    path = os.path.join(pft_dir, PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        return {}
    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
    da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
    all_sites = np.asarray(ds["site_id"].values).astype(str)
    arr = da.transpose("pft", "site").values.astype(np.float32)
    ds.close()
    idx_of = {s: i for i, s in enumerate(all_sites)}
    out = {}
    for s in map(str, site_ids):
        j = idx_of.get(s, -1)
        if j >= 0:
            out[s] = arr[:, j]
    return out


def _write_scatter_years(valid, base: str, logger: Logger) -> None:
    """
    Write one pred-vs-obs scatter per evaluated year.

    Parameters
    ----------
    valid : pandas.DataFrame
    base : str
    logger : Logger
    """
    year_dir = os.path.join(os.path.dirname(base) or ".", "scatter_year")
    os.makedirs(year_dir, exist_ok=True)
    for yr in sorted(int(y) for y in valid["year"].unique()):
        sub = valid[valid["year"] == yr]
        if len(sub) < 2:
            continue
        ss_res = float(np.sum((sub["lai_obs"] - sub["lai_pred"]) ** 2))
        ss_tot = float(np.sum((sub["lai_obs"] - sub["lai_obs"].mean()) ** 2))
        r2_y = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse_y = float(np.sqrt(np.mean((sub["lai_pred"] - sub["lai_obs"]) ** 2)))
        plot_pred_vs_obs(
            sub["lai_pred"].values,
            sub["lai_obs"].values,
            filename=os.path.join(year_dir, f"year_{yr}_pred_vs_obs.png"),
            title=f"Year {yr} — R²={r2_y:+.4f} RMSE={rmse_y:.4f} " f"n={len(sub):,}",
            logger=logger,
        )


def write_outputs(
    df,
    summary,
    valid,
    args,
    pft_dir: str,
    years: List[int],
    logger: Logger,
    checkpoint_meta: Optional[Dict] = None,
    run_metadata: Optional[Dict] = None,
) -> None:
    """
    Write the CSV, the metrics summary, and the diagnostic plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Full predictions table.
    summary : list of str
        Lines produced by :func:`compute_metrics`.
    valid : pandas.DataFrame
        Rows with finite observations.
    args : argparse.Namespace
    pft_dir : str
    years : list of int
    logger : Logger
    checkpoint_meta : dict, optional
        Epoch / validation metrics read from the checkpoint.
    run_metadata : dict, optional
        Output of :func:`collect_run_metadata`.
    """
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions → {args.output_csv}")

    base = os.path.splitext(args.output_csv)[0]
    if summary:
        metrics_path = base + "_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("\n".join(summary) + "\n")
        logger.info(f"Metrics → {metrics_path}")

    if valid.empty:
        return

    # ── Structured paper bundle ──
    bundle_dir = args.bundle_dir or os.path.dirname(args.output_csv) or "."
    try:
        B.write_prediction_bundle(
            df,
            valid,
            bundle_dir,
            predictions_file=args.output_csv,
            n_boot=int(getattr(args, "n_bootstrap", 1000)),
            seed=int(getattr(args, "seed", 42)),
            checkpoint_meta=checkpoint_meta,
            run_metadata=run_metadata,
            write_phenology=not bool(getattr(args, "no_phenology_metrics", False)),
            pheno_max_siteyears=int(getattr(args, "pheno_max_siteyears", 0)),
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - defensive, keep the CSV
        logger.warning(f"Bundle writer failed ({exc}); predictions were kept.")

    plot_pred_vs_obs(
        valid["lai_pred"].values,
        valid["lai_obs"].values,
        filename=base + "_pred_vs_obs.png",
        title=f"PhenoNN — Predicted vs observed LAI ({len(valid):,} pts)",
        logger=logger,
    )
    if bool(getattr(args, "_is_anomaly", False)):
        plot_pred_vs_obs(
            valid["lai_pred_norm"].values,
            valid["lai_obs_norm"].values,
            filename=base + "_pred_vs_obs_anomaly.png",
            title=f"ΔLAI anomaly ({len(valid):,} pts)",
            xlabel="Observed ΔLAI",
            ylabel="Predicted ΔLAI",
            logger=logger,
        )
    elif bool(getattr(args, "_denorm_lai", False)):
        plot_pred_vs_obs(
            valid["lai_pred_norm"].values,
            valid["lai_obs_norm"].values,
            filename=base + "_pred_vs_obs_norm.png",
            title=f"Normalized LAI ({len(valid):,} pts)",
            xlabel="Observed LAI (normalized)",
            ylabel="Predicted LAI (normalized)",
            logger=logger,
        )

    if valid["site_id"].nunique() >= 3:
        plot_gcc_curves(
            valid,
            filename=base + "_lai_curves.png",
            site_col="site_id",
            year_col="year",
            doy_col="doy",
            logger=logger,
        )

    sites_all = valid["site_id"].unique()
    if args.n_curves and 0 < args.n_curves < len(sites_all):
        rng_c = np.random.RandomState(args.seed)
        keep = set(rng_c.choice(sites_all, size=args.n_curves, replace=False))
        curve_df = valid[valid["site_id"].isin(keep)]
    else:
        curve_df = valid
    n_curve_sites = curve_df["site_id"].nunique()
    if n_curve_sites > 400:
        logger.warning(
            f"plotting {n_curve_sites:,} site curves in one grid — "
            f"use --n_curves to subsample."
        )
    pft_by_site = _load_pft_fracs(pft_dir, years[0], curve_df["site_id"].unique())
    plot_gcc_curves_all(
        curve_df,
        filename=base + "_lai_curves_all.png",
        site_col="site_id",
        year_col="year",
        doy_col="doy",
        pft_by_site=pft_by_site,
        pft_names=PFT_NAMES,
        pft_min_frac=args.pft_min_frac,
        logger=logger,
    )

    if args.scatter_years:
        _write_scatter_years(valid, base, logger)


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    """
    Orchestrate the inference pipeline.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Logger: sits next to the checkpoint ──
    ckpt_dir = os.path.dirname(args.checkpoint) or "."
    log_file = os.path.join(ckpt_dir, "predict.log")
    logger = Logger(
        console_output=True,
        file_output=True,
        log_file=log_file,
    )
    logger.show_header("PhenoNN inference")

    # ── Checkpoint ──
    ckpt = load_checkpoint(args, device, logger)
    train_args = ckpt["args"]
    log_configuration(ckpt, args, logger)

    # ── Extract the flags that drive the rest ──
    norm_stats = ckpt["norm_stats"]
    co2_lut = ckpt["co2_lut"]
    anomaly_clim = ckpt["anomaly_clim"]
    is_anomaly = ckpt["anomaly_mode"]
    lai_normalized = ckpt["normalize_lai"]
    denorm_lai = norm_stats is not None and lai_normalized and not is_anomaly
    args._is_anomaly = is_anomaly
    args._denorm_lai = denorm_lai

    # ── Data directories and parent map ──
    features_dir, target_dir, pft_dir = resolve_data_directories(args, train_args)
    parent_map = load_parent_map_if_requested(args, train_args, logger)

    # ── Model ──
    model, _ = initialize_model(ckpt, train_args, device, logger)
    install_ablation_hook(model, args, train_args, logger)

    # ── Sites and years ──
    site_ids = _resolve_sites(args, ckpt, train_args)
    if not site_ids:
        raise RuntimeError("Empty site list.")
    if 0 < args.n_predict_sites < len(site_ids):
        rng = np.random.RandomState(args.seed)
        site_ids = rng.choice(
            site_ids, size=args.n_predict_sites, replace=False
        ).tolist()
    logger.info(f"Sites         : {len(site_ids):,}")

    years = resolve_years(args, train_args)
    logger.info(f"Years         : {years}")

    # ── Dataset and loader ──
    dataset, loader = create_dataset_and_loader(
        args,
        features_dir,
        target_dir,
        pft_dir,
        years,
        site_ids,
        train_args,
        norm_stats,
        anomaly_clim,
        co2_lut,
        parent_map,
        logger,
    )

    # ── Inference ──
    df = run_inference(
        model,
        loader,
        dataset,
        ckpt["daily_lai"],
        is_anomaly,
        denorm_lai,
        anomaly_clim,
        norm_stats,
        device,
        logger,
    )

    # ── Climatology, PFT metadata and structured bundle ──
    clim_by_site, clim_years = resolve_climatology(
        args, ckpt, train_args, target_dir, site_ids, years, logger
    )
    if clim_by_site:
        B.attach_climatology(df, clim_by_site)
    if not args.no_pft_columns:
        B.attach_pft_columns(df, pft_dir, PFT_NAMES)

    checkpoint_meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "val_rmse": ckpt.get("val_rmse"),
        "val_r2": ckpt.get("val_r2"),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "best_r2_epoch": ckpt.get("best_r2_epoch"),
        "selection_rule": "lowest validation loss",
        "pft_mixing": ckpt.get("pft_mixing"),
        "anomaly_mode": is_anomaly,
        "clim_years": clim_years,
        "protocol": args.protocol,
    }
    data_paths = [
        args.checkpoint,
        train_args.get("stats_path", ""),
        train_args.get("co2_path", ""),
        train_args.get("parent_map", ""),
        args.selected_pixels,
    ]
    run_metadata = collect_run_metadata(
        args,
        data_paths=[p for p in data_paths if p],
        extra={
            "stage": "predict",
            "clim_years": clim_years,
            "checkpoint": checkpoint_meta,
        },
    )

    # ── Metrics ──
    summary, valid = compute_metrics(df, logger)
    if summary:
        for line in summary:
            logger.info(line)

    # ── Outputs ──
    write_outputs(
        df,
        summary,
        valid,
        args,
        pft_dir,
        years,
        logger,
        checkpoint_meta=checkpoint_meta,
        run_metadata=run_metadata,
    )

    logger.success("Inference complete.")


if __name__ == "__main__":
    main()
