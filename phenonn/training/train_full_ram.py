#!/usr/bin/env python3
# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN — RAM-resident training entry point (method C2, "working-set").

Standalone trainer: the per-epoch disk re-reads are removed — the raw daily
features (+ targets / PFT / climatology) for every site the training loop will
*ever* sample are loaded into RAM **once** at start-up. Each epoch then only
sub-selects sample indices already in memory — zero per-epoch I/O.

Why "working-set" and not the whole pool
----------------------------------------
The per-epoch sampler only ever touches ``n_sites_per_epoch`` sites per epoch,
drawn with ``RandomState(seed + epoch)``. Over the whole run the set of
*distinct* sites used is therefore the union of those draws over all epochs,
which we compute exactly by replaying the sampler's RNG here. Only those
sites are loaded.

Module overview
---------------
The training pipeline is organised as a chain of small functions, each
responsible for one phase of the run:

1.  ``parse_args``                       — CLI
2.  ``setup_directories_and_logging``    — mkdir + Logger, returns an EasyDict
3.  ``log_configuration``                — dump args
4.  ``setup_device_and_seed``            — torch.device + seeds
5.  ``load_norm_stats``                  — optional JSON
6.  ``load_co2_lut_if_requested``        — optional CO2 LUT
7.  ``load_parent_map_if_requested``     — optional 0.05°→0.1° map
8.  ``build_site_pools``                 — train / val site pools
9.  ``compute_working_set``              — sites ever sampled
10. ``compute_anomaly_climatology``      — optional per-site climatology
11. ``create_datasets_and_loaders``      — RAM datasets + DataLoaders
12. ``initialize_model``                 — build_model / build_model_pft
13. ``create_optimizer_and_scheduler``   — Adam + ReduceLROnPlateau
14. ``create_loss_function``             — nan_safe_loss closure
15. ``load_checkpoint_if_requested``     — resume via ModelUtils
16. ``run_training``                     — the loop
17. ``generate_summary_plots``           — loss / metric figures

Checkpoint writing delegates to :class:`phenonn.utils.model_utils.ModelUtils`,
so the on-disk format is shared with the rest of the project.

The same primitives are reused by :func:`load_shared` and
:func:`train_one_config`, which the in-process sweep drivers call to pay
the disk-load cost only once per process.

Usage
-----
::

    python -m phenonn.training.train_full_ram \
        --features_dir /data/.../era5_pixelset \
        --target_dir   /data/.../LAI_pixelset \
        --pft_dir      /data/.../PFT_pixelset \
        --selected_pixels /data/.../selected_pixels005_1.nc \
        --parent_map   /data/.../selected_pixels01_1.nc \
        --train_years 1992-2010 --val_years 2011-2018 \
        --num_epochs 30 --n_sites_per_epoch 1000 --n_years_per_epoch 5 ...
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from phenonn.utils.config import FEATURE_CHANNELS, PFT_START
from phenonn.data.lai_dataset import (
    RamLAIDataset,
    compute_climatology_lookup,
    list_valid_sites,
    load_co2_lut,
    load_parent01_map,
    load_selected_pixels,
)
from phenonn.utils.evaluater import make_loss_fn
from phenonn.utils.model_loader import build_model, build_model_pft
from phenonn.utils.model_utils import ModelUtils
from phenonn.utils.diagnostics import (
    make_history_dicts,
    plot_loss_histories,
    plot_metric_histories,
)
from phenonn.utils.logger import Logger
from phenonn.utils.utils import EasyDict, FileUtils
from phenonn.evaluation.leakage import (
    validate_climatology_years,
    validate_normalisation_years,
    validate_split,
)
from phenonn.evaluation.reproducibility import collect_run_metadata, write_json


# ── Shared container ────────────────────────────────────────────────────────


@dataclass
class SharedData:
    """
    Everything a training run reuses across hyperparameter configs.

    Loaded from disk into RAM once by :func:`load_shared`: the site pools,
    normalisation tables, and the resident validation and training datasets.
    Identical for every config whose ``subset`` / ``seed`` / years /
    ``n_sites_per_epoch`` / ``num_epochs`` match.
    """

    device: object
    train_pool: list
    val_sites: list
    train_years: list
    val_years: list
    norm_stats: object
    co2_lut: object
    anomaly_clim: object
    val_ds: object
    val_loader: object
    train_full: object


# ── Small helpers ───────────────────────────────────────────────────────────


def parse_year_list(s: str) -> list:
    """
    Parse a year specification into a list of ints.

    Parameters
    ----------
    s : str
        Either ``"1992-2010"`` (range), ``"1992,2000,2005"`` (list), or an
        empty string.

    Returns
    -------
    list of int
        The parsed years, or an empty list for an empty input.
    """
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _cli_explicit_args() -> set:
    """
    Names of the arguments the user actually wrote on the command line.

    Used when resuming, to let CLI overrides win over the checkpoint's
    stored arguments.

    Returns
    -------
    set of str
        Argument names, with ``-`` replaced by ``_``.
    """
    explicit = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            explicit.add(arg[2:].split("=")[0].replace("-", "_"))
    return explicit


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    """
    Parse the command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── Data ──
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument("--valid_dir", default="")
    p.add_argument("--selected_pixels", default="")
    p.add_argument("--parent_map", default="")
    p.add_argument("--stats_path", default="")
    p.add_argument("--co2_path", default="")
    p.add_argument(
        "--no_normalize_lai",
        dest="normalize_lai",
        action="store_false",
        default=True,
    )
    p.add_argument("--daily_lai", action="store_true", default=False)
    p.add_argument("--daily_target_dir", default="")
    p.add_argument(
        "--threaded_feature_read",
        action="store_true",
        default=False,
    )

    # ── Sub-grid ──
    p.add_argument("--row_min", type=int, default=0)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=0)
    p.add_argument("--col_max", type=int, default=-1)

    # ── Years ──
    p.add_argument("--train_years", default="")
    p.add_argument("--val_years", default="")
    p.add_argument(
        "--test_years",
        default="",
        help="Held-out test years, e.g. '2010-2019'. Required by protocol B "
        "and used only for leakage checks; never for early stopping.",
    )
    p.add_argument(
        "--protocol",
        default="none",
        choices=["A", "B", "none"],
        help="Split protocol: A = reproduction (val=test decade), "
        "B = paper split, none = ad-hoc (leakage checks still run).",
    )
    p.add_argument(
        "--norm_years",
        default="",
        help="Years used to build --stats_path. Checked against train_years.",
    )
    p.add_argument(
        "--allow_leakage",
        action="store_true",
        help="Downgrade leakage errors to warnings (documented diagnostics only).",
    )
    p.add_argument(
        "--selection_rule",
        default="val_loss",
        choices=["val_loss", "val_r2"],
        help="Checkpoint selection criterion (paper: val_loss).",
    )

    # ── Per-epoch sampling ──
    p.add_argument("--n_sites_per_epoch", type=int, default=500)
    p.add_argument("--n_years_per_epoch", type=int, default=3)
    p.add_argument("--n_val_sites", type=int, default=200)
    p.add_argument("--val_fraction_of_grid", type=float, default=0.1)
    p.add_argument("--subset", type=float, default=1.0)

    # ── Anomaly mode ──
    p.add_argument("--anomaly_mode", action="store_true")
    p.add_argument("--clim_years", default="")
    p.add_argument("--clim_target_dir", default="")

    # ── Model ──
    p.add_argument(
        "--type",
        default="lstm",
        choices=[
            "lstm",
            "gru",
            "transformer",
            "transformer_dec",
            "bitransformer",
            "bitransformer_v2",
            "attnlstm",
            "aelstm",
            "fcn",
            "linear",
            "linear_perday",
        ],
    )
    p.add_argument("--pft_mixing", action="store_true")
    p.add_argument("--pft_meteo_only", action="store_true")
    p.add_argument("--pft_nonneg", action="store_true")
    p.add_argument("--seq_length", type=int, default=730)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_layers1", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout2", type=float, default=0.0)
    p.add_argument("--dropout1", type=float, default=0.0)
    p.add_argument("--dropout_att", type=float, default=0.0)
    p.add_argument("--feed_forward_trans1", type=int, default=4)
    p.add_argument("--feed_forward_trans2", type=int, default=4)
    p.add_argument("--n_attn_blocks", type=int, default=2)
    p.add_argument("--stress_dim", type=int, default=8)
    p.add_argument("--d_layers", type=int, default=2)
    p.add_argument("--embed_size", type=int, default=64)

    # ── Training ──
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument(
        "--loss_type",
        default="huber",
        choices=["mse", "mae", "huber", "smoothl1", "nmse", "nmae", "gradient"],
    )
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--gradient_loss_weight", type=float, default=0.5)
    p.add_argument("--peak_penalty_weight", type=float, default=0.0)
    p.add_argument("--corr_loss_weight", type=float, default=0.0)
    p.add_argument("--amp_loss_weight", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=11)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--no_fused_adam",
        dest="fused_adam",
        action="store_false",
        default=True,
    )

    # ── Output / resume ──
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--experiment", default="exp_big")
    p.add_argument("--prefix", default="run", help="Prefix for checkpoint filenames.")
    p.add_argument(
        "--save_checkpoint_name",
        default="model",
        help="Base name for checkpoint files.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")

    return p.parse_args(argv)


# ── Phase 2 — directories and logger ────────────────────────────────────────


def setup_directories_and_logging(args) -> tuple:
    """
    Create the experiment directory tree and return a configured Logger.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``output_dir`` and ``experiment``.

    Returns
    -------
    paths : EasyDict
        Dictionary with attribute access to the created directories:
        ``exp``, ``checkpoints``, ``logs``.
    logger : Logger
        Configured logger writing to ``<logs>/train.log``.
    """
    exp_dir = os.path.join(args.output_dir, args.experiment)

    paths = EasyDict()
    paths.exp = exp_dir
    paths.checkpoints = os.path.join(exp_dir, "checkpoints")
    paths.logs = os.path.join(exp_dir, "logs")

    for p in (paths.exp, paths.checkpoints, paths.logs):
        FileUtils.makedir(p)

    logger = Logger(
        console_output=True,
        file_output=True,
        log_file=os.path.join(paths.logs, "train.log"),
    )
    logger.show_header("PhenoNN training (RAM / method C2 working-set)")
    return paths, logger


# ── Phase 3 — configuration dump ────────────────────────────────────────────


def log_configuration(args, logger) -> None:
    """
    Log the full configuration once, before the run starts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    logger : Logger
        Logger instance.
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Experiment      : {args.experiment}")
    logger.info(f"Output dir      : {args.output_dir}")
    logger.info(f"Model type      : {args.type}")
    logger.info(f"seq_length      : {args.seq_length}")
    logger.info(f"PFT mixing      : {args.pft_mixing}")
    logger.info(f"PFT meteo only  : {args.pft_meteo_only}")
    logger.info(f"Loss            : {args.loss_type}")
    logger.info(f"Train years     : {args.train_years}")
    logger.info(f"Val years       : {args.val_years}")
    logger.info(f"Batch size      : {args.batch_size}")
    logger.info(f"Num epochs      : {args.num_epochs}")
    logger.info(f"Learning rate   : {args.learning_rate}")
    logger.info(f"Seed            : {args.seed}")
    logger.info("=" * 60)


# ── Phase 4 — device and seeds ──────────────────────────────────────────────


def setup_device_and_seed(args, logger) -> torch.device:
    """
    Set the random seeds and pick the compute device.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``seed``.
    logger : Logger
        Logger instance.

    Returns
    -------
    torch.device
        The device every model and batch will be moved to.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device          : {device}")
    logger.info(f"Random seed     : {args.seed}")

    # Fixed input shapes → let cuDNN pick the best kernels.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


# ── Phase 5-7 — optional auxiliary tables ───────────────────────────────────


def load_norm_stats(args, logger):
    """
    Load the per-feature normalisation statistics, if requested.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``stats_path``.
    logger : Logger
        Logger instance.

    Returns
    -------
    dict or None
        The parsed statistics, or ``None`` when the path is empty.

    Raises
    ------
    FileNotFoundError
        If ``stats_path`` is set but the file does not exist.
    """
    if not args.stats_path:
        logger.info("Norm stats      : OFF (raw scales)")
        return None
    if not os.path.exists(args.stats_path):
        raise FileNotFoundError(args.stats_path)
    with open(args.stats_path) as f:
        norm_stats = json.load(f)
    logger.info(f"Norm stats      : {args.stats_path}  ({len(norm_stats)} entries)")
    return norm_stats


def load_co2_lut_if_requested(args, norm_stats, logger):
    """
    Load the CO2 lookup table, if requested.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``co2_path``.
    norm_stats : dict or None
        Feature statistics. When not ``None``, the LUT is checked for a
        matching ``co2`` entry.
    logger : Logger
        Logger instance.

    Returns
    -------
    dict or None
        Per-year CO2 concentrations, or ``None``.

    Raises
    ------
    FileNotFoundError
        If ``co2_path`` is set but the file is missing.
    RuntimeError
        If statistics are provided but do not contain a ``co2`` entry.
    """
    if not args.co2_path:
        return None
    if not os.path.exists(args.co2_path):
        raise FileNotFoundError(args.co2_path)
    co2_lut = load_co2_lut(args.co2_path)
    logger.info(f"CO2 LUT         : {args.co2_path}  ({len(co2_lut)} years)")
    if norm_stats is not None and "co2" not in norm_stats:
        raise RuntimeError(
            "stats_path is set but contains no 'co2' entry while CO2 is "
            "enabled. Regenerate the stats file with the CO2 LUT."
        )
    return co2_lut


def load_parent_map_if_requested(args, logger):
    """
    Load the optional 0.05° → 0.1° parent map.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``parent_map``.
    logger : Logger
        Logger instance.

    Returns
    -------
    dict or None
        Mapping from 0.05° site id to 0.1° cell id, or ``None``.
    """
    if not getattr(args, "parent_map", ""):
        logger.info("Parent map      : OFF (features indexed on the site pool)")
        return None
    if not os.path.exists(args.parent_map):
        raise FileNotFoundError(args.parent_map)
    parent_map = load_parent01_map(args.parent_map)
    logger.info(
        f"Parent map      : {args.parent_map}  "
        f"({len(parent_map):,} 0.05°→0.1° links)"
    )
    return parent_map


# ── Phase 8 — site pools ────────────────────────────────────────────────────


def build_site_pools(args, all_years, logger) -> tuple:
    """
    Build the train and validation site pools.

    Two sources are supported, in this order:

    - ``--selected_pixels``: a pre-sampled subset.
    - ``--valid_dir``: the union over all years of ``valid_pixels_{Y}.nc``
      masks, optionally restricted to the row/col bbox.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    all_years : list of int
        Union of train and validation years.
    logger : Logger
        Logger instance.

    Returns
    -------
    train_pool : list of str
    val_pool : list of str

    Raises
    ------
    ValueError
        If neither ``--selected_pixels`` nor ``--valid_dir`` is provided, or
        if ``--subset`` is out of range.
    RuntimeError
        If ``--valid_dir`` is used but no valid pixel is found.
    """
    if args.selected_pixels:
        all_sites = sorted(load_selected_pixels(args.selected_pixels))
        logger.info(
            f"Site pool       : {len(all_sites):,} sites from "
            f"{os.path.basename(args.selected_pixels)}"
        )
    elif args.valid_dir:
        bbox_rows = (args.row_min, args.row_max) if args.row_max >= 0 else None
        bbox_cols = (args.col_min, args.col_max) if args.col_max >= 0 else None
        valid_per_year = list_valid_sites(
            args.valid_dir, all_years, bbox_rows, bbox_cols
        )
        all_sites = sorted(set().union(*valid_per_year.values()))
        if not all_sites:
            raise RuntimeError(
                "No valid pixels found. Run compute_valid_pixels.py first, "
                "and check --valid_dir + --row/col bounds."
            )
        logger.info(
            f"Site pool       : {len(all_sites):,} unique valid pixels "
            f"(across {len(all_years)} years)"
        )
    else:
        raise ValueError(
            "Provide either --selected_pixels (recommended) or --valid_dir."
        )

    if not (0.0 < args.subset <= 1.0):
        raise ValueError(f"--subset must be in (0, 1], got {args.subset}")
    if args.subset < 1.0:
        n_keep = max(1, int(len(all_sites) * args.subset))
        sub_rng = np.random.RandomState(args.seed)
        all_sites = sorted(
            sub_rng.choice(all_sites, size=n_keep, replace=False).tolist()
        )
        logger.info(
            f"Subset          : {n_keep:,} sites kept "
            f"({args.subset:.3g}× pool) for train+val"
        )

    rng = np.random.RandomState(args.seed)
    if args.val_fraction_of_grid >= 100:
        logger.info(
            "Mode            : OVERLAP — train and val from the same pool "
            "(years must be disjoint)"
        )
        return list(all_sites), list(all_sites)

    shuffled = rng.permutation(all_sites).tolist()
    n_val_pool = max(args.n_val_sites, int(args.val_fraction_of_grid * len(shuffled)))
    val_pool = shuffled[:n_val_pool]
    train_pool = shuffled[n_val_pool:]
    logger.info(
        f"Mode            : DISJOINT — train={len(train_pool):,} "
        f"val={len(val_pool):,}"
    )
    return train_pool, val_pool


# ── Phase 9 — working set ───────────────────────────────────────────────────


def compute_working_set(args, train_pool, train_years) -> list:
    """
    Union of the sites the per-epoch sampler will ever draw.

    Replays the exact RNG sequence used by the training loop so the loaded
    set matches what the loop samples. Independent of ``--resume``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``n_sites_per_epoch``, ``n_years_per_epoch``,
        ``seed``, ``num_epochs``.
    train_pool : list of str
    train_years : list of int

    Returns
    -------
    list of str
        Sorted site ids.
    """
    working: set = set()
    n_years = min(args.n_years_per_epoch, len(train_years))
    for epoch in range(1, args.num_epochs + 1):
        rng = np.random.RandomState(args.seed + epoch)
        n_sites = min(args.n_sites_per_epoch, len(train_pool))
        sites = rng.choice(train_pool, size=n_sites, replace=False)
        rng.choice(train_years, size=n_years, replace=False)  # keep stream aligned
        working.update(sites.tolist())
    return sorted(working)


# ── Phase 10 — anomaly climatology ──────────────────────────────────────────


def compute_anomaly_climatology(args, working_sites, val_sites, resume_ckpt, logger):
    """
    Build the per-site dekadal climatology when ``--anomaly_mode`` is on.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``anomaly_mode``, ``clim_years``,
        ``clim_target_dir``, ``target_dir``.
    working_sites : list of str
    val_sites : list of str
    resume_ckpt : dict or None
        A previously-loaded checkpoint; when it contains ``anomaly_clim``,
        that is reused instead of being recomputed.
    logger : Logger
        Logger instance.

    Returns
    -------
    dict or None
    """
    if not args.anomaly_mode:
        return None
    clim_years = parse_year_list(args.clim_years)
    clim_dir = args.clim_target_dir or args.target_dir
    if resume_ckpt is not None and resume_ckpt.get("anomaly_clim") is not None:
        anomaly_clim = resume_ckpt["anomaly_clim"]
        logger.info(
            f"Climatology     : reused from checkpoint "
            f"({len(anomaly_clim):,} sites)"
        )
        return anomaly_clim
    wanted = list(set(working_sites) | set(val_sites))
    logger.start_task("Climatology", f"{len(clim_years)} years × {len(wanted):,} sites")
    anomaly_clim = compute_climatology_lookup(
        target_dir=clim_dir,
        clim_years=clim_years,
        wanted_sites=wanted,
        verbose=False,
    )
    logger.success(f"  {len(anomaly_clim):,} site climatologies built")
    return anomaly_clim


# ── Phase 11 — datasets and loaders ─────────────────────────────────────────


def create_datasets_and_loaders(
    args,
    val_sites,
    working_sites,
    train_years,
    val_years,
    norm_stats,
    co2_lut,
    anomaly_clim,
    parent_map,
    logger,
) -> tuple:
    """
    Build the validation and training datasets and their DataLoaders.

    The validation dataset uses ``daily_mode='obs'`` in ``--daily_lai`` mode,
    so the loss and metrics score only the real dekads. The training dataset
    uses ``daily_mode='interp'`` in that mode, so the loss covers all days
    of the interpolated curve.

    Parameters
    ----------
    args : argparse.Namespace
    val_sites, working_sites : list of str
    train_years, val_years : list of int
    norm_stats, co2_lut, anomaly_clim, parent_map : dict or None
    logger : Logger

    Returns
    -------
    val_ds : RamLAIDataset
    val_loader : DataLoader
    train_full : RamLAIDataset
    """
    logger.start_task(
        "Loading validation set into RAM",
        f"{len(val_sites)} sites × {len(val_years)} years",
    )
    val_ds = RamLAIDataset(
        features_dir=args.features_dir,
        target_dir=args.target_dir,
        pft_dir=args.pft_dir,
        years=val_years,
        site_ids=val_sites,
        seq_length=args.seq_length,
        norm_stats=norm_stats,
        anomaly_clim=anomaly_clim,
        co2_lut=co2_lut,
        normalize_lai=args.normalize_lai,
        verbose=True,
        threaded_read=args.threaded_feature_read,
        parent_map=parent_map,
        daily_mode=("obs" if args.daily_lai else "off"),
    )
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check inputs.")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Val samples     : {len(val_ds):,}")

    logger.start_task(
        "Loading training working-set into RAM",
        f"{len(working_sites)} sites × {len(train_years)} years",
    )
    train_full = RamLAIDataset(
        features_dir=args.features_dir,
        target_dir=args.target_dir,
        pft_dir=args.pft_dir,
        years=train_years,
        site_ids=working_sites,
        seq_length=args.seq_length,
        norm_stats=norm_stats,
        anomaly_clim=anomaly_clim,
        co2_lut=co2_lut,
        normalize_lai=args.normalize_lai,
        verbose=True,
        threaded_read=args.threaded_feature_read,
        parent_map=parent_map,
        daily_mode=("interp" if args.daily_lai else "off"),
        daily_target_dir=args.daily_target_dir,
    )
    if len(train_full) == 0:
        raise RuntimeError("Training dataset is empty. Check inputs.")
    logger.info(f"Train samples   : {len(train_full):,} resident in RAM")

    return val_ds, val_loader, train_full


# ── Phase 12 — model ────────────────────────────────────────────────────────


def initialize_model(args, norm_stats, device, logger):
    """
    Build the model and move it to the device.

    Parameters
    ----------
    args : argparse.Namespace
    norm_stats : dict or None
    device : torch.device
    logger : Logger

    Returns
    -------
    model : nn.Module
    wrapper_label : str
        Human-readable name of the wrapper applied.
    """
    if args.pft_mixing:
        model = build_model_pft(args, norm_stats).to(device)
        wrapper_label = "PFTMixingWrapper"
    else:
        model = build_model(args).to(device)
        wrapper_label = "DailyWrapper" if args.daily_lai else "Every10DaysWrapper"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model           : {args.type} + {wrapper_label}  "
        f"({n_params:,} parameters)"
    )
    return model, wrapper_label


# ── Phase 13 — optimizer and scheduler ──────────────────────────────────────


def create_optimizer_and_scheduler(args, model, device, logger):
    """
    Build the Adam optimizer and the plateau scheduler.

    Parameters
    ----------
    args : argparse.Namespace
    model : nn.Module
    device : torch.device
    logger : Logger

    Returns
    -------
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=(args.fused_adam and device.type == "cuda"),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    logger.info(f"Optimizer       : Adam (lr={args.learning_rate})")
    return optimizer, scheduler


# ── Phase 14 — loss ─────────────────────────────────────────────────────────


def create_loss_function(args, logger):
    """
    Build the NaN-safe loss closure from the CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
    logger : Logger

    Returns
    -------
    callable
        Function ``(pred, target) -> tensor``.
    """
    criterion = make_loss_fn(args)
    logger.info(
        f"Loss            : {args.loss_type} "
        f"(huber_beta={args.huber_beta}, "
        f"peak_penalty={args.peak_penalty_weight}, "
        f"corr_loss={args.corr_loss_weight}, "
        f"amp_loss={args.amp_loss_weight})"
    )
    return criterion


# ── Phase 15 — resume ───────────────────────────────────────────────────────


def load_checkpoint_if_requested(
    args,
    model,
    optimizer,
    device,
    logger,
):
    """
    Load a previous checkpoint, when ``--resume`` is set.

    Delegates to :meth:`ModelUtils.load_training_checkpoint`, which handles
    DataParallel key rewriting and returns the full training state.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``resume``.
    model : nn.Module
        Model the weights are loaded into.
    optimizer : torch.optim.Optimizer
        Optimizer whose state is restored.
    device : torch.device
        Device to load the checkpoint to.
    logger : Logger

    Returns
    -------
    resume_ckpt : dict or None
    start_epoch : int
    best_val_loss : float
    best_epoch : int
    epochs_no_improve : int
    train_hist, valid_hist : dict
    """
    train_hist, valid_hist = make_history_dicts()

    if not args.resume:
        return None, 1, float("inf"), 0, 0, train_hist, valid_hist

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"--resume {args.resume!r} does not exist")

    # ── Merge the checkpoint's stored args first (CLI overrides win) ──
    resume_ckpt_raw = torch.load(args.resume, map_location="cpu", weights_only=False)
    ckpt_args = resume_ckpt_raw.get("args", {})
    explicit = _cli_explicit_args()
    for k, v in ckpt_args.items():
        if k in explicit:
            continue
        if k == "resume":
            continue
        if hasattr(args, k):
            setattr(args, k, v)

    logger.info(f"Resuming from   : {args.resume}")
    logger.start_task("Loading checkpoint")

    (
        epoch,
        samples_processed,
        batches_processed,
        best_val_loss,
        best_epoch,
        checkpoint,
    ) = ModelUtils.load_training_checkpoint(
        args.resume,
        model,
        optimizer,
        device,
        logger=logger,
    )

    train_hist = checkpoint.get("train_hist", train_hist) if checkpoint else train_hist
    valid_hist = checkpoint.get("valid_hist", valid_hist) if checkpoint else valid_hist
    epochs_no_improve = checkpoint.get("epochs_no_improve", 0) if checkpoint else 0
    start_epoch = epoch + 1

    logger.info(
        f"Resumed         : epoch {epoch}, "
        f"best_val_loss={best_val_loss:.6f}@{best_epoch}"
    )
    return (
        checkpoint,
        start_epoch,
        best_val_loss,
        best_epoch,
        epochs_no_improve,
        train_hist,
        valid_hist,
    )


# ── Phase 16 — inner loops ──────────────────────────────────────────────────


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_grad_norm,
    use_amp=False,
    logger=None,
    epoch=None,
) -> float:
    """
    Run one training epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : callable
    optimizer : torch.optim.Optimizer
    device : torch.device
    max_grad_norm : float
    use_amp : bool, optional
    logger : Logger, optional
    epoch : int, optional
        Used in the log lines only.

    Returns
    -------
    float
        Mean NaN-weighted loss over the epoch.
    """
    model.train()
    total_weighted = torch.zeros((), device=device)
    total_valid = torch.zeros((), device=device)
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t_ep = time.time()
    for b, (features, targets) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            preds = model(features)
            loss = criterion(preds, targets)
        if loss.requires_grad and torch.isfinite(loss).item():
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_valid = torch.isfinite(targets).sum()
        total_weighted += loss.detach().float() * n_valid.float()
        total_valid += n_valid
        if logger is not None and (b % log_every == 0 or b == n_batches):
            logger.info(
                f"    epoch {epoch}: train batch {b}/{n_batches} "
                f"({100 * b // n_batches}%)  {time.time() - t_ep:.0f}s"
            )
    return (total_weighted / total_valid.clamp(min=1)).item()


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    use_amp=False,
    logger=None,
    epoch=None,
) -> tuple:
    """
    Run one validation pass.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : callable
    device : torch.device
    use_amp : bool, optional
    logger : Logger, optional
    epoch : int, optional

    Returns
    -------
    avg_loss : float
    rmse : float
    r2 : float
    """
    model.eval()
    total_weighted = torch.zeros((), device=device)
    total_valid = torch.zeros((), device=device)
    all_preds, all_targets = [], []
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t_ep = time.time()
    for b, (features, targets) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            preds = model(features)
            loss = criterion(preds, targets)
        n_valid = torch.isfinite(targets).sum()
        total_weighted += loss.detach().float() * n_valid.float()
        total_valid += n_valid
        all_preds.append(preds.reshape(-1).float().cpu())
        all_targets.append(targets.reshape(-1).float().cpu())
        if logger is not None and (b % log_every == 0 or b == n_batches):
            logger.info(
                f"    epoch {epoch}: val batch {b}/{n_batches} "
                f"({100 * b // n_batches}%)  {time.time() - t_ep:.0f}s"
            )
    avg_loss = (total_weighted / total_valid.clamp(min=1)).item()
    p = torch.cat(all_preds)
    t = torch.cat(all_targets)
    mask = torch.isfinite(t)
    p, t = p[mask], t[mask]
    if p.numel() == 0:
        return avg_loss, float("nan"), float("nan")
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - t.mean()) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
    return avg_loss, rmse, r2


# ── Phase 16b — epoch loader ────────────────────────────────────────────────


def build_epoch_loader(args, train_full, epoch) -> tuple:
    """
    Build the DataLoader for one epoch by sampling sites and years.

    Parameters
    ----------
    args : argparse.Namespace
        Must expose ``_train_pool`` and ``_train_years`` (attached by
        :func:`run_training`).
    train_full : RamLAIDataset
    epoch : int
        Used to derive the per-epoch RNG seed.

    Returns
    -------
    loader : DataLoader
    n_sites : int
    sampled_years : list of int
    n_indices : int
    """
    epoch_rng = np.random.RandomState(args.seed + epoch)
    n_sites = min(args.n_sites_per_epoch, len(args._train_pool))
    n_years = min(args.n_years_per_epoch, len(args._train_years))

    sampled_sites = epoch_rng.choice(
        args._train_pool,
        size=n_sites,
        replace=False,
    ).tolist()
    sampled_years = sorted(
        epoch_rng.choice(
            args._train_years,
            size=n_years,
            replace=False,
        ).tolist()
    )

    indices = [
        train_full.sample_index[(s, y)]
        for s in sampled_sites
        for y in sampled_years
        if (s, y) in train_full.sample_index
    ]
    loader = DataLoader(
        Subset(train_full, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(sampled_sites), sampled_years, len(indices)


# ── Phase 17 — summary plots ────────────────────────────────────────────────


def generate_summary_plots(exp_dir, train_hist, valid_hist, logger) -> None:
    """
    Write the loss and metric history figures to ``exp_dir``.

    Parameters
    ----------
    exp_dir : str
    train_hist, valid_hist : dict
    logger : Logger
    """
    plot_loss_histories(
        train_hist["loss"],
        valid_hist["loss"],
        filename=os.path.join(exp_dir, "loss_history.png"),
        logger=logger,
    )
    plot_metric_histories(
        train_hist,
        valid_hist,
        filename=os.path.join(exp_dir, "metric_history.png"),
        logger=logger,
    )


def _checkpoint_extras(
    args,
    shared,
    val_rmse,
    val_r2,
    best_r2_epoch=None,
    best_val_r2=None,
):
    """
    Build the dictionary of extra fields that go into every training
    snapshot.

    These are the fields that the predictor reads back from the checkpoint
    to rebuild the dataset and denormalise the output — the site pools,
    the normalisation tables, and the mode flags that drive the wrappers.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Reads the mode flags and the climatology
        year range.
    shared : SharedData
        The resident working-set container. Supplies the site pools and
        the two auxiliary tables.
    val_rmse : float
        Current epoch's validation RMSE, or NaN when unknown.
    val_r2 : float
        Current epoch's validation R², or NaN when unknown.
    best_r2_epoch : int, optional
        Epoch with the highest validation R². Diagnostic only; the paper
        selection rule remains validation loss.
    best_val_r2 : float, optional
        Best validation R² seen so far.

    Returns
    -------
    dict
        Merged into the snapshot by :meth:`ModelUtils.save_training_checkpoint`.
    """
    return {
        # ── Site pools ──
        "train_site_ids": list(shared.train_pool),
        "val_site_ids": list(shared.val_sites),
        # ── Normalisation and CO2 tables ──
        "norm_stats": shared.norm_stats,
        "co2_lut": shared.co2_lut,
        # ── Anomaly mode ──
        "anomaly_mode": bool(args.anomaly_mode),
        "anomaly_clim": shared.anomaly_clim,
        "clim_years": (parse_year_list(args.clim_years) if args.anomaly_mode else None),
        # ── Mode flags that drive the wrapper and denormalisation ──
        "pft_mixing": bool(args.pft_mixing),
        "pft_meteo_only": bool(args.pft_meteo_only),
        "normalize_lai": bool(args.normalize_lai),
        "daily_lai": bool(args.daily_lai),
        # ── Metrics for the run header ──
        "val_rmse": float(val_rmse),
        "val_r2": float(val_r2),
        "selection_rule": (
            "val_loss" if getattr(args, "selection_rule", "val_loss") == "val_loss"
            else "val_r2"
        ),
        "best_r2_epoch": (
            None if best_r2_epoch is None else int(best_r2_epoch)
        ),
        "best_val_r2": (
            None
            if best_val_r2 is None or not np.isfinite(best_val_r2)
            else float(best_val_r2)
        ),
    }


# ── Orchestration ───────────────────────────────────────────────────────────


_HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_rmse",
    "val_r2",
    "lr",
    "time_s",
    "is_best",
    "best_epoch",
    "best_val_loss",
    "is_best_r2",
    "best_r2_epoch",
    "best_val_r2",
]


def _initial_history_rows(train_hist, valid_hist):
    """Rebuild ``history.csv`` rows when resuming from a checkpoint."""
    n = len(valid_hist.get("loss", []))
    rows = []
    best = float("inf")
    best_ep = 0
    best_r2 = -float("inf")
    best_r2_ep = 0
    for i in range(n):
        ep = i + 1
        vl = float(valid_hist["loss"][i])
        vr = float(valid_hist.get("r2", [float("nan")] * n)[i])
        is_best = vl < best
        if is_best:
            best, best_ep = vl, ep
        is_best_r2 = np.isfinite(vr) and vr > best_r2
        if is_best_r2:
            best_r2, best_r2_ep = vr, ep
        rows.append(
            {
                "epoch": ep,
                "train_loss": float(train_hist.get("loss", [float("nan")] * n)[i]),
                "val_loss": vl,
                "val_rmse": float(valid_hist.get("rmse", [float("nan")] * n)[i]),
                "val_r2": vr,
                "lr": float("nan"),
                "time_s": float("nan"),
                "is_best": bool(is_best),
                "best_epoch": best_ep,
                "best_val_loss": best,
                "is_best_r2": bool(is_best_r2),
                "best_r2_epoch": best_r2_ep,
                "best_val_r2": (
                    best_r2 if np.isfinite(best_r2) else float("nan")
                ),
            }
        )
    return rows, best_r2, best_r2_ep


def _write_history(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _HISTORY_FIELDS})


def run_training(args, shared, logger, paths, model_objs, state) -> None:
    """
    Train one model on the resident ``SharedData``.

    Parameters
    ----------
    args : argparse.Namespace
    shared : SharedData
    logger : Logger
    paths : EasyDict
        Directory paths returned by :func:`setup_directories_and_logging`.
    model_objs : tuple
        ``(model, optimizer, scheduler, criterion)``.
    state : tuple
        ``(resume_ckpt, start_epoch, best_val_loss, best_epoch,
        epochs_no_improve, train_hist, valid_hist)``.
    """
    model, optimizer, scheduler, criterion = model_objs
    (
        resume_ckpt,
        start_epoch,
        best_val_loss,
        best_epoch,
        epochs_no_improve,
        train_hist,
        valid_hist,
    ) = state

    device = shared.device
    val_loader = shared.val_loader
    train_full = shared.train_full

    # Attach the epoch-sampling inputs to args for the loader builder.
    args._train_pool = shared.train_pool
    args._train_years = shared.train_years

    logger.start_task(
        "Training",
        f"{args.num_epochs} epochs " f"(start={start_epoch}, patience={args.patience})",
    )

    run_model = torch.compile(model) if args.compile else model

    # ── History / best-epoch bookkeeping ──
    history_rows, best_val_r2, best_r2_epoch = _initial_history_rows(
        train_hist, valid_hist
    )
    history_path = os.path.join(paths.exp, "history.csv")
    if history_rows:
        _write_history(history_path, history_rows)
    best_r2_metrics = None
    best_loss_metrics = None

    for epoch in range(start_epoch, args.num_epochs + 1):
        t0 = time.time()
        loader, n_sites, sampled_years, n_idx = build_epoch_loader(
            args, train_full, epoch
        )
        logger.info(
            f"\nEpoch {epoch:3d}/{args.num_epochs}  "
            f"sites={n_sites:,}  years={sampled_years}"
        )
        if n_idx == 0:
            logger.warning("  Empty epoch dataset — skipped.")
            continue
        logger.info(f"  Train samples: {n_idx:,}")

        train_loss = train_one_epoch(
            run_model,
            loader,
            criterion,
            optimizer,
            device,
            args.max_grad_norm,
            use_amp=args.amp,
            logger=logger,
            epoch=epoch,
        )
        val_loss, val_rmse, val_r2 = validate(
            run_model,
            val_loader,
            criterion,
            device,
            use_amp=args.amp,
            logger=logger,
            epoch=epoch,
        )
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        logger.info(
            f"  train={train_loss:.6f}  val={val_loss:.6f}  "
            f"RMSE={val_rmse:.5f}  R²={val_r2:.4f}  "
            f"lr={lr:.2e}  ({dt:.1f}s)"
        )

        train_hist["loss"].append(train_loss)
        train_hist["rmse"].append(float("nan"))
        train_hist["r2"].append(float("nan"))
        valid_hist["loss"].append(val_loss)
        valid_hist["rmse"].append(val_rmse)
        valid_hist["r2"].append(val_r2)

        is_best_loss = val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_loss
        is_best_r2 = np.isfinite(val_r2) and val_r2 > best_val_r2
        if is_best_r2:
            best_val_r2 = val_r2
            best_r2_epoch = epoch

        if args.selection_rule == "val_r2":
            is_best = is_best_r2
        else:
            is_best = is_best_loss
        if is_best:
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ── History + per-epoch best snapshots ──
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_rmse": float(val_rmse),
                "val_r2": float(val_r2),
                "lr": float(lr),
                "time_s": float(dt),
                "is_best": bool(is_best),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "is_best_r2": bool(is_best_r2),
                "best_r2_epoch": int(best_r2_epoch),
                "best_val_r2": (
                    float(best_val_r2)
                    if np.isfinite(best_val_r2)
                    else float("nan")
                ),
            }
        )
        _write_history(history_path, history_rows)
        epoch_metrics = {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
            "val_r2": float(val_r2),
            "selection_rule": (
                "lowest validation loss"
                if args.selection_rule == "val_loss"
                else "highest validation R²"
            ),
        }
        if is_best_loss:
            best_loss_metrics = dict(epoch_metrics, selection_rule="lowest validation loss")
            write_json(
                os.path.join(paths.exp, "metrics_at_best_loss.json"),
                best_loss_metrics,
            )
        if is_best_r2:
            best_r2_metrics = dict(
                epoch_metrics, selection_rule="highest validation R² (diagnostic)"
            )
            write_json(
                os.path.join(paths.exp, "metrics_at_best_r2.json"),
                best_r2_metrics,
            )

        # ── Every-epoch snapshot (lightweight) ──
        ModelUtils.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            samples_processed=0,
            batches_processed=0,
            train_loss_history=train_hist.get("loss", []),
            valid_loss_history=valid_hist.get("loss", []),
            valid_metrics_history=valid_hist,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            avg_val_loss=val_loss,
            avg_epoch_loss=train_loss,
            args=args,
            paths=paths,
            logger=None,  # silent for the every-epoch snapshot
            checkpoint_type="epoch",
            save_full_model=False,
            extra=_checkpoint_extras(
                args, shared, val_rmse, val_r2,
                best_r2_epoch=best_r2_epoch, best_val_r2=best_val_r2,
            ),
        )

        if is_best:
            ModelUtils.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                samples_processed=0,
                batches_processed=0,
                train_loss_history=train_hist.get("loss", []),
                valid_loss_history=valid_hist.get("loss", []),
                valid_metrics_history=valid_hist,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                avg_val_loss=val_loss,
                avg_epoch_loss=train_loss,
                args=args,
                paths=paths,
                logger=logger,
                checkpoint_type="best",
                save_full_model=True,
                extra=_checkpoint_extras(
                    args,
                    shared,
                    val_rmse,
                    val_r2,
                    best_r2_epoch=best_r2_epoch,
                    best_val_r2=best_val_r2,
                ),
            )
            rule = "val_loss" if args.selection_rule == "val_loss" else "val_R²"
            logger.success(
                f"  ✓ Best checkpoint saved ({rule}: "
                f"val_loss={val_loss:.6f}, val_R²={val_r2:.4f})"
            )

        if epochs_no_improve >= args.patience:
            logger.warning(
                f"Early stopping at epoch {epoch} — no improvement "
                f"for {args.patience} epochs."
            )
            break

        del loader

    logger.success(
        f"Training complete. Best epoch={best_epoch}, "
        f"best_val_loss={best_val_loss:.6f}"
    )

    # ── Final snapshot ──
    ModelUtils.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.num_epochs,
        samples_processed=0,
        batches_processed=0,
        train_loss_history=train_hist.get("loss", []),
        valid_loss_history=valid_hist.get("loss", []),
        valid_metrics_history=valid_hist,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        avg_val_loss=val_loss,
        avg_epoch_loss=train_loss,
        args=args,
        paths=paths,
        logger=logger,
        checkpoint_type="final",
        save_full_model=True,
        extra=_checkpoint_extras(
            args,
            shared,
            val_rmse,
            val_r2,
            best_r2_epoch=best_r2_epoch,
            best_val_r2=best_val_r2,
        ),
    )

    generate_summary_plots(paths.exp, train_hist, valid_hist, logger)
    _write_history(history_path, history_rows)
    # ``args`` carries runtime-only private attributes (for example the
    # in-memory site pool); keep them out of the on-disk config.
    public_args = {
        k: v for k, v in vars(args).items() if not str(k).startswith("_")
    }
    with open(os.path.join(paths.exp, "config.json"), "w") as f:
        json.dump(public_args, f, indent=2, default=str)
    logger.info(f"Config saved to {os.path.join(paths.exp, 'config.json')}")

    # ── Reproducibility metadata (git commit, data hashes, split) ──
    data_paths = [
        p
        for p in (
            args.stats_path,
            args.co2_path,
            args.selected_pixels,
            args.parent_map,
            args.resume,
        )
        if p
    ]
    run_meta = collect_run_metadata(
        args,
        data_paths=data_paths,
        extra={
            "stage": "train",
            "protocol": args.protocol,
            "train_years": shared.train_years,
            "val_years": shared.val_years,
            "test_years": getattr(args, "_test_years", []),
            "norm_years": parse_year_list(args.norm_years),
            "clim_years": (
                parse_year_list(args.clim_years) if args.anomaly_mode else []
            ),
            "selection_rule": args.selection_rule,
            "best_epoch": int(best_epoch),
            "best_val_loss": (
                None if not np.isfinite(best_val_loss) else float(best_val_loss)
            ),
            "best_r2_epoch": int(best_r2_epoch),
            "best_val_r2": (
                None if not np.isfinite(best_val_r2) else float(best_val_r2)
            ),
            "n_train_sites": len(shared.train_pool),
            "n_val_sites": len(shared.val_sites),
        },
    )
    write_json(os.path.join(paths.exp, "run_metadata.json"), run_meta)
    logger.info(f"Run metadata → {os.path.join(paths.exp, 'run_metadata.json')}")


# ── Orchestration helpers — used by main() and by the sweep drivers ─────────


def load_shared(args, logger, device, resume_ckpt=None) -> SharedData:
    """
    Load everything a run reuses across configs, into RAM once.

    Parameters
    ----------
    args : argparse.Namespace
    logger : Logger
    device : torch.device
    resume_ckpt : dict or None, optional
        When set, ``anomaly_clim`` is read from it rather than recomputed.

    Returns
    -------
    SharedData
    """
    if getattr(args, "daily_lai", False) and not args.daily_target_dir:
        raise SystemExit(
            "--daily_lai requires --daily_target_dir " "(folder of LAI_daily_{Y}.nc)."
        )

    train_years = parse_year_list(args.train_years)
    val_years = parse_year_list(args.val_years)
    if not train_years or not val_years:
        raise ValueError("--train_years and --val_years are required.")
    all_years = sorted(set(train_years) | set(val_years))
    logger.info(f"Train years     : {train_years}")
    logger.info(f"Val years       : {val_years}")

    # ── Split / leakage validation (PROTOCOL.md §2) ──
    test_years = parse_year_list(args.test_years)
    args._test_years = test_years
    if test_years:
        logger.info(f"Test years      : {test_years}")
    for w in validate_split(
        train_years,
        val_years,
        test_years,
        protocol=args.protocol,
        strict=not args.allow_leakage,
    ):
        logger.warning(w)
    norm_years = parse_year_list(args.norm_years)
    if norm_years:
        for w in validate_normalisation_years(
            norm_years, train_years, strict=not args.allow_leakage
        ):
            logger.warning(w)
    if args.anomaly_mode:
        # Never fall back to a hard-coded 1992–2010 window: that would leak
        # 2010 into the climatology whenever training stops earlier.  When the
        # user does not pass --clim_years, use the training years.
        if not parse_year_list(args.clim_years):
            args.clim_years = args.train_years
        clim_years = parse_year_list(args.clim_years)
        for w in validate_climatology_years(
            clim_years, train_years, strict=not args.allow_leakage
        ):
            logger.warning(w)

    norm_stats = load_norm_stats(args, logger)
    co2_lut = load_co2_lut_if_requested(args, norm_stats, logger)
    parent_map = load_parent_map_if_requested(args, logger)

    logger.info(
        f"Features        : {FEATURE_CHANNELS} channels, "
        f"seq_length {args.seq_length}"
    )
    logger.info(f"LAI target norm : {'ON' if args.normalize_lai else 'OFF'}")
    if getattr(args, "daily_lai", False):
        logger.info(
            f"Daily LAI mode  : ON — train on 365-day curve "
            f"({args.daily_target_dir}); val scores real dekads only"
        )
    logger.info(f"Anomaly mode    : {args.anomaly_mode}")
    logger.info(
        f"PFT mixing      : {args.pft_mixing}"
        + (
            f" (meteo_only: base sees {PFT_START} channels)"
            if args.pft_mixing and args.pft_meteo_only
            else ""
        )
        + (
            " (nonneg: L_k soft-floored at 0)"
            if args.pft_mixing and args.pft_nonneg
            else ""
        )
    )

    train_pool, val_pool = build_site_pools(args, all_years, logger)

    rng_split = np.random.RandomState(args.seed + 1)
    n_val_used = min(args.n_val_sites, len(val_pool))
    val_sites = rng_split.choice(val_pool, size=n_val_used, replace=False).tolist()
    logger.info(f"Val sites used  : {len(val_sites):,}")

    working_sites = compute_working_set(args, train_pool, train_years)
    logger.info(
        f"Working-set     : {len(working_sites):,} train sites "
        f"(of {len(train_pool):,} pool) loaded into RAM"
    )

    anomaly_clim = compute_anomaly_climatology(
        args, working_sites, val_sites, resume_ckpt, logger
    )

    val_ds, val_loader, train_full = create_datasets_and_loaders(
        args,
        val_sites,
        working_sites,
        train_years,
        val_years,
        norm_stats,
        co2_lut,
        anomaly_clim,
        parent_map,
        logger,
    )

    return SharedData(
        device=device,
        train_pool=train_pool,
        val_sites=val_sites,
        train_years=train_years,
        val_years=val_years,
        norm_stats=norm_stats,
        co2_lut=co2_lut,
        anomaly_clim=anomaly_clim,
        val_ds=val_ds,
        val_loader=val_loader,
        train_full=train_full,
    )


def train_one_config(args, shared, logger, paths, resume_ckpt=None) -> None:
    """
    Train one configuration on an already-resident ``SharedData``.

    The expensive disk load has already happened in :func:`load_shared`.
    This function builds the model and optimizer, loads the resume
    checkpoint if requested, then calls :func:`run_training`.

    Parameters
    ----------
    args : argparse.Namespace
    shared : SharedData
    logger : Logger
    paths : EasyDict
    resume_ckpt : dict or None
    """
    device = shared.device

    # ── Model, optimizer, loss ──
    model, _ = initialize_model(args, shared.norm_stats, device, logger)
    optimizer, scheduler = create_optimizer_and_scheduler(args, model, device, logger)
    criterion = create_loss_function(args, logger)

    # ── Resume ──
    state = load_checkpoint_if_requested(args, model, optimizer, device, logger)
    (
        loaded_ckpt,
        start_epoch,
        best_val_loss,
        best_epoch,
        epochs_no_improve,
        train_hist,
        valid_hist,
    ) = state

    # Scheduler state is loaded after the scheduler exists.
    if loaded_ckpt is not None and "scheduler_state_dict" in loaded_ckpt:
        scheduler.load_state_dict(loaded_ckpt["scheduler_state_dict"])
        logger.info("Scheduler state restored")

    # ── Hand off to the training loop ──
    run_training(
        args,
        shared,
        logger,
        paths,
        (model, optimizer, scheduler, criterion),
        (
            loaded_ckpt,
            start_epoch,
            best_val_loss,
            best_epoch,
            epochs_no_improve,
            train_hist,
            valid_hist,
        ),
    )


# ── Entry point ─────────────────────────────────────────────────────────────


def main():
    """
    Parse the CLI, set up the environment, load the working-set and train.
    """
    args = parse_args()

    # ── Resume: load the checkpoint early so its args can be merged ──
    resume_ckpt = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume {args.resume!r} does not exist")
        print(f"Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_args = resume_ckpt.get("args", {})
        explicit = _cli_explicit_args()
        for k, v in ckpt_args.items():
            if k in explicit:
                continue
            if k == "resume":
                continue
            if hasattr(args, k):
                setattr(args, k, v)

    # ── Environment ──
    paths, logger = setup_directories_and_logging(args)
    log_configuration(args, logger)
    device = setup_device_and_seed(args, logger)

    # ── Load the working-set once ──
    shared = load_shared(args, logger, device, resume_ckpt)

    # ── Train one config ──
    train_one_config(args, shared, logger, paths, resume_ckpt)


if __name__ == "__main__":
    main()
