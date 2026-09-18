#!/usr/bin/env python3
"""
PhenoNN — RAM-resident training entry point (method C2, "working-set").

Standalone trainer: the per-epoch disk re-reads are removed — the raw daily
features (+ targets / PFT / climatology) for every site the training loop will
*ever* sample are loaded into RAM **once** at start-up. Each epoch then only
sub-selects sample indices already in memory — zero per-epoch I/O.

Why "working-set" and not the whole pool
-----------------------------------------
The per-epoch sampler only ever touches `n_sites_per_epoch` sites per epoch,
drawn with `RandomState(seed + epoch)`. Over the whole run the set of *distinct*
sites used is therefore `union over epochs of those draws`
≈ `min(pool, n_sites_per_epoch × n_epochs)`. We replay that exact RNG sequence
here to compute the union, and load only those sites. RAM ≈ N_working × n_years
× 365 × n_dyn × 4 bytes (raw, not the ~6× bigger assembled tensor — assembly is
done on the fly in __getitem__).

This module is self-contained: the CLI, the site-pool builder and the
train/validate loops are all defined below (no shared training module).

Usage, e.g.
    python -m phenonn.training.train_full_ram \\
        --features_dir /data/sbarbu/era5_features --target_dir ... \\
        --selected_pixels /data/sbarbu/PhenoNN/data/selected_pixels_big.nc \\
        --train_years 1992-2010 --val_years 2011-2018 \\
        --num_epochs 30 --n_sites_per_epoch 1000 --n_years_per_epoch 5 ...
"""

import argparse
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
    RamLAIDataset, compute_climatology_lookup, list_valid_sites,
    load_co2_lut, load_parent01_map, load_selected_pixels,
)
from phenonn.utils.evaluater import make_loss_fn
from phenonn.utils.model_loader import build_model, build_model_pft


from phenonn.utils.diagnostics import (
    make_history_dicts, plot_loss_histories, plot_metric_histories,
)
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils


@dataclass
class SharedData:
    """Everything a training run reuses across hyperparameter configs, loaded
    from disk into RAM ONCE by load_shared(): the site pools, normalisation
    tables, and the resident validation / training datasets. Identical for every
    config whose subset / seed / years / n_sites_per_epoch / num_epochs match,
    so run_sweep_inproc.py builds it once and hands it to each train_one_config."""
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


def parse_year_list(s: str):
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _cli_explicit_args():
    explicit = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            explicit.add(arg[2:].split("=")[0].replace("-", "_"))
    return explicit


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Data ──
    p.add_argument("--features_dir", required=False, default="")
    p.add_argument("--target_dir",   required=False, default="")
    p.add_argument("--pft_dir",      required=False, default="")
    p.add_argument("--valid_dir",    required=False, default="",
                   help="Folder of valid_pixels_{Y}.nc (output of "
                        "compute_valid_pixels.py). Used to draw the site "
                        "pool when --selected_pixels is not provided.")
    p.add_argument("--selected_pixels", required=False, default="",
                   help="Path to selected_pixels.nc produced by "
                        "select_pixels.py. When set, the site pool is taken "
                        "from this file (and --valid_dir is ignored).")
    p.add_argument("--parent_map", required=False, default="",
                   help="Optional selected_pixels_01.nc (phenonn.data_creation."
                        "make_selected_pixels_01). When set, ERA5 features are "
                        "read from the deduplicated 0.1° ERA5_daily_pixelset "
                        "(site_id 'E{lat}_{lon}') via the 0.05°→0.1° parent map, "
                        "instead of a 0.05°-indexed feature file. Targets/PFT "
                        "stay on the 0.05° sites. RAM-dataset path only.")
    p.add_argument("--stats_path", default="",
                   help="Optional norm_stats.json for log1p + z-scoring.")
    p.add_argument("--co2_path", default="",
                   help="Optional CO2 LUT (Annee_YYYY=VALUE per line).")
    p.add_argument("--no_normalize_lai", dest="normalize_lai",
                   action="store_false", default=True,
                   help="Skip LAI z-scoring (features still normalized).")
    p.add_argument("--daily_lai", action="store_true", default=False,
                   help="Daily-LAI mode: the model outputs 365 days "
                        "(DailyWrapper, no every-10-days subselect) and trains on "
                        "the linear-interpolated daily curve (--daily_target_dir), "
                        "so the loss covers ALL days. Validation still scores ONLY "
                        "the real dekad observations (from --target_dir). "
                        "Incompatible with --anomaly_mode.")
    p.add_argument("--daily_target_dir", default="",
                   help="Folder of LAI_daily_{Y}.nc (365-day interpolated LAI, "
                        "from interpolate_lai_daily_linear.py). Required with "
                        "--daily_lai; used for the training target only.")
    p.add_argument("--threaded_feature_read", action="store_true", default=False,
                   help="EXPERIMENTAL (RAM dataset only): read the per-year "
                        "feature files concurrently (ThreadPoolExecutor). Faster "
                        "on high-latency NFS, but UNSAFE on venvs whose "
                        "netCDF4/h5py bundle a non-thread-safe libhdf5 "
                        "(concurrent opens segfault). Default: sequential.")

    # ── Sub-grid (used to bound the valid pool) ──
    p.add_argument("--row_min", type=int, default=0)
    p.add_argument("--row_max", type=int, default=-1,
                   help="Inclusive end of lat-index range. -1 = whole grid.")
    p.add_argument("--col_min", type=int, default=0)
    p.add_argument("--col_max", type=int, default=-1)

    # ── Year ranges ──
    p.add_argument("--train_years", default="")
    p.add_argument("--val_years",   default="")

    # ── Per-epoch sampling ──
    p.add_argument("--n_sites_per_epoch", type=int, default=500)
    p.add_argument("--n_years_per_epoch", type=int, default=3)
    p.add_argument("--n_val_sites",       type=int, default=200)
    p.add_argument("--val_fraction_of_grid", type=float, default=0.1,
                   help="Sentinel 100 → OVERLAP mode (train ∪ val drawn from "
                        "the same pool, years must be disjoint).")
    p.add_argument("--subset", type=float, default=1.0,
                   help="Keep only this FRACTION (0, 1] of the site pool "
                        "(applied to train AND val), sampled once with --seed "
                        "BEFORE the train/val split. Caps the sites ever loaded "
                        "into RAM at fraction×pool, independent of --num_epochs "
                        "(the per-epoch --n_sites_per_epoch sampling then draws "
                        "from this reduced pool). 1.0 = full pool.")

    # ── Anomaly mode ──
    p.add_argument("--anomaly_mode", action="store_true")
    p.add_argument("--clim_years", default="1992-2010")
    p.add_argument("--clim_target_dir", default="",
                   help="Folder of LAI_dekadal_{Y}.nc for the climatology "
                        "source (defaults to --target_dir).")

    # ── Model ──
    # The trailing "# models:" note on each option lists which of the models you
    # train read it (all = lstm, aelstm, bitransformer_v2, attnlstm).
    p.add_argument("--type", default="lstm",                    # models: all
                   choices=["lstm", "gru", "transformer", "transformer_dec",
                            "bitransformer", "bitransformer_v2", "attnlstm",
                            "aelstm", "fcn", "linear", "linear_perday"])
    p.add_argument("--pft_mixing", action="store_true")         # models: all (wrapper)
    p.add_argument("--pft_meteo_only", action="store_true",     # models: all (with --pft_mixing)
                   help="With --pft_mixing: the base model sees ONLY the "
                        "meteo/cyclic/co2 channels (not the PFT fractions), so "
                        "each of its 15 pure-LAI outputs L_k depends on climate "
                        "alone; the PFT fractions enter only as mixing weights "
                        "in PFTMixingWrapper. Changes the input channel count "
                        "→ not resumable from a non-meteo_only checkpoint.")
    p.add_argument("--pft_nonneg", action="store_true",         # models: all (with --pft_mixing)
                   help="With --pft_mixing: soft-floor each per-PFT pure LAI at "
                        "physical 0 (softplus anchored at the z-scored zero), so "
                        "no L_k can be negative. Keeps the additive mixing "
                        "consistent. Changes the model → requires retraining.")
    p.add_argument("--seq_length", type=int, default=730)       # models: all (input window via dataset; PE/mask inside aelstm/bitransformer_v2/attnlstm, not lstm)
    p.add_argument("--hidden_size", type=int, default=128)      # models: all (lstm/aelstm hidden; bitransformer_v2 stage-2; attnlstm LSTM)
    p.add_argument("--d_model", type=int, default=None,         # models: bitransformer_v2, attnlstm (stage-1 transformer)
                   help="Transformer embedding dim for bitransformer_v2 / "
                        "attnlstm stage-1. Falls back to --hidden_size when "
                        "unset. For attnlstm, --hidden_size stays the LSTM "
                        "hidden size (decoupled from d_model).")
    p.add_argument("--num_layers", type=int, default=2)         # models: all (bitransformer_v2 stage-2 blocks; attnlstm LSTM layers)
    p.add_argument("--num_layers1", type=int, default=2)        # models: bitransformer_v2, attnlstm (stage-1 transformer blocks)

    p.add_argument("--nhead", type=int, default=4)              # models: aelstm, bitransformer_v2, attnlstm
    p.add_argument("--forward_expansion", type=int, default=4)  # models: aelstm
    p.add_argument("--dropout2", type=float, default=0.0)       # models: aelstm, bitransformer_v2 (stage-2 transformer), attnlstm (LSTM inter-layer, needs num_layers≥2)
    p.add_argument("--dropout1", type=float, default=0.0)       # models: bitransformer_v2, attnlstm (stage-1 transformer)
    p.add_argument("--dropout_att",   type=float, default=0.0)  # models: aelstm

    p.add_argument("--feed_forward_trans1",  type=int, default=4,  # models: bitransformer_v2 stage-1, attnlstm stage-1
                   help="Stage-1 transformer FFN multiplier "
                        "(bitransformer_v2 / attnlstm / bitransformer).")
    p.add_argument("--feed_forward_trans2",  type=int, default=4,  # models: bitransformer_v2 stage-2
                   help="Stage-2 transformer FFN multiplier "
                        "(bitransformer_v2 / bitransformer).")
    p.add_argument("--n_attn_blocks", type=int, default=2)      # models: aelstm
    p.add_argument("--stress_dim", type=int, default=8)         # models: bitransformer_v2, attnlstm

    p.add_argument("--d_layers",   type=int, default=2)         # models: none of lstm/aelstm/bitransformer_v2/attnlstm
    p.add_argument("--embed_size", type=int, default=64)        # models: none of lstm/aelstm/bitransformer_v2/attnlstm

    # ── Training ──
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--loss_type", default="huber",
                   choices=["mse", "mae", "huber", "smoothl1",
                            "nmse", "nmae", "gradient"])
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--gradient_loss_weight", type=float, default=0.5)
    p.add_argument("--peak_penalty_weight",  type=float, default=0.0)
    p.add_argument("--corr_loss_weight", type=float, default=0.0,
                   help="Weight λ for the (1 − masked Pearson) shape/phase term "
                        "(0 = off). Scale-invariant, anti-damping.")
    p.add_argument("--amp_loss_weight", type=float, default=0.0,
                   help="Weight λ for the |std(pred) − std(target)| amplitude "
                        "term (0 = off).")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=11)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast (mixed precision): faster + less GPU "
                        "memory on modern GPUs.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the model in torch.compile (kernel fusion).")
    p.add_argument("--no_fused_adam", dest="fused_adam", action="store_false",
                   default=True,
                   help="Disable the fused CUDA Adam kernel (on by default on GPU).")

    # ── Output / resume ──
    p.add_argument("--output_dir", default="runs_final")
    p.add_argument("--experiment", default="exp_big")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="",
                   help="Path to last_model.pth from a previous run.")

    return p.parse_args(argv)


# ── Train / validate loops (NaN-safe, same logic as main_big) ───────────────


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm,
                    use_amp=False, logger=None, epoch=None):
    model.train()
    # Accumulate on-GPU to avoid a per-batch .item() sync; one sync at epoch end.
    total_weighted = torch.zeros((), device=device)
    total_valid    = torch.zeros((), device=device)
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t_ep = time.time()
    for b, (features, targets) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        targets  = targets .to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp):
            preds = model(features)
            loss  = criterion(preds, targets)
        if loss.requires_grad and torch.isfinite(loss).item():   # NaN guard (1 sync)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_valid = torch.isfinite(targets).sum()
        total_weighted += loss.detach().float() * n_valid.float()
        total_valid    += n_valid
        if logger is not None and (b % log_every == 0 or b == n_batches):
            logger.info(f"    epoch {epoch}: train batch {b}/{n_batches} "
                        f"({100 * b // n_batches}%)  {time.time() - t_ep:.0f}s")
    return (total_weighted / total_valid.clamp(min=1)).item()


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False, logger=None, epoch=None):
    model.eval()
    total_weighted = torch.zeros((), device=device)
    total_valid    = torch.zeros((), device=device)
    all_preds, all_targets = [], []
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)
    t_ep = time.time()
    for b, (features, targets) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        targets  = targets .to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp):
            preds = model(features)
            loss  = criterion(preds, targets)
        n_valid = torch.isfinite(targets).sum()
        total_weighted += loss.detach().float() * n_valid.float()
        total_valid    += n_valid
        all_preds.append(preds.reshape(-1).float().cpu())
        all_targets.append(targets.reshape(-1).float().cpu())
        if logger is not None and (b % log_every == 0 or b == n_batches):
            logger.info(f"    epoch {epoch}: val batch {b}/{n_batches} "
                        f"({100 * b // n_batches}%)  {time.time() - t_ep:.0f}s")
    avg_loss = (total_weighted / total_valid.clamp(min=1)).item()
    p = torch.cat(all_preds)
    t = torch.cat(all_targets)
    mask = torch.isfinite(t)
    p = p[mask]
    t = t[mask]
    if p.numel() == 0:
        return avg_loss, float("nan"), float("nan")
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - t.mean()) ** 2)
    r2 = (1.0 - ss_res / (ss_tot + 1e-12)).item()
    return avg_loss, rmse, r2


# ── Site pool builder ───────────────────────────────────────────────────────


def _build_pools(args, all_years, logger) -> tuple:
    """
    Return (train_pool, val_pool) as lists of site_id strings.

    Two sources for the site pool:
      - `--selected_pixels selected_pixels.nc` (preferred)  : pre-sampled
        subset (e.g. 10 % of valid pixels) used by the new pixelset pipeline.
      - `--valid_dir`  : union over all_years of valid_pixels_{Y}.nc masks,
        optionally restricted to the row/col bbox.
    """
    if args.selected_pixels:
        all_sites = sorted(load_selected_pixels(args.selected_pixels))
        logger.info(f"Site pool       : {len(all_sites):,} sites from "
                    f"{os.path.basename(args.selected_pixels)}")
    elif args.valid_dir:
        bbox_rows = (args.row_min, args.row_max) if args.row_max >= 0 else None
        bbox_cols = (args.col_min, args.col_max) if args.col_max >= 0 else None
        valid_per_year = list_valid_sites(
            args.valid_dir, all_years, bbox_rows, bbox_cols,
        )
        all_sites = sorted(set().union(*valid_per_year.values()))
        if not all_sites:
            raise RuntimeError(
                "No valid pixels found. Run compute_valid_pixels.py first, "
                "and check --valid_dir + --row/col bounds."
            )
        logger.info(f"Site pool       : {len(all_sites):,} unique valid "
                    f"pixels (across {len(all_years)} years)")
    else:
        raise ValueError(
            "Provide either --selected_pixels (recommended for pixelset "
            "pipeline) or --valid_dir."
        )

    # Optional pool subsetting: keep a fixed random fraction of the pool BEFORE
    # the train/val split, so BOTH pools shrink and the working-set (union of
    # per-epoch samples) can never exceed fraction×pool — decoupling resident
    # RAM from --num_epochs. Deterministic (own RandomState(seed)) so the choice
    # is stable across runs/--resume; a separate instance leaves the split RNG
    # below untouched.
    if not (0.0 < args.subset <= 1.0):
        raise ValueError(f"--subset must be in (0, 1], got {args.subset}")
    if args.subset < 1.0:
        n_keep = max(1, int(len(all_sites) * args.subset))
        sub_rng = np.random.RandomState(args.seed)
        all_sites = sorted(sub_rng.choice(all_sites, size=n_keep,
                                          replace=False).tolist())
        logger.info(f"Subset          : {n_keep:,} sites kept "
                    f"({args.subset:.3g}× pool) for train+val")

    rng = np.random.RandomState(args.seed)
    overlap = args.val_fraction_of_grid >= 100
    if overlap:
        train_pool = list(all_sites)
        val_pool   = list(all_sites)
        logger.info("Mode            : OVERLAP — train and val drawn from "
                    "the same pool (years must be disjoint)")
    else:
        shuffled = rng.permutation(all_sites).tolist()
        n_val_pool = max(args.n_val_sites,
                         int(args.val_fraction_of_grid * len(shuffled)))
        val_pool   = shuffled[:n_val_pool]
        train_pool = shuffled[n_val_pool:]
        logger.info(f"Mode            : DISJOINT — train={len(train_pool):,} "
                    f"val={len(val_pool):,}")
    return train_pool, val_pool


# ── Working-set computation (replay the train.py per-epoch sampler) ───────────


def _compute_working_set(args, train_pool, train_years):
    """Union of the sites the per-epoch sampler will draw over the whole run.

    Replays the EXACT RNG sequence used by the training loop below so
    the loaded set matches what the loop samples (each epoch uses an
    independent RandomState(seed + epoch), so a superset over all epochs is
    safe regardless of --resume).
    """
    working: set = set()
    n_years = min(args.n_years_per_epoch, len(train_years))
    for epoch in range(1, args.num_epochs + 1):
        rng = np.random.RandomState(args.seed + epoch)
        n_sites = min(args.n_sites_per_epoch, len(train_pool))
        sites = rng.choice(train_pool, size=n_sites, replace=False)
        # consume the years draw too, to keep the stream identical to train.py
        rng.choice(train_years, size=n_years, replace=False)
        working.update(sites.tolist())
    return sorted(working)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Resume: merge checkpoint args first (same as train.py) ──
    resume_ckpt = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume {args.resume!r} does not exist")
        print(f"Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=False)
        ckpt_args = resume_ckpt.get("args", {})
        explicit  = _cli_explicit_args()
        for k, v in ckpt_args.items():
            if k in explicit and k != "resume":
                continue
            if hasattr(args, k):
                setattr(args, k, v)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    exp_dir  = os.path.join(args.output_dir, args.experiment)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir  = os.path.join(exp_dir, "logs")
    for d in [ckpt_dir, log_dir]:
        FileUtils.makedir(d)

    logger = Logger(console_output=True, file_output=True,
                    log_file=os.path.join(log_dir, "train.log"))
    logger.show_header("PhenoNN training (RAM / method C2 working-set)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device          : {device}")
    # Fixed input shapes (seq_length, batch) → let cuDNN pick the best kernels.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True     # TF32 matmuls (Ampere+)
    torch.backends.cudnn.allow_tf32 = True

    shared = load_shared(args, logger, device, resume_ckpt)
    train_one_config(args, shared, logger, resume_ckpt)


def load_shared(args, logger, device, resume_ckpt=None):
    """Load into RAM ONCE everything a run reuses across hyperparameter configs.

    Parses the year ranges, loads the norm stats / CO2 LUT / 0.05°→0.1° parent
    map, builds the site pools and the working-set, computes the anomaly
    climatology, and loads the validation set and the training working-set into
    RAM. Returns a SharedData. For a sweep whose subset / seed / years /
    n_sites_per_epoch / num_epochs are fixed, this is identical for every config
    — so run_sweep_inproc.py calls it once and reuses the result.
    """
    if getattr(args, "daily_lai", False) and not args.daily_target_dir:
        raise SystemExit("--daily_lai requires --daily_target_dir "
                         "(folder of LAI_daily_{Y}.nc).")

    train_years = parse_year_list(args.train_years)
    val_years   = parse_year_list(args.val_years)
    if not train_years or not val_years:
        raise ValueError("--train_years and --val_years are required.")
    all_years = sorted(set(train_years) | set(val_years))
    logger.info(f"Train years     : {train_years}")
    logger.info(f"Val years       : {val_years}")

    # ── Norm stats (optional) ──
    norm_stats = None
    if args.stats_path:
        if not os.path.exists(args.stats_path):
            raise FileNotFoundError(args.stats_path)
        with open(args.stats_path) as f:
            norm_stats = json.load(f)
        logger.info(f"Norm stats      : {args.stats_path}  "
                    f"({len(norm_stats)} entries)")
    else:
        logger.info("Norm stats      : OFF (raw scales)")

    # ── CO2 LUT (optional) ──
    co2_lut = None
    if args.co2_path:
        if not os.path.exists(args.co2_path):
            raise FileNotFoundError(args.co2_path)
        co2_lut = load_co2_lut(args.co2_path)
        logger.info(f"CO2 LUT         : {args.co2_path}  ({len(co2_lut)} years)")
        if norm_stats is not None and "co2" not in norm_stats:
            raise RuntimeError(
                "stats_path is set but contains no 'co2' entry while CO2 is "
                "enabled. Regenerate the stats file with the CO2 LUT."
            )

    # ── 0.05°→0.1° parent map (optional) ──
    # When set, ERA5 features are read from the deduplicated 0.1° pixelset via
    # each 0.05° site's containing cell; no 0.05°-indexed feature file needed.
    parent_map = None
    if getattr(args, "parent_map", ""):
        if not os.path.exists(args.parent_map):
            raise FileNotFoundError(args.parent_map)
        parent_map = load_parent01_map(args.parent_map)
        logger.info(f"Parent map      : {args.parent_map}  "
                    f"({len(parent_map):,} 0.05°→0.1° links)")
    else:
        logger.info("Parent map      : OFF (features indexed on the site pool)")

    logger.info(f"Features        : {FEATURE_CHANNELS} channels, "
                f"seq_length {args.seq_length}")
    logger.info(f"LAI target norm : {'ON' if args.normalize_lai else 'OFF'}")
    if getattr(args, "daily_lai", False):
        logger.info(f"Daily LAI mode  : ON — train on 365-day curve "
                    f"({args.daily_target_dir}); val scores real dekads only")
    logger.info(f"Anomaly mode    : {args.anomaly_mode}")
    logger.info(f"PFT mixing      : {args.pft_mixing}"
                + (f" (meteo_only: base sees {PFT_START} channels)"
                   if args.pft_mixing and args.pft_meteo_only else "")
                + (" (nonneg: L_k soft-floored at 0)"
                   if args.pft_mixing and args.pft_nonneg else ""))

    # ── Site pools (shared logic with train.py) ──
    train_pool, val_pool = _build_pools(args, all_years, logger)

    rng_split = np.random.RandomState(args.seed + 1)
    n_val_used = min(args.n_val_sites, len(val_pool))
    val_sites = rng_split.choice(val_pool, size=n_val_used,
                                 replace=False).tolist()
    logger.info(f"Val sites used  : {len(val_sites):,}")

    # ── Working-set = sites the loop will ever sample ──
    working_sites = _compute_working_set(args, train_pool, train_years)
    logger.info(f"Working-set     : {len(working_sites):,} train sites "
                f"(of {len(train_pool):,} pool) loaded into RAM")

    # ── Anomaly climatology (optional) — over working-set ∪ val ──
    anomaly_clim = None
    if args.anomaly_mode:
        clim_years = parse_year_list(args.clim_years)
        clim_dir   = args.clim_target_dir or args.target_dir
        if resume_ckpt is not None and resume_ckpt.get("anomaly_clim") is not None:
            anomaly_clim = resume_ckpt["anomaly_clim"]
            logger.info(f"Climatology     : reused from checkpoint "
                        f"({len(anomaly_clim):,} sites)")
        else:
            wanted = list(set(working_sites) | set(val_sites))
            logger.start_task("Climatology",
                              f"{len(clim_years)} years × {len(wanted):,} sites")
            anomaly_clim = compute_climatology_lookup(
                target_dir=clim_dir, clim_years=clim_years,
                wanted_sites=wanted, verbose=False,
            )
            logger.success(f"  {len(anomaly_clim):,} site climatologies built")

    # ── Load datasets into RAM ONCE ──
    logger.start_task("Loading validation set into RAM",
                      f"{len(val_sites)} sites × {len(val_years)} years")
    val_ds = RamLAIDataset(
        features_dir=args.features_dir, target_dir=args.target_dir,
        pft_dir=args.pft_dir, years=val_years, site_ids=val_sites,
        seq_length=args.seq_length, norm_stats=norm_stats,
        anomaly_clim=anomaly_clim, co2_lut=co2_lut,
        normalize_lai=args.normalize_lai, verbose=True,
        threaded_read=args.threaded_feature_read,
        parent_map=parent_map,
        daily_mode=("obs" if getattr(args, "daily_lai", False) else "off"),
    )
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check inputs.")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Val samples     : {len(val_ds):,}")

    logger.start_task("Loading training working-set into RAM",
                      f"{len(working_sites)} sites × {len(train_years)} years")
    train_full = RamLAIDataset(
        features_dir=args.features_dir, target_dir=args.target_dir,
        pft_dir=args.pft_dir, years=train_years, site_ids=working_sites,
        seq_length=args.seq_length, norm_stats=norm_stats,
        anomaly_clim=anomaly_clim, co2_lut=co2_lut,
        normalize_lai=args.normalize_lai, verbose=True,
        threaded_read=args.threaded_feature_read,
        parent_map=parent_map,
        daily_mode=("interp" if getattr(args, "daily_lai", False) else "off"),
        daily_target_dir=getattr(args, "daily_target_dir", ""),
    )
    if len(train_full) == 0:
        raise RuntimeError("Training dataset is empty. Check inputs.")
    logger.info(f"Train samples   : {len(train_full):,} resident in RAM")

    return SharedData(
        device=device, train_pool=train_pool, val_sites=val_sites,
        train_years=train_years, val_years=val_years,
        norm_stats=norm_stats, co2_lut=co2_lut, anomaly_clim=anomaly_clim,
        val_ds=val_ds, val_loader=val_loader, train_full=train_full,
    )


def train_one_config(args, shared, logger, resume_ckpt=None):
    """Build the model/optimizer for ONE hyperparameter config and train it on
    the already-resident SharedData, writing its own checkpoints and plots under
    args.output_dir/args.experiment. Reuses load_shared()'s RAM datasets, so a
    sweep pays the disk load only once.
    """
    device       = shared.device
    train_pool   = shared.train_pool
    val_sites    = shared.val_sites
    train_years  = shared.train_years
    val_years    = shared.val_years
    norm_stats   = shared.norm_stats
    co2_lut      = shared.co2_lut
    anomaly_clim = shared.anomaly_clim
    val_loader   = shared.val_loader
    train_full   = shared.train_full

    exp_dir  = os.path.join(args.output_dir, args.experiment)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    FileUtils.makedir(ckpt_dir)

    # ── Model ──
    if args.pft_mixing:
        model = build_model_pft(args, norm_stats).to(device)
        wrapper_label = "PFTMixingWrapper"
    else:
        model = build_model(args).to(device)
        wrapper_label = ("DailyWrapper" if getattr(args, "daily_lai", False)
                         else "Every10DaysWrapper")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model           : {args.type} + {wrapper_label}  "
                f"({n_params:,} parameters)")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay,
                           fused=(args.fused_adam and device.type == "cuda"))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = make_loss_fn(args)
    logger.info(f"Loss            : {args.loss_type} "
                f"(huber_beta={args.huber_beta}, "
                f"peak_penalty={args.peak_penalty_weight}, "
                f"corr_loss={args.corr_loss_weight}, "
                f"amp_loss={args.amp_loss_weight})")

    # ── Resume restore ──
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 1
    train_hist, valid_hist = make_history_dicts()
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        train_hist        = resume_ckpt.get("train_hist", train_hist)
        valid_hist        = resume_ckpt.get("valid_hist", valid_hist)
        best_val_loss     = resume_ckpt.get("best_val_loss", best_val_loss)
        best_epoch        = resume_ckpt.get("best_epoch", best_epoch)
        epochs_no_improve = resume_ckpt.get("epochs_no_improve", 0)
        start_epoch       = resume_ckpt["epoch"] + 1
        logger.info(f"Resumed         : epoch {resume_ckpt['epoch']}, "
                    f"best_val_loss={best_val_loss:.6f}@{best_epoch}")
        del resume_ckpt

    # ── Training loop (same per-epoch sampling as train.py, but RAM Subset) ──
    logger.start_task("Training", f"{args.num_epochs} epochs "
                      f"(start={start_epoch}, patience={args.patience})")

    # torch.compile wraps only the forward path; `model` stays uncompiled so
    # state_dict / resume / predict keep clean keys (no "_orig_mod." prefix).
    run_model = torch.compile(model) if args.compile else model

    for epoch in range(start_epoch, args.num_epochs + 1):
        t0 = time.time()

        epoch_rng = np.random.RandomState(args.seed + epoch)
        n_sites = min(args.n_sites_per_epoch, len(train_pool))
        n_years = min(args.n_years_per_epoch, len(train_years))
        sampled_sites = epoch_rng.choice(
            train_pool, size=n_sites, replace=False).tolist()
        sampled_years = sorted(epoch_rng.choice(
            train_years, size=n_years, replace=False).tolist())

        # Pull the matching sample indices straight from the RAM dataset.
        indices = [train_full.sample_index[(s, y)]
                   for s in sampled_sites for y in sampled_years
                   if (s, y) in train_full.sample_index]

        logger.info(f"\nEpoch {epoch:3d}/{args.num_epochs}  "
                    f"sites={n_sites:,}  years={sampled_years}")
        if not indices:
            logger.warning("  Empty epoch dataset — skipped.")
            continue
        train_loader = DataLoader(
            Subset(train_full, indices), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=False)
        logger.info(f"  Train samples: {len(indices):,}")

        train_loss = train_one_epoch(
            run_model, train_loader, criterion, optimizer, device,
            args.max_grad_norm, use_amp=args.amp, logger=logger, epoch=epoch)
        val_loss, val_rmse, val_r2 = validate(
            run_model, val_loader, criterion, device, use_amp=args.amp,
            logger=logger, epoch=epoch)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        logger.info(f"  train={train_loss:.6f}  val={val_loss:.6f}  "
                    f"RMSE={val_rmse:.5f}  R²={val_r2:.4f}  "
                    f"lr={lr:.2e}  ({dt:.1f}s)")

        train_hist["loss"].append(train_loss)
        train_hist["rmse"].append(float("nan"))
        train_hist["r2"].append(float("nan"))
        valid_hist["loss"].append(val_loss)
        valid_hist["rmse"].append(val_rmse)
        valid_hist["r2"].append(val_r2)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        snapshot = {
            "epoch": epoch,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimizer.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "val_loss": val_loss, "val_rmse": val_rmse, "val_r2": val_r2,
            "args":            vars(args),
            "norm_stats":      norm_stats,
            "train_site_ids":  train_pool,
            "val_site_ids":    val_sites,
            "train_years":     train_years,
            "val_years":       val_years,
            "model_kind":      "phenon_big",
            "pft_mixing":      bool(args.pft_mixing),
            "normalize_lai":   bool(args.normalize_lai),
            "anomaly_mode":    bool(args.anomaly_mode),
            "anomaly_clim":    anomaly_clim,
            "clim_years":      (parse_year_list(args.clim_years)
                                if args.anomaly_mode else None),
            "co2_lut":         co2_lut,
            "co2_enabled":     co2_lut is not None,
            "train_hist":      train_hist,
            "valid_hist":      valid_hist,
            "best_val_loss":   best_val_loss,
            "best_epoch":      best_epoch,
            "epochs_no_improve": epochs_no_improve,
        }
        torch.save(snapshot, os.path.join(ckpt_dir, "last_model.pth"))
        if is_best:
            torch.save(snapshot, os.path.join(ckpt_dir, "best_model.pth"))
            logger.success(f"  ✓ Best checkpoint saved "
                           f"(val_loss={val_loss:.6f})")

        if epochs_no_improve >= args.patience:
            logger.warning(f"Early stopping at epoch {epoch} — no improvement "
                           f"for {args.patience} epochs.")
            break

        del train_loader

    logger.success(f"Training complete. Best epoch={best_epoch}, "
                   f"best_val_loss={best_val_loss:.6f}")

    plot_loss_histories(
        train_hist["loss"], valid_hist["loss"],
        filename=os.path.join(exp_dir, "loss_history.png"), logger=logger)
    plot_metric_histories(
        train_hist, valid_hist,
        filename=os.path.join(exp_dir, "metric_history.png"), logger=logger)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {os.path.join(exp_dir, 'config.json')}")


if __name__ == "__main__":
    main()
