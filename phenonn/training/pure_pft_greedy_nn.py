#!/usr/bin/env python3
"""
pure_pft_greedy_nn.py
=====================

Greedy, per-PFT SEQUENCE-model training — the neural-network analogue of
`phenonn.prediction.pure_pft_greedy_XGBoost`, on the *pure-PFT* pools produced by
`phenonn.data_creation.greedy_pure_pft` (`selected_pixels_PFT{n}.nc` +
`extraction_order.txt`, ORCHIDEE numbering 1..15).

Idea — linear-mixture unmixing across PFTs (identical to the XGBoost version)
----------------------------------------------------------------------------
Observed dekadal LAI at a pixel is a cover-weighted sum of pure-PFT LAI curves,
each a function of the WEATHER only (the 15 PFT-fraction channels are DROPPED
from the model input — inside one pure pool they are near-constant and would make
a model extrapolate wildly when reused on another PFT's pixels):

    LAI_obs(pixel) ≈ Σ_p  frac_p(pixel) · L_p(weather)

Given the greedy ORDER of PFTs [p1, p2, …], train one sequence model m_k = L_{p_k}
at a time. When training m_k on p_k's pool, remove the contributions of the PFTs
already fitted, then divide by p_k's own fraction so the model learns the PURE
curve:

    target_k = ( LAI_obs − Σ_{j<k} frac_{p_j} · m_j(weather) ) / frac_{p_k}

Everything is in RAW LAI units (the mixture only holds there), so the datasets are
built with normalize_lai=False. Rows whose frac_{p_k} < --frac_floor are dropped
(dividing by a tiny fraction is unstable).

Model
-----
Each m_k is a base model (--type ∈ lstm/aelstm/attnlstm/bitransformer_v2) built
METEO-ONLY (feature_channel=PFT_START, no PFT input) with a single output curve,
wrapped in Every10DaysWrapper → (B, 1, 36). One fixed --type is used for every
PFT (rerun with a different --type to compare). No PFTMixingWrapper here: the
mixing is done explicitly by the residual scheme above.

Reuses train_full_ram's RAM dataset (RamLAIDataset, which already holds the raw
per-site PFT fractions and raw dekadal LAI), loss, and train/validate loops.
Training ONLY — saves one model per PFT plus a meta describing order + residual.

Output
------
    {output_dir}/{experiment}/
        model_PFT{n}.pth   one model per PFT, in the given order
        meta.json          order, per-PFT stats, args

Usage
-----
    python -m phenonn.training.pure_pft_greedy_nn \\
        --greedy_dir  /data/.../greedy_PFT_10000_0.1 \\
        --features_dir DATA/era5_pixelset --target_dir DATA/LAI_pixelset \\
        --pft_dir DATA/PFT_pixelset --parent_map DATA/selected_pixels01_1.nc \\
        --stats_path DATA/norm_stats_1992_2019.json --co2_path DATA/CO2....txt \\
        --train_years 1992-2009 --val_years 2010-2019 --val_fraction_of_grid 100 \\
        --type lstm --hidden_size 256 --num_layers 2 --learning_rate 1e-3 \\
        --num_epochs 300 --patience 20 --batch_size 32 --amp \\
        --output_dir runs_greedy --experiment greedy_pft_lstm
        # --pft_order 4,10,6   (optional; else read greedy_dir/extraction_order.txt)
"""

import argparse
import copy
import json
import os
import re
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from phenonn.utils.config import PFT_START
from phenonn.data.lai_dataset import (
    RamLAIDataset, load_co2_lut, load_parent01_map, load_selected_pixels,
)
from phenonn.utils.evaluater import make_loss_fn
from phenonn.utils.wrappers import Every10DaysWrapper, permuteWrapper
from phenonn.utils.model_loader import (
    _resolve_d_model, _ff1, _ff2, _nl1, _do1, _do2,
)
from phenonn.models.rnn import RNN_LSTM
from phenonn.models.aelstm import AELSTM
from phenonn.models.attn_lstm import AttnLSTM
from phenonn.models.bitransformer import BiTransformerV2
from phenonn.training import train_full_ram as T
from phenonn.utils.logger import Logger
from phenonn.utils.utils import FileUtils

PFT_FILE = "selected_pixels_PFT{n}.nc"
ORDER_FILE = "extraction_order.txt"


# ── PFT order resolution (same parsing as pure_pft_greedy_XGBoost) ────────────


def _read_order(greedy_dir: str) -> List[int]:
    path = os.path.join(greedy_dir, ORDER_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No --pft_order given and no {ORDER_FILE} in {greedy_dir}.")
    order: List[int] = []
    with open(path) as f:
        for line in f:
            m = re.search(r"\bPFT\s*(\d+)", line)
            if m:
                order.append(int(m.group(1)))
    if not order:
        raise ValueError(f"Could not parse any PFT number from {path}.")
    return order


# ── Meteo-only, single-output base (feature_channel=PFT_START, output=1) ──────


def build_pure_base(args) -> torch.nn.Module:
    """Base model that sees ONLY the meteo/co2 channels (PFT_START) and outputs a
    single pure-LAI curve, wrapped in Every10DaysWrapper → (B, 1, 36)."""
    t = args.type.lower()
    ch = PFT_START
    if t == "lstm":
        base = RNN_LSTM(feature_channel=ch, output_channel=1,
                        hidden_size=args.hidden_size, num_layers=args.num_layers)
    elif t == "aelstm":
        base = AELSTM(feature_channel=ch, output_channel=1,
                      hidden_size=args.hidden_size, num_layers=args.num_layers,
                      n_attn_blocks=args.n_attn_blocks, nhead=args.nhead,
                      ff_expansion=args.forward_expansion,
                      dropout=_do2(args), dropout_att=args.dropout_att,
                      seq_length=args.seq_length)
    elif t == "bitransformer_v2":
        base = permuteWrapper(BiTransformerV2(
            input_dim=ch, output_dim=1,
            d_model=_resolve_d_model(args), d_model2=args.hidden_size, n_pft=0,
            stress_dim=args.stress_dim,
            nr_blocks_stage1=_nl1(args), nr_blocks_stage2=args.num_layers,
            nhead=args.nhead, feed_forward_trans=_ff1(args),
            feed_forward_encoder=_ff2(args),
            dropout_trans=_do1(args), dropout_encoder=_do2(args),
            seq_length=args.seq_length, causal=True))
    elif t == "attnlstm":
        base = permuteWrapper(AttnLSTM(
            input_dim=ch, output_dim=1,
            d_model=_resolve_d_model(args), lstm_hidden=args.hidden_size, n_pft=0,
            stress_dim=args.stress_dim,
            nr_blocks_stage1=_nl1(args), lstm_layers=args.num_layers,
            nhead=args.nhead, feed_forward_trans=_ff1(args),
            dropout_trans=_do1(args), dropout_lstm=_do2(args),
            seq_length=args.seq_length, causal=True))
    else:
        raise SystemExit("greedy NN supports --type lstm/aelstm/attnlstm/"
                         f"bitransformer_v2, got {args.type!r}")
    return Every10DaysWrapper(base)


# ── Residual target (uses RamLAIDataset's raw LAI + raw PFT fractions) ────────


class _ResidualView(Dataset):
    """Yields (weather_features (PFT_START, seq), residual_target (1, 36))."""

    def __init__(self, ds: RamLAIDataset, kept: List[int], resid: torch.Tensor):
        self.ds, self.kept, self.resid = ds, kept, resid

    def __len__(self) -> int:
        return len(self.kept)

    def __getitem__(self, i: int):
        feat, _ = self.ds[self.kept[i]]              # (FEATURE_CHANNELS, seq)
        return feat[:PFT_START].contiguous(), self.resid[i]


def _residual_targets(
    ds: RamLAIDataset, pft_num: int,
    prev_models: List[Tuple[int, torch.nn.Module]],
    frac_floor: float, device, batch_size: int,
) -> Tuple[List[int], torch.Tensor]:
    """Compute residual targets for every kept sample of `ds`.

    kept = samples whose frac_{pft_num} > frac_floor (and finite). For each,
    residual = (LAI_obs − Σ_{prev} frac_j · m_j(weather)) / frac_k, in raw units.
    """
    n = len(ds)
    kept: List[int] = []
    resid_chunks: List[torch.Tensor] = []
    for start in range(0, n, batch_size):
        feats, tgts, fk, loc = [], [], [], []
        fjs = {num: [] for num, _ in prev_models}
        for idx in range(start, min(start + batch_size, n)):
            site, year = ds.index[idx]
            fr = ds._pft[year][site]
            fkk = float(fr[pft_num - 1])
            if not np.isfinite(fkk) or fkk <= frac_floor:
                continue
            feat, tgt = ds[idx]
            feats.append(feat[:PFT_START])
            tgts.append(tgt)
            fk.append(fkk)
            for num, _ in prev_models:
                v = float(fr[num - 1])
                fjs[num].append(v if np.isfinite(v) else 0.0)
            loc.append(idx)
        if not loc:
            continue
        W = torch.stack(feats).to(device)            # (b, PFT_START, seq)
        Y = torch.stack(tgts).to(device)             # (b, 1, 36)
        contrib = torch.zeros_like(Y)
        with torch.no_grad():
            for num, mdl in prev_models:
                pred = mdl(W)
                w = torch.tensor(fjs[num], device=device).view(-1, 1, 1)
                contrib += w * pred
        fkt = torch.tensor(fk, device=device).view(-1, 1, 1)
        resid_chunks.append(((Y - contrib) / fkt).cpu())
        kept.extend(loc)
    resid = (torch.cat(resid_chunks, 0) if resid_chunks
             else torch.empty(0, 1, 36))
    return kept, resid


# ── Site split ────────────────────────────────────────────────────────────────


def _split_pool(args, pool: List[str]) -> Tuple[List[str], List[str]]:
    """OVERLAP (val_fraction_of_grid >= 100): train ∪ val = pool, disjoint years.
    Else: random hold-out of --n_val_sites for validation."""
    if args.val_fraction_of_grid >= 100:
        return pool, pool
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(pool))
    n_val = min(args.n_val_sites, max(1, len(pool) // 5))
    val = [pool[i] for i in perm[:n_val]]
    train = [pool[i] for i in perm[n_val:]]
    return train, val


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--greedy_dir", required=True,
                    help="Dir with selected_pixels_PFT{n}.nc (+ extraction_order.txt).")
    ap.add_argument("--pft_order", default="",
                    help="Comma ORCHIDEE PFT numbers, e.g. '4,10,6'. "
                         "Empty → read greedy_dir/extraction_order.txt.")
    ap.add_argument("--frac_floor", type=float, default=0.05,
                    help="Skip rows whose current-PFT fraction is below this.")
    ap.add_argument("--max_sites", type=int, default=0,
                    help="Cap each per-PFT pool to this many sites (random, "
                         "seeded subsample). 0 = whole pool. Guards RAM: a "
                         "widespread PFT's pure pool can reach >100k sites, "
                         "which OOMs when loaded × all years.")
    known, rest = ap.parse_known_args(argv)
    args = T.parse_args(rest)                      # standard train_full_ram CLI
    args.greedy_dir = known.greedy_dir
    args.pft_order = known.pft_order
    args.frac_floor = known.frac_floor
    args.max_sites = known.max_sites
    return args


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args.output_dir, args.experiment)
    FileUtils.makedir(exp_dir)
    logger = Logger(console_output=True, file_output=True,
                    log_file=os.path.join(exp_dir, "train.log"))

    order = ([int(x) for x in args.pft_order.split(",") if x.strip()]
             if args.pft_order else _read_order(args.greedy_dir))
    logger.show_header(f"Greedy pure-PFT NN — {args.type}  order={order}")

    train_years = T.parse_year_list(args.train_years)
    val_years = T.parse_year_list(args.val_years)
    if not train_years or not val_years:
        raise ValueError("--train_years and --val_years are required.")
    if args.val_fraction_of_grid >= 100 and set(train_years) & set(val_years):
        raise ValueError("OVERLAP mode (val_fraction_of_grid>=100) needs "
                         "disjoint train/val years.")

    norm_stats = None
    if args.stats_path:
        if not os.path.exists(args.stats_path):
            raise FileNotFoundError(args.stats_path)
        with open(args.stats_path) as f:
            norm_stats = json.load(f)
    co2_lut = load_co2_lut(args.co2_path) if args.co2_path else None
    parent_map = (load_parent01_map(args.parent_map)
                  if getattr(args, "parent_map", "") else None)

    def _make_ds(site_ids, years, tag):
        logger.start_task(f"Loading {tag}", f"{len(site_ids):,} sites × {len(years)} yr")
        return RamLAIDataset(
            features_dir=args.features_dir, target_dir=args.target_dir,
            pft_dir=args.pft_dir, years=years, site_ids=site_ids,
            seq_length=args.seq_length, norm_stats=norm_stats,
            anomaly_clim=None, co2_lut=co2_lut,
            normalize_lai=False,                    # residual scheme needs RAW LAI
            verbose=True, threaded_read=args.threaded_feature_read,
            parent_map=parent_map,
        )

    prev_models: List[Tuple[int, torch.nn.Module]] = []
    meta_pft: List[dict] = []

    for step, pft_num in enumerate(order, start=1):
        sel_path = os.path.join(args.greedy_dir, PFT_FILE.format(n=pft_num))
        if not os.path.exists(sel_path):
            logger.warning(f"[step {step}] PFT{pft_num} — missing "
                           f"{os.path.basename(sel_path)}, skipped")
            continue
        pool = load_selected_pixels(sel_path)
        if args.max_sites and len(pool) > args.max_sites:
            rng = np.random.RandomState(args.seed + pft_num)
            capped = rng.choice(pool, size=args.max_sites, replace=False)
            pool = sorted(capped.tolist())
            logger.info(f"  PFT{pft_num} pool capped {args.max_sites:,} "
                        f"(--max_sites)")
        train_sites, val_sites = _split_pool(args, pool)
        logger.info(f"\n[step {step}] PFT{pft_num}  pool={len(pool):,} "
                    f"(train={len(train_sites):,}, val={len(val_sites):,})")

        train_ds = _make_ds(train_sites, train_years, f"PFT{pft_num}-train")
        val_ds = _make_ds(val_sites, val_years, f"PFT{pft_num}-val")

        kept_tr, resid_tr = _residual_targets(
            train_ds, pft_num, prev_models, args.frac_floor, device, args.batch_size)
        kept_va, resid_va = _residual_targets(
            val_ds, pft_num, prev_models, args.frac_floor, device, args.batch_size)
        logger.info(f"  usable samples: train {len(kept_tr):,}/{len(train_ds):,}  "
                    f"val {len(kept_va):,}/{len(val_ds):,}  "
                    f"(frac<{args.frac_floor} dropped)")
        if not kept_tr or not kept_va:
            logger.warning(f"  → PFT{pft_num} skipped (empty after masking).")
            continue

        train_loader = DataLoader(
            _ResidualView(train_ds, kept_tr, resid_tr), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(
            _ResidualView(val_ds, kept_va, resid_va), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = build_pure_base(args).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  model: {args.type} meteo-only→1  ({n_params:,} params)")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5)
        criterion = make_loss_fn(args)

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        no_improve = 0
        t0 = time.time()
        for epoch in range(1, args.num_epochs + 1):
            train_loss = T.train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                args.max_grad_norm, use_amp=args.amp, logger=logger, epoch=epoch)
            val_loss, val_rmse, val_r2 = T.validate(
                model, val_loader, criterion, device, use_amp=args.amp,
                logger=logger, epoch=epoch)
            scheduler.step(val_loss)
            logger.info(f"  PFT{pft_num} epoch {epoch:3d}/{args.num_epochs}  "
                        f"train={train_loss:.6f}  val={val_loss:.6f}  "
                        f"RMSE={val_rmse:.5f}  R²={val_r2:.4f}")
            if val_loss < best_val:
                best_val, best_epoch, no_improve = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    logger.warning(f"  early stopping at epoch {epoch}.")
                    break

        model.load_state_dict(best_state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        prev_models.append((pft_num, model))

        model_path = os.path.join(exp_dir, f"model_PFT{pft_num}.pth")
        torch.save({
            "model_state_dict": best_state,
            "model_kind":       "nn_greedy_pure_pft",
            "type":             args.type,
            "pft_orchidee":     pft_num,
            "step":             step,
            "feature_channels": PFT_START,
            "residual":         "target_k = (LAI - sum_{j<k} frac_j*m_j)/frac_k",
            "normalize_lai":    False,
            "norm_stats":       norm_stats,
            "co2_lut":          co2_lut,
            "train_years":      train_years,
            "val_years":        val_years,
            "args":             vars(args),
        }, model_path)
        logger.success(f"  PFT{pft_num} done in {time.time()-t0:.1f}s  "
                       f"best_val={best_val:.6f}@{best_epoch} → {os.path.basename(model_path)}")
        meta_pft.append({
            "step": step, "pft_orchidee": pft_num,
            "model_file": os.path.basename(model_path),
            "n_train": len(kept_tr), "n_val": len(kept_va),
            "best_epoch": best_epoch, "best_val_loss": best_val,
        })

    if not meta_pft:
        raise RuntimeError("No PFT model trained — check --greedy_dir / order.")

    meta = {
        "model_kind":   "nn_greedy_pure_pft",
        "type":         args.type,
        "residual":     "target_k = (LAI - sum_{j<k} frac_j*m_j(weather))/frac_k",
        "pft_channels_in_features": False,
        "order":        [m["pft_orchidee"] for m in meta_pft],
        "frac_floor":   args.frac_floor,
        "train_years":  train_years, "val_years": val_years,
        "per_pft":      meta_pft, "args": vars(args),
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.success(f"\nTrained {len(meta_pft)} PFT model(s) → {exp_dir}")


if __name__ == "__main__":
    main()
