#!/usr/bin/env python3
"""
stress_sensitivity.py
=====================

Temporal sensitivity of the LAI prediction to the internal *stress* signal of
a `bitransformer_v2` checkpoint (built with `stress_dim=1`).

Idea
----
BiTransformerV2 factorises the forward as

    weather ──stage1──► stress (B, L, stress_dim) ──stage2──► LAI

We perturb the stress ONE DAY AT A TIME by +δ (δ = one standard deviation of
the stress sequence) and re-run ONLY stage 2. Comparing the perturbed daily
LAI of the last year against the baseline gives, for every input day `i` and
every output day `j`,

    M[i, j] = LAI_perturbed_at_i(j) − LAI_baseline(j)

i.e. "how much does nudging the stress at day i move the predicted LAI at
day j". The result is an (L × 365) matrix (L = input length, 365 = last year).

Because stage 2 is now causal, M[i, j] must be ≈ 0 whenever i > 365 + j
(day j of the last year, global position 365+j, cannot attend to a later
input day). The script checks this and reports the residual leak.

Reduction to a scalar LAI
-------------------------
  - scalar checkpoint (output_dim=1): stage-2 output IS the daily LAI.
  - pft_mixing checkpoint (output_dim=15): the final LAI is the PFT-weighted
    sum Σ_k L_k · f̂_k done by PFTMixingWrapper. The fractions f̂_k do NOT
    depend on the stress, so ΔLAI = Σ_k f̂_k · ΔL_k. We compute the effective
    per-PFT weights (physical fractions, PFT-1 zeroed, optionally renormalised)
    once from the site input and apply them to the 15-channel differences.

Usage
-----
    python -m prediction.stress_sensitivity \\
        --checkpoint runs/exp/checkpoints/best_model.pth \\
        --site pix_0453_01987 \\
        --output out/sensitivity_pix_0453_01987
"""

import argparse
import os

import numpy as np
import torch

from phenonn.utils.config import PFT_NAMES
from phenonn.data.lai_dataset import RamLAIDataset, load_selected_pixels
from phenonn.utils.model_loader import build_model, build_model_pft
from phenonn.models.bitransformer import BiTransformerV2
from phenonn.utils.utils import EasyDict


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir",   default="")
    p.add_argument("--pft_dir",      default="")
    p.add_argument("--site", default="",
                   help="Explicit site_id (pix_LLLL_OOOOO) to probe.")
    p.add_argument("--selected_pixels", default="",
                   help="If --site is empty, take the FIRST site of this "
                        "selected_pixels*.nc.")
    p.add_argument("--year", type=int, default=0,
                   help="Year to probe (default: first val year of the ckpt).")
    p.add_argument("--delta_scale", type=float, default=1.0,
                   help="Perturbation = delta_scale × std(stress). Default 1.0 "
                        "(= one standard deviation).")
    p.add_argument("--per_pft", action="store_true",
                   help="pft_mixing checkpoints only: instead of one scalar-LAI "
                        "matrix, output one matrix per PFT of the PURE per-PFT "
                        "LAI L_k (the base model's 15 output channels, NOT "
                        "weighted by the site fractions). Result is (L, 365, 15).")
    p.add_argument("--average", action="store_true",
                   help="Probe EVERY site of --selected_pixels (or the ckpt's "
                        "val sites) and save a single matrix = elementwise mean "
                        "over sites (each perturbed by its own 1σ of stress), "
                        "instead of one matrix for the first site only.")
    p.add_argument("--n_sites", type=int, default=0,
                   help="With --average: cap to the first N sites (0 = all). "
                        "Cost is N × L stage-2 passes, so start small.")
    p.add_argument("--output", default="stress_sensitivity",
                   help="Output prefix (writes <prefix>.npy and <prefix>.png).")
    return p.parse_args()


# ── Model plumbing ───────────────────────────────────────────────────────────


def _find_bitransformer(model) -> BiTransformerV2:
    """Descend base_model links until the raw BiTransformerV2 is reached."""
    m = model
    for _ in range(6):
        if isinstance(m, BiTransformerV2):
            return m
        m = getattr(m, "base_model", None)
        if m is None:
            break
    raise RuntimeError("No BiTransformerV2 found inside the model — is the "
                       "checkpoint really --type bitransformer_v2?")


def _stage2_from_stress(bit: BiTransformerV2, stress, pft, mask):
    """Replicate BiTransformerV2 stage 2 (Bitransformer.py:268-276) so we can
    feed an arbitrary stress tensor without re-running stage 1."""
    y = torch.cat([stress, pft], dim=-1)          # (B, L, stress_dim + n_pft)
    y = bit.fuse_proj(y)
    y = bit.pos_enc2(y)
    y = bit.norm_emb2(y)
    y = bit.drop_emb2(y)                          # eval() → identity
    y = bit.stage2(y, mask=mask)
    y = bit.stage2_norm(y)
    return bit.to_output(y)                       # (B, L, output_dim)


def _mixing_weights(model, x_full, device) -> torch.Tensor:
    """Effective per-PFT weights w_k = mask_k · f̂_k for a PFTMixingWrapper,
    so that final LAI = Σ_k L_k · w_k. Returns (output_dim,) or None if the
    model is a plain scalar wrapper."""
    if not hasattr(model, "pft_stds"):            # not a PFTMixingWrapper
        return None
    ps, n = model.pft_start, model.n_pft
    pft_norm = x_full[:, ps:ps + n, -1:]          # (1, n, 1)
    pft_real = pft_norm * model.pft_stds + model.pft_means
    pft_real = pft_real.clamp(min=0.0)
    if model.renormalize:
        pft_real = pft_real / pft_real.sum(dim=1, keepdim=True).clamp(min=1e-6)
    w = (model.lai_pft_mask * pft_real).view(-1)  # (n,); PFT-1 zeroed by mask
    return w.to(device)


def _plot(M: np.ndarray, per_pft: bool, label: str, year: int,
          delta_note: str, out_prefix: str) -> None:
    """Heatmap of the (mean) sensitivity matrix. per_pft → 15-panel grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                          # noqa: BLE001
        print(f"[warn] heatmap skipped: {e}")
        return
    vmax = float(np.abs(M).max()) or 1.0
    if per_pft:
        n = M.shape[2]
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(4 * ncols, 3 * nrows))
        im = None
        for k in range(nrows * ncols):
            ax = axes[k // ncols][k % ncols]
            if k >= n:
                ax.axis("off")
                continue
            im = ax.imshow(M[:, :, k], aspect="auto", origin="lower",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            name = PFT_NAMES[k] if k < len(PFT_NAMES) else ""
            ax.set_title(f"PFT{k + 1} {name}", fontsize=8)
        fig.suptitle(f"Pure per-PFT ΔL_k sensitivity to stress — {label} {year} "
                     f"({delta_note})\n"
                     f"y = perturbed input day i, x = output day j (last year)")
        fig.colorbar(im, ax=axes, label="ΔL_k", shrink=0.6)
        fig.savefig(out_prefix + ".png", dpi=130)
    else:
        fig, ax = plt.subplots(figsize=(8, 10))
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Output day j (last year, 0–364)")
        ax.set_ylabel("Perturbed input day i (0–%d)" % (M.shape[0] - 1))
        ax.set_title(f"ΔLAI sensitivity to stress\n{label} {year}  ({delta_note})")
        fig.colorbar(im, ax=ax, label="ΔLAI")
        fig.tight_layout()
        fig.savefig(out_prefix + ".png", dpi=130)
    print(f"Heatmap        → {out_prefix}.png")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args    = EasyDict(ckpt["args"])
    norm_stats    = ckpt.get("norm_stats", None)
    co2_lut       = ckpt.get("co2_lut", None)
    anomaly_clim  = ckpt.get("anomaly_clim", None) if ckpt.get("anomaly_mode") else None
    lai_normalized = bool(ckpt.get("normalize_lai", True))
    is_pft_mixing = bool(ckpt.get("pft_mixing", False))

    if str(train_args.get("type", "")).lower() != "bitransformer_v2":
        raise ValueError("This probe only supports --type bitransformer_v2 "
                         f"(checkpoint is {train_args.get('type')!r}).")

    # ── Rebuild model, extract the raw BiTransformerV2 ──
    if is_pft_mixing:
        model = build_model_pft(train_args, norm_stats).to(device)
    else:
        model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    bit = _find_bitransformer(model)
    if bit.stress_dim != 1:
        print(f"[warn] stress_dim={bit.stress_dim} (expected 1). The probe "
              f"perturbs ALL stress components jointly per day.")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model      : bitransformer_v2 (stress_dim={bit.stress_dim}, "
          f"n_pft={bit.n_pft}, output_dim={bit.output_dim}, "
          f"pft_mixing={is_pft_mixing})")

    # ── Resolve the site list (single site, or many for --average) ──
    if args.average:
        if args.selected_pixels:
            site_list = load_selected_pixels(args.selected_pixels)
        else:
            site_list = list(ckpt.get("val_site_ids", []))
        if not site_list:
            raise ValueError("--average needs --selected_pixels or "
                             "val_site_ids in the checkpoint.")
        if args.n_sites and 0 < args.n_sites < len(site_list):
            site_list = site_list[:args.n_sites]
    else:
        site = args.site
        if not site and args.selected_pixels:
            site = load_selected_pixels(args.selected_pixels)[0]
        if not site:
            val_ids = list(ckpt.get("val_site_ids", []))
            if not val_ids:
                raise ValueError("Provide --site or --selected_pixels.")
            site = val_ids[0]
        site_list = [site]

    year = args.year
    if not year:
        val_years = str(train_args.get("val_years", "") or "")
        year = int(val_years.split("-")[0].split(",")[0])
    print(f"Sites      : {len(site_list)}"
          + ("" if args.average else f"  ({site_list[0]})"))
    print(f"Year       : {year}")

    features_dir = args.features_dir or train_args.get("features_dir", "")
    target_dir   = args.target_dir   or train_args.get("target_dir",   "")
    pft_dir      = args.pft_dir      or train_args.get("pft_dir",      "")

    # One RAM dataset over every requested site (avoids per-site file reopens).
    dataset = RamLAIDataset(
        features_dir=features_dir, target_dir=target_dir, pft_dir=pft_dir,
        years=[year], site_ids=site_list,
        seq_length=int(train_args.get("seq_length", 720)),
        norm_stats=norm_stats, anomaly_clim=anomaly_clim, co2_lut=co2_lut,
        normalize_lai=lai_normalized, verbose=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No site produced a sample for {year} "
                           "(no valid LAI?). Try another year.")

    out_len = 365
    per_pft = args.per_pft
    if per_pft and bit.output_dim <= 1:
        raise ValueError("--per_pft requires a pft_mixing checkpoint "
                         f"(output_dim={bit.output_dim}); rerun training with "
                         "--pft_mixing or drop --per_pft.")

    def probe_site(x_full, mask):
        """L perturbed stage-2 passes for one site → (M, delta, max_dev)."""
        base_in = x_full[:, : bit.input_dim, :]          # channels the base sees
        inp = base_in.permute(0, 2, 1).contiguous()      # (1, L, input_dim)
        with torch.no_grad():
            out_full, stress0 = bit(inp, return_stress=True)
        pft = inp[:, :, -bit.n_pft:] if bit.n_pft > 0 else inp[:, :, :0]
        # Sanity: stage-2 replay must match the model's own forward.
        with torch.no_grad():
            replay = _stage2_from_stress(bit, stress0, pft, mask)
        max_dev = (replay - out_full).abs().max().item()
        w = _mixing_weights(model, x_full, device)       # (out_dim,) or None

        def reduce_last_year(out):                       # → (365,) or (365, n_pft)
            last = out[0, -out_len:, :]
            if per_pft:
                return last                              # pure per-PFT L_k
            if w is None:
                return last[:, 0]                        # scalar model
            return (last * w.view(1, -1)).sum(dim=-1)    # PFT-weighted scalar LAI

        base_red = reduce_last_year(out_full)
        delta = args.delta_scale * float(stress0[0, :, :].std())
        Lx = x_full.size(-1)
        M = np.zeros((Lx,) + tuple(base_red.shape), dtype=np.float32)
        with torch.no_grad():
            for i in range(Lx):
                s = stress0.clone()
                s[0, i, :] += delta                      # nudge every stress-dim
                red_p = reduce_last_year(
                    _stage2_from_stress(bit, s, pft, mask))
                M[i] = (red_p - base_red).cpu().numpy()
        return M, delta, max_dev

    def causality(M):
        """max |M| in the future region (i > 365+j) vs the causal region."""
        Lx = M.shape[0]
        ii = np.arange(Lx)[:, None]
        jj = np.arange(out_len)[None, :]
        future = ii > (Lx - out_len) + jj
        leak = float(np.abs(M[future]).max()) if future.any() else 0.0
        signal = float(np.abs(M[~future]).max())
        return leak, signal

    # ── Iterate sites, accumulate the (mean) matrix ──
    mask = None
    M_sum = None
    n_done = n_skip = 0
    max_leak = max_signal = 0.0
    last_delta = 0.0
    for s_id in site_list:
        idx = dataset.sample_index.get((s_id, year))
        if idx is None:
            n_skip += 1
            continue
        feats, _ = dataset[idx]                          # (C, L)
        x_full = feats.unsqueeze(0).to(device)           # (1, C, L)
        Lx = x_full.size(-1)
        if Lx < out_len:
            raise RuntimeError(f"Input length {Lx} < 365; cannot slice a year.")
        if mask is None:
            mask = bit._get_mask(Lx)
        M, last_delta, max_dev = probe_site(x_full, mask)
        if max_dev > 1e-4:
            print(f"[warn] {s_id}: stage-2 replay deviates by {max_dev:.2e}")
        leak, signal = causality(M)
        max_leak = max(max_leak, leak)
        max_signal = max(max_signal, signal)
        M_sum = M.astype(np.float64) if M_sum is None else M_sum + M
        n_done += 1
        if args.average and (n_done == 1 or n_done % 10 == 0):
            print(f"  probed {n_done} site(s) …")

    if n_done == 0:
        raise RuntimeError("No site in the list produced a usable sample.")
    M_mean = (M_sum / n_done).astype(np.float32)
    if args.average:
        delta_note = f"mean of {n_done} sites, each δ={args.delta_scale}×std"
        print(f"\nAveraged over {n_done} site(s) ({n_skip} skipped for no "
              f"sample). Each site is perturbed by its own "
              f"{args.delta_scale}×std(stress); matrices averaged elementwise.")
    else:
        delta_note = f"δ={last_delta:.3g}"
        print(f"stress     : perturbation δ = {last_delta:.5f}")

    # ── Causality check (worst case over sites) ──
    print(f"\nCausality  : max |M| in the future region = {max_leak:.3e}  "
          f"(vs {max_signal:.3e} in the causal region)")
    if max_leak > 1e-3 * max(max_signal, 1e-12):
        print("  [warn] non-negligible future leak — stage 2 may not be fully "
              "causal for this checkpoint.")
    else:
        print("  ✓ output day j only responds to input days ≤ its own position.")

    # ── Save ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    np.save(args.output + ".npy", M_mean)
    print(f"\nMatrix ({'×'.join(str(s) for s in M_mean.shape)}) → "
          f"{args.output}.npy")

    label = f"mean of {n_done} sites" if args.average else site_list[0]
    _plot(M_mean, per_pft, label, year, delta_note, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
