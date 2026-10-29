# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
NaN-safe loss functions for LAI training.

The target tensor produced by the pixelset dataset carries NaN on dekads
whose source observation is missing. These functions mask those positions
out of every reduction, so missing dekads contribute zero gradient and the
loss magnitude stays comparable across samples.

Supported loss types
--------------------
mse, mae, huber (alias smoothl1), nmse, nmae, gradient

Optional additive penalties
---------------------------
- ``peak_penalty_weight`` : ``|max_t pred - max_t target|``
- ``corr_weight``         : ``1 - Pearson(pred, target)`` (shape / phase)
- ``amp_weight``          : ``|std_t pred - std_t target|`` (seasonal amplitude)

The shape and amplitude terms are scale-invariant complements to MSE/Huber:
a flat prediction scores ``r ≈ 0`` and is penalised, not rewarded.
"""

import torch
import torch.nn as nn

_MIN_VALID_DEKADS = 4  # below this, per-sample corr/amplitude is too noisy


def _peak_penalty_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    NaN-aware L1 penalty on the yearly peak.

    Compares ``max_t pred`` and ``max_t target`` per (sample, channel),
    ignoring NaN entries and skipping samples whose target is entirely NaN.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape ``(..., T)``.
    target : torch.Tensor
        Targets of shape ``(..., T)``, possibly containing NaN.

    Returns
    -------
    torch.Tensor
        Scalar penalty, averaged over samples with at least one valid dekad.
    """
    mask = torch.isfinite(target)
    target_for_max = torch.where(mask, target, torch.full_like(target, float("-inf")))
    pred_max = pred.amax(dim=-1)
    target_max = target_for_max.amax(dim=-1)
    any_valid = mask.any(dim=-1)
    diff = (pred_max - target_max).abs()
    diff = torch.where(any_valid, diff, torch.zeros_like(diff))
    denom = any_valid.to(pred.dtype).sum().clamp(min=1)
    return diff.sum() / denom


def _pearson_term(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Mean of ``1 - Pearson r`` over samples, on the dekad axis.

    Samples with fewer than ``_MIN_VALID_DEKADS`` valid targets or with a
    flat target (zero variance) are excluded from the reduction. A constant
    prediction yields ``r ≈ 0``, so flattening is penalised.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape ``(..., T)``.
    target : torch.Tensor
        Targets of shape ``(..., T)``, possibly containing NaN.
    eps : float, optional
        Numerical floor added to the variance product.

    Returns
    -------
    torch.Tensor
        Scalar term in ``[0, 2]``; 0 means perfect linear agreement.
    """
    mask = torch.isfinite(target)
    mask_f = mask.to(pred.dtype)
    n = mask_f.sum(dim=-1)
    denom = n.clamp(min=1)
    tgt = torch.where(mask, target, torch.zeros_like(target))

    p_mean = (pred * mask_f).sum(dim=-1) / denom
    t_mean = (tgt * mask_f).sum(dim=-1) / denom
    pc = (pred - p_mean.unsqueeze(-1)) * mask_f
    tc = (tgt - t_mean.unsqueeze(-1)) * mask_f

    cov = (pc * tc).sum(dim=-1)
    vp = (pc * pc).sum(dim=-1)
    vt = (tc * tc).sum(dim=-1)
    r = cov / torch.sqrt(vp * vt + eps)

    valid = (n >= _MIN_VALID_DEKADS) & (vt > eps)
    term = torch.where(valid, 1.0 - r, torch.zeros_like(r))
    return term.sum() / valid.to(pred.dtype).sum().clamp(min=1)


def _amplitude_std_term(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Mean of ``|std_t pred - std_t target|`` over samples, on the dekad axis.

    Uses the population standard deviation over the finite target dekads.
    Samples with fewer than ``_MIN_VALID_DEKADS`` valid targets are dropped.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape ``(..., T)``.
    target : torch.Tensor
        Targets of shape ``(..., T)``, possibly containing NaN.
    eps : float, optional
        Numerical floor inside the square roots.

    Returns
    -------
    torch.Tensor
        Scalar amplitude mismatch, in the same units as ``pred``.
    """
    mask = torch.isfinite(target)
    mask_f = mask.to(pred.dtype)
    n = mask_f.sum(dim=-1)
    denom = n.clamp(min=1)
    tgt = torch.where(mask, target, torch.zeros_like(target))

    p_mean = (pred * mask_f).sum(dim=-1) / denom
    t_mean = (tgt * mask_f).sum(dim=-1) / denom
    p_std = torch.sqrt(
        ((pred - p_mean.unsqueeze(-1)) ** 2 * mask_f).sum(dim=-1) / denom + eps
    )
    t_std = torch.sqrt(
        ((tgt - t_mean.unsqueeze(-1)) ** 2 * mask_f).sum(dim=-1) / denom + eps
    )

    valid = n >= _MIN_VALID_DEKADS
    diff = torch.where(valid, (p_std - t_std).abs(), torch.zeros_like(p_std))
    return diff.sum() / valid.to(pred.dtype).sum().clamp(min=1)


def nan_safe_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
    gradient_loss_weight: float = 0.5,
    peak_penalty_weight: float = 0.0,
    corr_weight: float = 0.0,
    amp_weight: float = 0.0,
) -> torch.Tensor:
    """
    NaN-safe loss over the last axis, with optional additive penalties.

    Only positions where ``target`` is finite contribute to the reduction.
    Samples whose target is entirely NaN produce a zero-gradient scalar.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape ``(..., T)``.
    target : torch.Tensor
        Targets of shape ``(..., T)``, possibly containing NaN.
    loss_type : {"mse", "mae", "huber", "smoothl1", "nmse", "nmae", "gradient"}
        Base loss. ``huber`` and ``smoothl1`` are aliases and use
        ``huber_beta`` as the transition point.
    huber_beta : float, optional
        Huber transition between quadratic and linear regimes.
    gradient_loss_weight : float, optional
        Weight of the temporal gradient term (only used when
        ``loss_type == "gradient"``).
    peak_penalty_weight : float, optional
        Weight of the yearly-peak L1 penalty. 0 disables it.
    corr_weight : float, optional
        Weight of the ``1 - Pearson`` shape term. 0 disables it.
    amp_weight : float, optional
        Weight of the amplitude-std term. 0 disables it.

    Returns
    -------
    torch.Tensor
        Scalar loss, differentiable with respect to ``pred``.

    Raises
    ------
    ValueError
        If ``loss_type`` is not one of the supported names.

    Notes
    -----
    Reduction is the mean over valid positions, so the loss magnitude is
    comparable across samples regardless of how many dekads are missing.
    """
    mask = torch.isfinite(target)
    n_valid = mask.sum()
    if n_valid.item() == 0:
        return torch.zeros(
            (), device=pred.device, dtype=pred.dtype, requires_grad=pred.requires_grad
        )

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    mask_flat = mask.reshape(-1)
    eps = 1e-8

    pred_valid = pred_flat[mask_flat]
    target_valid = target_flat[mask_flat]

    # target_clean = torch.where(mask, target, torch.zeros_like(target))
    # diff = pred - target_clean
    # mask_f = mask.to(pred.dtype)

    if loss_type == "mse":
        # base = (diff.pow(2) * mask_f).sum() / n_valid
        base = nn.MSELoss(reduction="sum")(pred_valid, target_valid) / n_valid

    elif loss_type == "mae":
        # base = (diff.abs() * mask_f).sum() / n_valid
        base = nn.L1Loss(reduction="sum")(pred_valid, target_valid) / n_valid

    elif loss_type in ("huber", "smoothl1"):
        # abs_diff = diff.abs()
        # quad = 0.5 * diff.pow(2)
        # linear = huber_beta * (abs_diff - 0.5 * huber_beta)
        # per_elem = torch.where(abs_diff < huber_beta, quad, linear)
        # base = (per_elem * mask_f).sum() / n_valid
        base = (
            nn.SmoothL1Loss(beta=huber_beta, reduction="sum")(pred_valid, target_valid)
            / n_valid
        )

    elif loss_type == "nmse":
        # mse_val = (diff.pow(2) * mask_f).sum() / n_valid
        # target_valid = target[mask]
        # var = target_valid.var(unbiased=False)
        # base = mse_val / (var + eps)
        mse_val = nn.MSELoss(reduction="sum")(pred_valid, target_valid) / n_valid
        var = target_valid.var(unbiased=False)
        base = mse_val / (var + eps)

    elif loss_type == "nmae":
        # mae_val = (diff.abs() * mask_f).sum() / n_valid
        # target_valid = target[mask]
        # std = target_valid.std(unbiased=False)
        # base = mae_val / (std + eps)
        mae_val = nn.L1Loss(reduction="sum")(pred_valid, target_valid) / n_valid
        std = target_valid.std(unbiased=False)
        base = mae_val / (std + eps)

    elif loss_type == "gradient":
        base = nn.MSELoss(reduction="sum")(pred_valid, target_valid) / n_valid
        if pred.shape[-1] >= 2:
            # Temporal differences; the pair is valid iff both target positions are.
            target_d = target[..., 1:] - target[..., :-1]
            pred_d = pred[..., 1:] - pred[..., :-1]
            pair_ok = torch.isfinite(target_d)
            n_pair = pair_ok.sum().clamp(min=1)
            target_d_clean = torch.where(pair_ok, target_d, torch.zeros_like(target_d))
            pair_mask_f = pair_ok.to(pred.dtype)
            grad_loss = ((pred_d - target_d_clean).pow(2) * pair_mask_f).sum() / n_pair
            base = base + gradient_loss_weight * grad_loss
    else:
        raise ValueError(
            f"Unsupported loss_type={loss_type!r}. "
            f"Use one of: mse, mae, huber, smoothl1, nmse, nmae, gradient."
        )

    if peak_penalty_weight > 0.0:
        base = base + peak_penalty_weight * _peak_penalty_l1(pred, target)
    if corr_weight > 0.0:
        base = base + corr_weight * _pearson_term(pred, target)
    if amp_weight > 0.0:
        base = base + amp_weight * _amplitude_std_term(pred, target)
    return base


def make_loss_fn(args) -> "callable":
    """
    Build a loss closure from a parsed CLI namespace.

    Parameters
    ----------
    args : argparse.Namespace or mapping
        Must provide ``loss_type``, ``huber_beta``, ``gradient_loss_weight``.
        Optionally ``peak_penalty_weight``, ``corr_loss_weight``,
        ``amp_loss_weight`` — missing attributes default to 0.

    Returns
    -------
    callable
        A function ``loss_fn(pred, target) -> torch.Tensor`` suitable for
        ``loss.backward()``.
    """

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nan_safe_loss(
            pred,
            target,
            loss_type=args.loss_type,
            huber_beta=args.huber_beta,
            gradient_loss_weight=args.gradient_loss_weight,
            peak_penalty_weight=getattr(args, "peak_penalty_weight", 0.0),
            corr_weight=getattr(args, "corr_loss_weight", 0.0),
            amp_weight=getattr(args, "amp_loss_weight", 0.0),
        )

    return loss_fn


# Backward-compatible alias matching the LaiNN naming
def nan_safe_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nan_safe_loss(pred, target, loss_type="mse")
