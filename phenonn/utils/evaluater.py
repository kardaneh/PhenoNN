"""
NaN-safe loss functions for LAI training.

Behaviour copied (without modification) from
LaiNN/phenocam/main_big.nan_safe_loss / make_loss_fn. The target tensor
typically carries NaN on the dekads whose source H5 was missing; these
positions are masked out so they contribute zero gradient and the
reduction stays comparable across samples.

Supported loss types
--------------------
    mse, mae, huber/smoothl1, nmse, nmae, gradient

Peak penalty
------------
When `peak_penalty_weight > 0`, adds
    peak_penalty_weight · |max_t(pred) − max_t(target)|
averaged over (sample, channel). Matches main_big behaviour.

Shape / amplitude penalties (anti-damping)
------------------------------------------
Both default to weight 0 (inactive) and are NaN-safe, reduced per
(sample, channel) over the dekad axis:
    corr_weight · (1 − Pearson(pred, target))   → shape / phase, scale-invariant
    amp_weight  · |std_t(pred) − std_t(target)|  → seasonal amplitude
Pearson cannot be gamed by flattening (a constant prediction scores r=0),
so it complements MSE/Huber when the model damps the seasonal cycle.
"""

import torch

_MIN_VALID_DEKADS = 4   # below this, per-sample corr/amplitude is too noisy


def _peak_penalty_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NaN-aware yearly-peak L1 penalty."""
    mask = torch.isfinite(target)
    target_for_max = torch.where(
        mask, target, torch.full_like(target, float("-inf"))
    )
    pred_max   = pred.amax(dim=-1)
    target_max = target_for_max.amax(dim=-1)
    any_valid  = mask.any(dim=-1)
    diff       = (pred_max - target_max).abs()
    diff       = torch.where(any_valid, diff, torch.zeros_like(diff))
    denom      = any_valid.to(pred.dtype).sum().clamp(min=1)
    return diff.sum() / denom


def _pearson_term(pred: torch.Tensor, target: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Mean over samples of (1 − Pearson r) computed on the dekad axis, masked to
    the finite target dekads. Samples with < _MIN_VALID_DEKADS valid points or a
    ~flat target (zero variance) are excluded from the reduction. A constant
    prediction yields r≈0 → term≈1, so flattening is penalised, not rewarded.
    """
    mask   = torch.isfinite(target)
    mask_f = mask.to(pred.dtype)
    n      = mask_f.sum(dim=-1)
    denom  = n.clamp(min=1)
    tgt    = torch.where(mask, target, torch.zeros_like(target))

    p_mean = (pred * mask_f).sum(dim=-1) / denom
    t_mean = (tgt  * mask_f).sum(dim=-1) / denom
    pc = (pred - p_mean.unsqueeze(-1)) * mask_f
    tc = (tgt  - t_mean.unsqueeze(-1)) * mask_f

    cov = (pc * tc).sum(dim=-1)
    vp  = (pc * pc).sum(dim=-1)
    vt  = (tc * tc).sum(dim=-1)
    r   = cov / torch.sqrt(vp * vt + eps)

    valid  = (n >= _MIN_VALID_DEKADS) & (vt > eps)
    term   = torch.where(valid, 1.0 - r, torch.zeros_like(r))
    return term.sum() / valid.to(pred.dtype).sum().clamp(min=1)


def _amplitude_std_term(pred: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Mean over samples of |std_t(pred) − std_t(target)| on the dekad axis, masked
    to the finite target dekads (population std over the valid positions).
    """
    mask   = torch.isfinite(target)
    mask_f = mask.to(pred.dtype)
    n      = mask_f.sum(dim=-1)
    denom  = n.clamp(min=1)
    tgt    = torch.where(mask, target, torch.zeros_like(target))

    p_mean = (pred * mask_f).sum(dim=-1) / denom
    t_mean = (tgt  * mask_f).sum(dim=-1) / denom
    p_std  = torch.sqrt(((pred - p_mean.unsqueeze(-1)) ** 2 * mask_f).sum(dim=-1) / denom + eps)
    t_std  = torch.sqrt(((tgt  - t_mean.unsqueeze(-1)) ** 2 * mask_f).sum(dim=-1) / denom + eps)

    valid = n >= _MIN_VALID_DEKADS
    diff  = torch.where(valid, (p_std - t_std).abs(), torch.zeros_like(p_std))
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
    mask = torch.isfinite(target)
    n_valid = mask.sum()
    if n_valid.item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype,
                           requires_grad=pred.requires_grad)

    target_clean = torch.where(mask, target, torch.zeros_like(target))
    diff   = pred - target_clean
    mask_f = mask.to(pred.dtype)
    eps    = 1e-8

    if loss_type == "mse":
        base = (diff.pow(2) * mask_f).sum() / n_valid

    elif loss_type == "mae":
        base = (diff.abs() * mask_f).sum() / n_valid

    elif loss_type in ("huber", "smoothl1"):
        abs_diff = diff.abs()
        quad     = 0.5 * diff.pow(2)
        linear   = huber_beta * (abs_diff - 0.5 * huber_beta)
        per_elem = torch.where(abs_diff < huber_beta, quad, linear)
        base     = (per_elem * mask_f).sum() / n_valid

    elif loss_type == "nmse":
        mse_val      = (diff.pow(2) * mask_f).sum() / n_valid
        target_valid = target[mask]
        var          = target_valid.var(unbiased=False)
        base         = mse_val / (var + eps)

    elif loss_type == "nmae":
        mae_val      = (diff.abs() * mask_f).sum() / n_valid
        target_valid = target[mask]
        std          = target_valid.std(unbiased=False)
        base         = mae_val / (std + eps)

    elif loss_type == "gradient":
        base = (diff.pow(2) * mask_f).sum() / n_valid
        if pred.shape[-1] >= 2:
            target_d = target[..., 1:] - target[..., :-1]
            pred_d   = pred[..., 1:]  - pred[..., :-1]
            pair_ok  = torch.isfinite(target_d)
            n_pair   = pair_ok.sum().clamp(min=1)
            target_d_clean = torch.where(
                pair_ok, target_d, torch.zeros_like(target_d)
            )
            pair_mask_f = pair_ok.to(pred.dtype)
            grad_loss   = (
                (pred_d - target_d_clean).pow(2) * pair_mask_f
            ).sum() / n_pair
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
    """Closure capturing the CLI args; signature `(pred, target) -> tensor`."""
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nan_safe_loss(
            pred, target,
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
