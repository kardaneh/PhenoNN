"""Evaluation utilities for PhenoNN paper-grade runs.

The module is deliberately dependency-light (numpy + pandas) so that it can be
used from inference, training diagnostics, the figure package and the tests
without importing torch.

Metric definitions follow ``PROTOCOL.md`` §5.  All functions accept a tidy
``pandas.DataFrame`` with one row per ``(site, year, dekad)`` point and never
mutate their input.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "nse",
    "pooled_metrics",
    "evaluate_predictions",
    "per_site_metrics",
    "per_siteyear_metrics",
    "rmse_by_doy",
    "phenology_metrics",
    "sos_eos_curve",
    "table_row",
    "METRIC_NAMES",
]

METRIC_NAMES = (
    "overall_r2",
    "centered_r2",
    "pearson_r2",
    "rmse",
    "mae",
    "bias",
    "slope",
    "intercept",
)

_EPS = 1e-12


# ── Scalar metrics ──────────────────────────────────────────────────────────


def nse(obs, pred, nmin: int = 5) -> Optional[float]:
    """Nash–Sutcliffe efficiency, or ``None`` when it is ill-defined."""
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if o.size < nmin:
        return None
    sst = float(np.sum((o - o.mean()) ** 2))
    if sst <= _EPS:
        return None
    return 1.0 - float(np.sum((p - o) ** 2)) / sst


def _pooled_from_sums(
    n: np.ndarray,
    sum_o: np.ndarray,
    sum_p: np.ndarray,
    sum_o2: np.ndarray,
    sum_p2: np.ndarray,
    sum_op: np.ndarray,
    sum_e2: np.ndarray,
    sum_ae: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return pooled metrics from per-site sufficient statistics.

    All inputs are arrays broadcast against each other.  They may be scalars or
    the result of a bootstrap draw; the return values keep the same shape.
    """
    n = np.asarray(n, dtype=float)
    sum_o = np.asarray(sum_o, dtype=float)
    sum_p = np.asarray(sum_p, dtype=float)
    sum_o2 = np.asarray(sum_o2, dtype=float)
    sum_p2 = np.asarray(sum_p2, dtype=float)
    sum_op = np.asarray(sum_op, dtype=float)
    sum_e2 = np.asarray(sum_e2, dtype=float)
    sum_ae = np.asarray(sum_ae, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_o = sum_o / n
        mean_p = sum_p / n
        sst = sum_o2 - sum_o * sum_o / n
        ssv = sum_p2 - sum_p * sum_p / n
        cov = sum_op - sum_o * sum_p / n
        # A constant observation series has no variance to explain: R², slope
        # and Pearson² are undefined there.  Return NaN rather than dividing
        # by an epsilon floor, which would produce a huge negative number.
        sst_ok = sst > _EPS
        sst_safe = np.where(sst_ok, sst, 1.0)
        r2 = np.where(sst_ok, 1.0 - sum_e2 / sst_safe, np.nan)
        rmse = np.sqrt(sum_e2 / n)
        mae = sum_ae / n
        bias = (sum_p - sum_o) / n
        slope = np.where(sst_ok, cov / sst_safe, np.nan)
        intercept = np.where(sst_ok, mean_p - slope * mean_o, np.nan)
        pearson2 = np.where(
            sst_ok & (ssv > _EPS),
            cov * cov / np.maximum(sst * ssv, _EPS),
            np.nan,
        )

    return {
        "overall_r2": r2,
        "pearson_r2": pearson2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "slope": slope,
        "intercept": intercept,
    }


def pooled_metrics(df: pd.DataFrame, obs_col: str, pred_col: str) -> Dict[str, float]:
    """Pooled R²/RMSE/MAE/bias/slope/intercept for one prediction column."""
    sub = df[[obs_col, pred_col]].dropna()
    if sub.empty:
        return {k: float("nan") for k in METRIC_NAMES}
    o = sub[obs_col].to_numpy(float)
    p = sub[pred_col].to_numpy(float)
    e = p - o
    out = _pooled_from_sums(
        np.array([o.size]),
        np.array([o.sum()]),
        np.array([p.sum()]),
        np.array([np.dot(o, o)]),
        np.array([np.dot(p, p)]),
        np.array([np.dot(o, p)]),
        np.array([np.dot(e, e)]),
        np.array([np.abs(e).sum()]),
    )
    return {k: float(v[0]) for k, v in out.items()}


# ── Site-level sufficient statistics ───────────────────────────────────────


def _aggregate(df: pd.DataFrame, obs_col: str, pred_col: str, keys: Sequence[str]):
    """Per-key sums needed for every metric, plus the key columns."""
    d = pd.DataFrame(
        {
            **{k: df[k].to_numpy() for k in keys},
            "o": df[obs_col].to_numpy(float),
            "p": df[pred_col].to_numpy(float),
        }
    )
    d["o2"] = d["o"] * d["o"]
    d["p2"] = d["p"] * d["p"]
    d["op"] = d["o"] * d["p"]
    d["e2"] = (d["p"] - d["o"]) ** 2
    d["ae"] = (d["p"] - d["o"]).abs()
    g = d.groupby(list(keys), sort=False, observed=True)
    agg = g[["o", "p", "o2", "p2", "op", "e2", "ae"]].sum()
    agg.insert(0, "n", g.size())
    return agg.rename(
        columns={
            "o": "sum_o",
            "p": "sum_p",
            "o2": "sum_o2",
            "p2": "sum_p2",
            "op": "sum_op",
            "e2": "sum_e2",
            "ae": "sum_ae",
        }
    )


def _site_metrics_from_agg(agg: pd.DataFrame) -> pd.Series:
    """Per-site R² (NSE) from an aggregate table indexed by site."""
    n = agg["n"].to_numpy(float)
    so = agg["sum_o"].to_numpy(float)
    se2 = agg["sum_e2"].to_numpy(float)
    sst = agg["sum_o2"].to_numpy(float) - so * so / n
    r2 = np.full(len(agg), np.nan)
    ok = (n >= 5) & (sst > _EPS)
    r2[ok] = 1.0 - se2[ok] / sst[ok]
    return pd.Series(r2, index=agg.index, name="r2")


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "pct_gt0": float("nan"),
        }
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "pct_gt0": float(100.0 * np.mean(v > 0)),
    }


# ── Bootstrap ──────────────────────────────────────────────────────────────


def _bootstrap_pooled(
    agg: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    site_r2: Optional[np.ndarray] = None,
) -> Dict[str, list]:
    """Poisson block bootstrap over sites for the pooled metrics.

    Each bootstrap draw assigns ``Poisson(1)`` weights to sites, which is the
    standard Poisson bootstrap for i.i.d. clusters and is equivalent in the
    limit to resampling sites with replacement.  It avoids materialising a
    ``(n_boot, n_sites)`` index matrix, which matters for 10^6-site runs.
    """
    if n_boot <= 0 or len(agg) < 2:
        return {}
    rng = np.random.default_rng(seed)
    cols = ["n", "sum_o", "sum_p", "sum_o2", "sum_p2", "sum_op", "sum_e2", "sum_ae"]
    X = agg[cols].to_numpy(float)
    names = ["overall_r2", "pearson_r2", "rmse", "mae", "bias", "slope", "intercept"]
    draws = {k: np.empty(n_boot) for k in names}
    r2_mean = np.empty(n_boot) if site_r2 is not None else None

    for b in range(n_boot):
        w = rng.poisson(1.0, size=len(agg)).astype(float)
        s = w @ X
        vals = _pooled_from_sums(*[np.array([x]) for x in s])
        for k in names:
            draws[k][b] = float(vals[k][0])
        if r2_mean is not None:
            r2_mean[b] = float(np.sum(w * site_r2) / max(w.sum(), _EPS))

    out: Dict[str, list] = {}
    for k, arr in draws.items():
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[k] = [
                float(np.percentile(arr, 100 * alpha / 2)),
                float(np.percentile(arr, 100 * (1 - alpha / 2))),
            ]
    if r2_mean is not None:
        arr = r2_mean[np.isfinite(r2_mean)]
        if arr.size:
            out["per_site_mean_r2"] = [
                float(np.percentile(arr, 100 * alpha / 2)),
                float(np.percentile(arr, 100 * (1 - alpha / 2))),
            ]
    return out


# ── Public evaluation entry point ──────────────────────────────────────────


def evaluate_predictions(
    df: pd.DataFrame,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    site_col: str = "site_id",
    year_col: str = "year",
    doy_col: str = "doy",
    clim_col: Optional[str] = "lai_clim",
    n_boot: int = 1000,
    seed: int = 42,
    nmin_site: int = 5,
) -> Dict:
    """Compute the complete metric set on a tidy prediction table.

    Parameters
    ----------
    df : pandas.DataFrame
    obs_col, pred_col, site_col, year_col, doy_col : str
    clim_col : str or None
        Column holding the per-point climatology.  When present, anomaly
        metrics are computed as well.
    n_boot : int
        Number of bootstrap draws.  ``0`` disables the confidence intervals.
    seed : int
    nmin_site : int
        Minimum number of points for a per-site R².

    Returns
    -------
    dict
        A JSON-serialisable metric bundle.  ``ci95`` holds percentile
        confidence intervals for the pooled metrics.
    """
    required = [site_col, year_col, doy_col, obs_col, pred_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns for metrics: {missing}")

    cols = list(dict.fromkeys(required + ([clim_col] if clim_col else [])))
    sub = df[cols].dropna(subset=[obs_col, pred_col]).copy()
    if sub.empty:
        return {"n_points": 0, "n_sites": 0, "n_site_years": 0}

    site_agg = _aggregate(sub, obs_col, pred_col, [site_col])
    sy_agg = _aggregate(sub, obs_col, pred_col, [site_col, year_col])
    doy_agg = _aggregate(sub, obs_col, pred_col, [doy_col])

    totals = site_agg[
        ["n", "sum_o", "sum_p", "sum_o2", "sum_p2", "sum_op", "sum_e2", "sum_ae"]
    ].sum()
    pooled = _pooled_from_sums(*[np.array([float(v)]) for v in totals])
    pooled = {k: float(v[0]) for k, v in pooled.items()}

    # Centred R²: remove each site's mean from both series.
    n_s = site_agg["n"].to_numpy(float)
    d_s = site_agg["sum_p"].to_numpy(float) - site_agg["sum_o"].to_numpy(float)
    sst_c = float(
        np.sum(site_agg["sum_o2"].to_numpy(float) - site_agg["sum_o"].to_numpy(float) ** 2 / n_s)
    )
    ssr_c = float(np.sum(site_agg["sum_e2"].to_numpy(float) - d_s * d_s / n_s))
    pooled["centered_r2"] = 1.0 - ssr_c / sst_c if sst_c > _EPS else float("nan")

    site_r2 = _site_metrics_from_agg(site_agg).to_numpy()
    sy_r2 = _site_metrics_from_agg(sy_agg).to_numpy()
    ia_r2 = _site_metrics_from_agg(doy_agg).to_numpy()

    out: Dict = {
        "n_points": int(len(sub)),
        "n_sites": int(len(site_agg)),
        "n_site_years": int(len(sy_agg)),
        "n_doys": int(len(doy_agg)),
        **pooled,
        "per_site": _summary_stats(site_r2),
        "site_year": _summary_stats(sy_r2),
        "interannual": _summary_stats(ia_r2),
        "selection_rule": "lowest validation loss",
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "scheme": "poisson-sites"},
    }

    # ── Anomaly metrics ──
    if clim_col and clim_col in sub.columns:
        a = sub.dropna(subset=[clim_col])
        if not a.empty:
            oa = a[obs_col].to_numpy(float) - a[clim_col].to_numpy(float)
            pa = a[pred_col].to_numpy(float) - a[clim_col].to_numpy(float)
            e = pa - oa
            anom_site = pd.DataFrame(
                {
                    site_col: a[site_col].to_numpy(),
                    "o": oa,
                    "p": pa,
                }
            )
            anom_site["o2"] = anom_site["o"] ** 2
            anom_site["p2"] = anom_site["p"] ** 2
            anom_site["op"] = anom_site["o"] * anom_site["p"]
            anom_site["e2"] = (anom_site["p"] - anom_site["o"]) ** 2
            anom_site["ae"] = (anom_site["p"] - anom_site["o"]).abs()
            g = anom_site.groupby(site_col, sort=False, observed=True)
            aagg = g[["o", "p", "o2", "p2", "op", "e2", "ae"]].sum()
            aagg.insert(0, "n", g.size())
            aagg.columns = [
                "n",
                "sum_o",
                "sum_p",
                "sum_o2",
                "sum_p2",
                "sum_op",
                "sum_e2",
                "sum_ae",
            ]
            atot = aagg.sum()
            ap = _pooled_from_sums(*[np.array([float(v)]) for v in atot])
            ar2_site = _site_metrics_from_agg(aagg).to_numpy()
            out["anomaly"] = {
                "n_points": int(len(a)),
                "n_sites": int(len(aagg)),
                "overall_r2": float(ap["overall_r2"][0]),
                "rmse": float(ap["rmse"][0]),
                "mae": float(ap["mae"][0]),
                "bias": float(ap["bias"][0]),
                "per_site": _summary_stats(ar2_site),
            }
        else:
            out["anomaly"] = None

    # ── Bootstrap CIs ──
    out["ci95"] = _bootstrap_pooled(
        site_agg, n_boot=n_boot, seed=seed, site_r2=site_r2
    )

    # ── Per-site / per-site-year tables are attached for convenience ──
    out["_site_agg"] = site_agg
    out["_siteyear_agg"] = sy_agg
    return out


def per_site_metrics(
    df: pd.DataFrame,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    site_col: str = "site_id",
) -> pd.DataFrame:
    """Return one row per site with n, R², RMSE, MAE, bias and means."""
    sub = df[[site_col, obs_col, pred_col]].dropna()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                site_col,
                "n",
                "r2",
                "rmse",
                "mae",
                "bias",
                "obs_mean",
                "pred_mean",
            ]
        )
    agg = _aggregate(sub, obs_col, pred_col, [site_col])
    n = agg["n"].to_numpy(float)
    out = pd.DataFrame({site_col: agg.index.to_numpy()})
    out["n"] = n.astype(int)
    out["r2"] = _site_metrics_from_agg(agg).to_numpy()
    out["rmse"] = np.sqrt(agg["sum_e2"].to_numpy(float) / n)
    out["mae"] = agg["sum_ae"].to_numpy(float) / n
    out["bias"] = (agg["sum_p"].to_numpy(float) - agg["sum_o"].to_numpy(float)) / n
    out["obs_mean"] = agg["sum_o"].to_numpy(float) / n
    out["pred_mean"] = agg["sum_p"].to_numpy(float) / n
    return out.reset_index(drop=True)


def per_siteyear_metrics(
    df: pd.DataFrame,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    site_col: str = "site_id",
    year_col: str = "year",
) -> pd.DataFrame:
    """Return one row per site-year with n, R², RMSE, MAE and bias."""
    sub = df[[site_col, year_col, obs_col, pred_col]].dropna()
    if sub.empty:
        return pd.DataFrame(
            columns=[site_col, year_col, "n", "r2", "rmse", "mae", "bias"]
        )
    agg = _aggregate(sub, obs_col, pred_col, [site_col, year_col])
    n = agg["n"].to_numpy(float)
    out = pd.DataFrame(
        {
            site_col: agg.index.get_level_values(0).to_numpy(),
            year_col: agg.index.get_level_values(1).to_numpy(),
        }
    )
    out["n"] = n.astype(int)
    out["r2"] = _site_metrics_from_agg(agg).to_numpy()
    out["rmse"] = np.sqrt(agg["sum_e2"].to_numpy(float) / n)
    out["mae"] = agg["sum_ae"].to_numpy(float) / n
    out["bias"] = (agg["sum_p"].to_numpy(float) - agg["sum_o"].to_numpy(float)) / n
    return out.reset_index(drop=True)


def rmse_by_doy(
    df: pd.DataFrame,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    clim_col: Optional[str] = "lai_clim",
    doy_col: str = "doy",
) -> pd.DataFrame:
    """RMSE and bias per day-of-year for the model and the climatology."""
    rows = []
    for doy, g in df.groupby(doy_col, sort=True):
        o = g[obs_col].to_numpy(float)
        p = g[pred_col].to_numpy(float)
        m = np.isfinite(o) & np.isfinite(p)
        rec = {"doy": int(doy), "n": int(m.sum())}
        if m.any():
            rec["rmse_model"] = float(np.sqrt(np.mean((p[m] - o[m]) ** 2)))
            rec["bias_model"] = float(np.mean(p[m] - o[m]))
        else:
            rec["rmse_model"] = float("nan")
            rec["bias_model"] = float("nan")
        if clim_col and clim_col in g.columns:
            c = g[clim_col].to_numpy(float)
            mc = np.isfinite(o) & np.isfinite(c)
            if mc.any():
                rec["rmse_clim"] = float(np.sqrt(np.mean((c[mc] - o[mc]) ** 2)))
                rec["bias_clim"] = float(np.mean(c[mc] - o[mc]))
            else:
                rec["rmse_clim"] = float("nan")
                rec["bias_clim"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Phenology ──────────────────────────────────────────────────────────────


def sos_eos_curve(
    doy: np.ndarray,
    lai: np.ndarray,
    fractions: Iterable[float] = (0.10, 0.25),
) -> Dict[str, float]:
    """SOS/EOS/LOS for one seasonal curve.

    ``SOS`` is the first day of the rising phase at or above
    ``min + f·(max − min)``; ``EOS`` is the last day of the falling phase at or
    above the same threshold.  The curve is not interpolated.
    """
    doy = np.asarray(doy, dtype=float)
    lai = np.asarray(lai, dtype=float)
    m = np.isfinite(lai) & np.isfinite(doy)
    out: Dict[str, float] = {}
    for f in fractions:
        tag = f"{int(round(f * 100)):02d}"
        out[f"sos_{tag}"] = float("nan")
        out[f"eos_{tag}"] = float("nan")
        out[f"los_{tag}"] = float("nan")
    if m.sum() < 3:
        return out
    d, v = doy[m], lai[m]
    order = np.argsort(d)
    d, v = d[order], v[order]
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= _EPS:
        return out
    imax = int(np.nanargmax(v))
    for f in fractions:
        tag = f"{int(round(f * 100)):02d}"
        thr = lo + f * (hi - lo)
        rising = np.where(v[: imax + 1] >= thr)[0]
        falling = np.where(v[imax:] >= thr)[0]
        sos = float(d[rising[0]]) if rising.size else float("nan")
        eos = float(d[imax + falling[-1]]) if falling.size else float("nan")
        out[f"sos_{tag}"] = sos
        out[f"eos_{tag}"] = eos
        out[f"los_{tag}"] = eos - sos if np.isfinite(sos) and np.isfinite(eos) else float("nan")
    return out


def phenology_metrics(
    df: pd.DataFrame,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    site_col: str = "site_id",
    year_col: str = "year",
    doy_col: str = "doy",
) -> pd.DataFrame:
    """One row per site-year with observed and predicted SOS/EOS/LOS.

    Returns columns ``sos_10_obs``, ``sos_10_pred``, ``dsos_10`` … so the
    table can feed both F2/F10 and the phenology deltas of the perturbation
    study.
    """
    rows = []
    for (site, year), g in df.groupby([site_col, year_col], sort=False):
        o = sos_eos_curve(g[doy_col].to_numpy(), g[obs_col].to_numpy())
        p = sos_eos_curve(g[doy_col].to_numpy(), g[pred_col].to_numpy())
        rec = {site_col: site, year_col: int(year)}
        for k, v in o.items():
            rec[f"{k}_obs"] = v
        for k, v in p.items():
            rec[f"{k}_pred"] = v
        for k in o:
            rec[f"d{k}"] = (
                rec[f"{k}_pred"] - rec[f"{k}_obs"]
                if np.isfinite(rec[f"{k}_pred"]) and np.isfinite(rec[f"{k}_obs"])
                else float("nan")
            )
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Table helper ───────────────────────────────────────────────────────────


def table_row(name: str, m: Dict, n_boot: Optional[int] = None) -> Dict:
    """Flatten a metric bundle into one table row.

    Parameters
    ----------
    name : str
        Model or baseline label.
    m : dict
        Output of :func:`evaluate_predictions`.
    n_boot : int, optional
        Unused placeholder kept for signature symmetry with the figure code.
    """
    ci = m.get("ci95", {}) or {}
    ps = m.get("per_site", {}) or {}
    sy = m.get("site_year", {}) or {}
    ia = m.get("interannual", {}) or {}
    an = m.get("anomaly") or {}

    def ci_txt(key):
        if key in ci:
            lo, hi = ci[key]
            return f"[{lo:+.4f}, {hi:+.4f}]"
        return ""

    return {
        "model": name,
        "n_points": m.get("n_points"),
        "n_sites": m.get("n_sites"),
        "overall_r2": m.get("overall_r2"),
        "overall_r2_ci": ci_txt("overall_r2"),
        "centered_r2": m.get("centered_r2"),
        "pearson_r2": m.get("pearson_r2"),
        "rmse": m.get("rmse"),
        "rmse_ci": ci_txt("rmse"),
        "mae": m.get("mae"),
        "bias": m.get("bias"),
        "slope": m.get("slope"),
        "intercept": m.get("intercept"),
        "site_r2_median": ps.get("median"),
        "site_r2_mean": ps.get("mean"),
        "site_r2_std": ps.get("std"),
        "site_r2_p5": ps.get("p5"),
        "site_r2_p95": ps.get("p95"),
        "site_r2_pct_gt0": ps.get("pct_gt0"),
        "site_year_r2": sy.get("mean"),
        "site_year_r2_std": sy.get("std"),
        "interannual_r2": ia.get("mean"),
        "interannual_r2_std": ia.get("std"),
        "anomaly_r2": an.get("overall_r2"),
        "anomaly_rmse": an.get("rmse"),
        "anomaly_pct_sites_gt0": (an.get("per_site") or {}).get("pct_gt0"),
    }
