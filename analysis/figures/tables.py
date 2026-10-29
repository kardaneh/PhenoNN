"""LaTeX (booktabs) tables generated from the run registry.

Tables are generated from CSV/JSON artefacts, never hand-edited, so every
number in the paper traces back to a run bundle (``PROTOCOL.md`` §9).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from analysis.figures.registry import Registry, Run

__all__ = [
    "to_latex",
    "architecture_table",
    "model_vs_climatology_table",
    "anomaly_table",
    "ablation_table",
    "warming_co2_table",
]

_MODEL_ORDER = ("lstm", "bitransformer", "aelstm", "attnlstm")


def _fmt(v, nd=4, signed=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if x != x:
        return "--"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _fmt_ms(values, nd=4, signed=False):
    """Format a seed spread as ``mean ± std`` (LaTeX-safe)."""
    if not values:
        return "--"
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:+.{nd}f}" if signed else f"{mean:.{nd}f}"
    std = float(np.std(values, ddof=1))
    return (
        f"{mean:+.{nd}f} $\\pm$ {std:.{nd}f}"
        if signed
        else (f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}")
    )


def _ci(v) -> str:
    if not v or len(v) != 2:
        return ""
    return f"[{_fmt(v[0])}, {_fmt(v[1])}]"


def _model_order(label: str):
    """Stable paper order: LSTM, BiTransformer, AELSTM, then the rest."""
    try:
        return (_MODEL_ORDER.index(label), label)
    except ValueError:
        return (len(_MODEL_ORDER), label)


def _group_runs(registry: Registry):
    """Group runs by architecture label, preserving the paper order."""
    groups = {}
    for run in registry:
        if not run.metrics:
            continue
        groups.setdefault(run.model_label, []).append(run)
    return {label: groups[label] for label in sorted(groups, key=_model_order)}


def _metric(run: Run, key: str):
    metrics = run.metrics or {}
    if key == "median_site_r2":
        return (metrics.get("per_site") or {}).get("median")
    if key == "pct_site_r2":
        return (metrics.get("per_site") or {}).get("pct_gt0")
    return metrics.get(key)


def _anomaly_metric(run: Run, key: str):
    anomaly = (run.metrics or {}).get("anomaly") or {}
    if key == "median_site_anomaly_r2":
        return (anomaly.get("per_site") or {}).get("median")
    if key == "pct_sites":
        return (anomaly.get("per_site") or {}).get("pct_gt0")
    return anomaly.get(key)


def _values(runs, getter):
    """Finite numeric values for one metric across a list of runs."""
    out = []
    for run in runs:
        try:
            value = float(getter(run))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def _seeds(runs) -> str:
    seeds = sorted({run.seed for run in runs if run.seed is not None})
    if not seeds:
        return "--"
    if len(seeds) == 1:
        return str(seeds[0])
    return str(len(seeds))


def _best_epoch(runs) -> str:
    epochs = [run.best_epoch for run in runs if run.best_epoch is not None]
    if not epochs:
        return "--"
    return str(int(round(float(np.mean(epochs)))))


def to_latex(
    df: pd.DataFrame,
    path: str,
    caption: str = "",
    label: str = "",
    column_format: Optional[str] = None,
) -> str:
    """Write a booktabs LaTeX table and return the LaTeX string."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tex = df.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        column_format=column_format,
        caption=caption or None,
        label=label or None,
    )
    # This pandas version does not expose ``booktabs`` on ``DataFrame.to_latex``;
    # convert the horizontal rules so the output uses booktabs in the paper.
    if "\\hline" in tex:
        parts = tex.split("\\hline")
        rebuilt = [parts[0]]
        for i, part in enumerate(parts[1:]):
            rule = (
                "\\toprule"
                if i == 0
                else ("\\bottomrule" if i == len(parts) - 2 else "\\midrule")
            )
            rebuilt.append(rule + part)
        tex = "".join(rebuilt)
    with open(path, "w") as f:
        f.write(tex)
    return tex


def architecture_table(registry: Registry, path: str) -> pd.DataFrame:
    """F5: one row per architecture, averaged over seeds when available."""
    rows = []
    for label, runs in _group_runs(registry).items():
        rows.append(
            {
                "Model": label,
                "Seeds": _seeds(runs),
                "R²": _fmt_ms(_values(runs, lambda r: _metric(r, "overall_r2"))),
                "R² (centred)": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "centered_r2"))
                ),
                "Pearson²": _fmt_ms(_values(runs, lambda r: _metric(r, "pearson_r2"))),
                "RMSE": _fmt_ms(_values(runs, lambda r: _metric(r, "rmse"))),
                "MAE": _fmt_ms(_values(runs, lambda r: _metric(r, "mae"))),
                "Bias": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "bias")), signed=True
                ),
                "Slope": _fmt_ms(_values(runs, lambda r: _metric(r, "slope")), nd=3),
                "Site R² (med.)": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "median_site_r2"))
                ),
                "% sites R²>0": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "pct_site_r2")), nd=1
                ),
                "Best epoch": _best_epoch(runs),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        to_latex(
            df,
            path,
            caption="Architecture comparison on Dataset 1 (Protocol B). "
            "Metrics are pooled over all test site-years; the best epoch is "
            "selected on validation loss only. Mean $\\pm$ std across seeds "
            "where more than one run is available.",
            label="tab:architecture",
        )
    return df


def model_vs_climatology_table(registry: Registry, path: str) -> pd.DataFrame:
    """F6: the six paper metrics for each model and the climatology."""
    rows = []
    for label, runs in _group_runs(registry).items():
        rows.append(
            {
                "Model": label,
                "Seeds": _seeds(runs),
                "R²": _fmt_ms(_values(runs, lambda r: _metric(r, "overall_r2"))),
                "Centred R²": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "centered_r2"))
                ),
                "Pearson²": _fmt_ms(_values(runs, lambda r: _metric(r, "pearson_r2"))),
                "RMSE": _fmt_ms(_values(runs, lambda r: _metric(r, "rmse"))),
                "Median site R²": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "median_site_r2"))
                ),
                "% sites R²>0": _fmt_ms(
                    _values(runs, lambda r: _metric(r, "pct_site_r2")), nd=1
                ),
            }
        )

    climatology = None
    for run in registry:
        if run.climatology:
            climatology = run.climatology
            break
    if climatology:
        per_site = climatology.get("per_site") or {}
        rows.append(
            {
                "Model": "Climatology",
                "Seeds": "--",
                "R²": _fmt(climatology.get("overall_r2")),
                "Centred R²": _fmt(climatology.get("centered_r2")),
                "Pearson²": _fmt(climatology.get("pearson_r2")),
                "RMSE": _fmt(climatology.get("rmse")),
                "Median site R²": _fmt(per_site.get("median")),
                "% sites R²>0": _fmt(per_site.get("pct_gt0"), nd=1),
            }
        )
    else:
        print("[warn] F6: no climatology_metrics.json found; no climatology row")

    df = pd.DataFrame(rows)
    if not df.empty:
        to_latex(
            df,
            path,
            caption="Model skill against the training-year climatology "
            "(Dataset 1, test decade). The climatology row is computed with "
            "the same code path on the same points. Mean $\\pm$ std across "
            "seeds where more than one run is available.",
            label="tab:climatology",
        )
    return df


def anomaly_table(registry: Registry, path: str) -> pd.DataFrame:
    """F7: anomaly skill and the fraction of sites beating climatology."""
    rows = []
    for label, runs in _group_runs(registry).items():
        anomaly_runs = [r for r in runs if (r.metrics or {}).get("anomaly")]
        if not anomaly_runs:
            continue
        rows.append(
            {
                "Model": label,
                "Seeds": _seeds(anomaly_runs),
                "Anomaly R²": _fmt_ms(
                    _values(anomaly_runs, lambda r: _anomaly_metric(r, "overall_r2"))
                ),
                "Anomaly RMSE": _fmt_ms(
                    _values(anomaly_runs, lambda r: _anomaly_metric(r, "rmse"))
                ),
                "Median site anomaly R²": _fmt_ms(
                    _values(
                        anomaly_runs,
                        lambda r: _anomaly_metric(r, "median_site_anomaly_r2"),
                    )
                ),
                "% sites beating climatology": _fmt_ms(
                    _values(anomaly_runs, lambda r: _anomaly_metric(r, "pct_sites")),
                    nd=1,
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        to_latex(
            df,
            path,
            caption="Anomaly skill against the training-year climatology. "
            "The last column is the fraction of sites with anomaly R² > 0, "
            "i.e. sites where the model beats the climatology. Mean $\\pm$ "
            "std across seeds where more than one run is available.",
            label="tab:anomaly",
        )
    return df


def ablation_table(
    df: pd.DataFrame,
    path: str,
    r2_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """F8 table: ΔR² per ablated feature group with bootstrap CI.

    The aggregate script writes ``label`` and ``delta_r2``; older hand-made
    tables may use ``group`` and ``dR2``.  Both spellings are accepted.
    """
    if r2_col is None:
        r2_col = "delta_r2" if "delta_r2" in df.columns else "dR2"
    if group_col is None:
        group_col = "group" if "group" in df.columns else "label"
    missing = [name for name in (r2_col, group_col) if name not in df.columns]
    if missing:
        raise ValueError(
            f"ablation table needs columns {missing}; got {list(df.columns)}"
        )

    rows = []
    for _, r in df.sort_values(r2_col, ascending=False).iterrows():
        rows.append(
            {
                "Ablated group": r[group_col],
                "ΔR²": _fmt(r.get(r2_col), signed=True),
                "95% CI": _ci(
                    [r.get("ci_low"), r.get("ci_high")]
                    if "ci_low" in r and "ci_high" in r
                    else None
                ),
                "ΔRMSE": _fmt(r.get("delta_rmse"), signed=True),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        to_latex(
            out,
            path,
            caption="Feature-group ablation (AELSTM, MSE loss, Protocol B). "
            "ΔR² is the full-model R² minus the ablated-model R².",
            label="tab:ablation",
        )
    return out


def warming_co2_table(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """F10 table: phenology response to warming and CO2 scenarios."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Scenario": r.get("scenario", ""),
                "ΔT (°C)": _fmt(r.get("delta_t"), nd=1, signed=True),
                "CO₂": r.get("co2", ""),
                "ΔLAI": _fmt(r.get("mean_delta_lai"), signed=True),
                "ΔSOS (d)": _fmt(r.get("median_dsos_10"), nd=1, signed=True),
                "ΔEOS (d)": _fmt(r.get("median_deos_10"), nd=1, signed=True),
                "ΔLOS (d)": _fmt(r.get("median_dlos_10"), nd=1, signed=True),
                "Flag": r.get("flag", ""),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        to_latex(
            out,
            path,
            caption="Phenological response to warming and CO₂. Rows flagged "
            "``out-of-range`` extrapolate beyond the training CO₂ range.",
            label="tab:warming_co2",
        )
    return out
