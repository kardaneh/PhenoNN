"""LaTeX (booktabs) tables generated from the run registry.

Tables are generated from CSV/JSON artefacts, never hand-edited, so every
number in the paper traces back to a run bundle (``PROTOCOL.md`` §9).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd

from figures.registry import Registry, Run

__all__ = [
    "to_latex",
    "architecture_table",
    "model_vs_climatology_table",
    "anomaly_table",
    "ablation_table",
    "warming_co2_table",
]


def _fmt(v, nd=4, signed=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if x != x:
        return "--"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _ci(v) -> str:
    if not v or len(v) != 2:
        return ""
    return f"[{_fmt(v[0])}, {_fmt(v[1])}]"


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


def _row(run: Run) -> dict:
    m = run.metrics
    ps = m.get("per_site", {}) or {}
    return {
        "Model": run.model_type or run.name,
        "Seed": run.seed if run.seed is not None else "--",
        "R²": _fmt(m.get("overall_r2")),
        "R² (centred)": _fmt(m.get("centered_r2")),
        "Pearson²": _fmt(m.get("pearson_r2")),
        "RMSE": _fmt(m.get("rmse")),
        "MAE": _fmt(m.get("mae")),
        "Bias": _fmt(m.get("bias"), signed=True),
        "Slope": _fmt(m.get("slope"), nd=3),
        "Site R² (med.)": _fmt(ps.get("median")),
        "% sites R²>0": _fmt(ps.get("pct_gt0"), nd=1),
    }


def architecture_table(registry: Registry, path: str) -> pd.DataFrame:
    """F5: one row per run with the architecture-comparison metrics."""
    rows = [_row(r) for r in registry if r.metrics]
    df = pd.DataFrame(rows)
    if not df.empty:
        to_latex(
            df,
            path,
            caption="Architecture comparison on Dataset 1 (Protocol B). "
            "Metrics are pooled over all test site-years; the best epoch is "
            "selected on validation loss only.",
            label="tab:architecture",
        )
    return df


def model_vs_climatology_table(registry: Registry, path: str) -> pd.DataFrame:
    """F6: all six Table-2 metrics for each model and the climatology."""
    rows = []
    for r in registry:
        if not r.metrics:
            continue
        m = r.metrics
        ps = m.get("per_site", {}) or {}
        sy = m.get("site_year", {}) or {}
        ia = m.get("interannual", {}) or {}
        rows.append(
            {
                "Model": r.model_type or r.name,
                "R²": _fmt(m.get("overall_r2")),
                "Centred R²": _fmt(m.get("centered_r2")),
                "Pearson²": _fmt(m.get("pearson_r2")),
                "RMSE": _fmt(m.get("rmse")),
                "Site-year R²": _fmt(sy.get("mean")),
                "Interannual R²": _fmt(ia.get("mean")),
                "% sites R²>0": _fmt(ps.get("pct_gt0"), nd=1),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        to_latex(
            df,
            path,
            caption="Model skill against the training-year climatology "
            "(Dataset 1, test decade). The climatology row is computed with "
            "the same code path on the same points.",
            label="tab:climatology",
        )
    return df


def anomaly_table(registry: Registry, path: str) -> pd.DataFrame:
    """F7: anomaly skill and the fraction of sites beating climatology."""
    rows = []
    for r in registry:
        an = r.metrics.get("anomaly") if r.metrics else None
        if not an:
            continue
        ps = an.get("per_site", {}) or {}
        rows.append(
            {
                "Model": r.model_type or r.name,
                "Anomaly R²": _fmt(an.get("overall_r2")),
                "Anomaly RMSE": _fmt(an.get("rmse")),
                "Anomaly MAE": _fmt(an.get("mae")),
                "Bias": _fmt(an.get("bias"), signed=True),
                "Site anomaly R² (med.)": _fmt(ps.get("median")),
                "% sites R²>0": _fmt(ps.get("pct_gt0"), nd=1),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        to_latex(
            df,
            path,
            caption="Anomaly skill against the training-year climatology. "
            "'% sites R²>0' is the fraction of sites where the model beats the "
            "climatology (R² > 0).",
            label="tab:anomaly",
        )
    return df


def ablation_table(
    df: pd.DataFrame,
    path: str,
    r2_col: str = "delta_r2",
    group_col: str = "group",
) -> pd.DataFrame:
    """F8 table: ΔR² per ablated feature group with bootstrap CI."""
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
            caption="Feature-group ablation (AELSTM, Huber loss, Protocol B). "
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
