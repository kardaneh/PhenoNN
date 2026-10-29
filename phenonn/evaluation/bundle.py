"""Write the structured output bundle of a prediction run.

The bundle layout is fixed by ``PROTOCOL.md`` §7.  ``predict.py`` calls
:func:`write_prediction_bundle` once, after the predictions table has been
assembled; this module owns the file names, the JSON schema and the
climatology comparison so inference stays a thin orchestration layer.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from phenonn.evaluation import metrics as M
from phenonn.evaluation.reproducibility import jsonable, write_json

__all__ = [
    "attach_climatology",
    "attach_pft_columns",
    "write_prediction_bundle",
]

_PFT_FNAME = "PFTmap_{year}.nc"


# ── Column enrichment ──────────────────────────────────────────────────────


def attach_climatology(
    df: pd.DataFrame,
    clim_by_site: Optional[Dict[str, np.ndarray]],
    doy_col: str = "doy",
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
) -> pd.DataFrame:
    """Add ``lai_clim``, ``obs_anom`` and ``pred_anom`` columns in place.

    ``clim_by_site`` maps a site id to its 36-dekad climatology ordered like
    the canonical 5/15/25 calendar.  The row-to-dekad mapping uses ``doy`` when
    present and falls back to the row order inside each ``(site, year)``.
    """
    if not clim_by_site:
        return df
    import datetime

    obs_doys = [
        datetime.date(2001, m, d).timetuple().tm_yday
        for m in range(1, 13)
        for d in [5, 15, 25]
    ]
    doy_to_idx = {int(d): i for i, d in enumerate(obs_doys)}
    if doy_col in df.columns:
        dek = df[doy_col].map(doy_to_idx).to_numpy()
    else:
        dek = df.groupby(["site_id", "year"], sort=False).cumcount().to_numpy()
    clim_vals = np.full(len(df), np.nan, dtype=np.float32)
    site_arr = df["site_id"].to_numpy()
    # Vectorise the lookup: build a (n_sites, 36) matrix once.
    uniq, inv = np.unique(site_arr, return_inverse=True)
    mat = np.full((len(uniq), 36), np.nan, dtype=np.float32)
    for i, s in enumerate(uniq):
        c = clim_by_site.get(str(s))
        if c is not None:
            c = np.asarray(c, dtype=np.float32).ravel()
            mat[i, : min(36, c.size)] = c[:36]
    ok = np.isfinite(dek) & (dek < 36)
    dek = np.where(ok, dek, 0).astype(int)
    clim_vals[ok] = mat[inv[ok], dek[ok]]
    df["lai_clim"] = clim_vals
    df["obs_anom"] = df[obs_col].to_numpy(float) - clim_vals
    df["pred_anom"] = df[pred_col].to_numpy(float) - clim_vals
    return df


def _read_pft_year(
    pft_dir: str, year: int, wanted: Iterable[str]
) -> Dict[str, np.ndarray]:
    """Read ``PFTmap_{year}.nc`` and return fractions for the wanted sites."""
    import xarray as xr

    path = os.path.join(pft_dir, _PFT_FNAME.format(year=year))
    if not os.path.exists(path):
        return {}
    wanted = {str(s) for s in wanted}
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        da = ds["pft_frac"] if "pft_frac" in ds.data_vars else ds[list(ds.data_vars)[0]]
        sites = np.asarray(ds["site_id"].values).astype(str)
        arr = da.transpose("pft", "site").values.astype(np.float32)
    idx = {s: i for i, s in enumerate(sites)}
    return {s: arr[:, idx[s]] for s in wanted if s in idx}


def attach_pft_columns(
    df: pd.DataFrame,
    pft_dir: str,
    pft_names: Optional[List[str]] = None,
    year_col: str = "year",
) -> pd.DataFrame:
    """Add ``pft_dom``, ``pft_dom_name``, ``bare_frac`` and ``purity``.

    Fractions are read per year and reduced to four scalar columns immediately,
    so the 15-column PFT block is never materialised for the full table.
    """
    if not pft_dir or not os.path.isdir(pft_dir):
        return df
    if pft_names is None:
        from phenonn.utils.config import PFT_NAMES

        pft_names = PFT_NAMES
    dom = np.full(len(df), -1, dtype=np.int16)
    bare = np.full(len(df), np.nan, dtype=np.float32)
    purity = np.full(len(df), np.nan, dtype=np.float32)
    for year, positions in df.groupby(year_col, sort=False).indices.items():
        fracs = _read_pft_year(pft_dir, int(year), df["site_id"].to_numpy()[positions])
        if not fracs:
            continue
        sites = df["site_id"].to_numpy()[positions]
        d = np.array([int(np.argmax(fracs[s])) if s in fracs else -1 for s in sites])
        b = np.array(
            [fracs[s][0] if s in fracs and fracs[s].size else np.nan for s in sites]
        )
        p = np.array(
            [
                float(np.max(fracs[s])) if s in fracs and fracs[s].size else np.nan
                for s in sites
            ]
        )
        dom[positions] = d
        bare[positions] = b
        purity[positions] = p
    df["pft_dom"] = dom
    names = np.array(list(pft_names) + ["unknown"], dtype=object)
    safe = np.where(dom >= 0, dom, len(pft_names))
    df["pft_dom_name"] = names[safe]
    df["bare_frac"] = bare
    df["purity"] = purity
    return df


# ── Bundle writer ──────────────────────────────────────────────────────────


def _strip_private(m: Dict) -> Dict:
    return {k: v for k, v in m.items() if not str(k).startswith("_")}


def _link_or_copy(src: str, dst: str) -> None:
    """Create ``dst`` as a hard link to ``src`` when possible, else copy."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def write_prediction_bundle(
    df: pd.DataFrame,
    valid: pd.DataFrame,
    out_dir: str,
    predictions_file: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    obs_col: str = "lai_obs",
    pred_col: str = "lai_pred",
    clim_col: Optional[str] = "lai_clim",
    checkpoint_meta: Optional[Dict] = None,
    run_metadata: Optional[Dict] = None,
    metadata_name: str = "run_metadata.json",
    write_phenology: bool = True,
    pheno_max_siteyears: int = 0,
    logger=None,
) -> Dict:
    """Compute and write every artefact of the prediction bundle.

    Returns the model metric dictionary (private aggregate tables removed).
    """
    os.makedirs(out_dir, exist_ok=True)
    if valid is None or valid.empty:
        if logger:
            logger.warning("Empty prediction table — bundle metrics skipped.")
        return {}

    model_m = M.evaluate_predictions(
        valid,
        obs_col=obs_col,
        pred_col=pred_col,
        clim_col=clim_col,
        n_boot=n_boot,
        seed=seed,
    )
    clim_m = None
    if clim_col and clim_col in valid.columns and valid[clim_col].notna().any():
        clim_m = M.evaluate_predictions(
            valid,
            obs_col=obs_col,
            pred_col=clim_col,
            clim_col=None,
            n_boot=n_boot,
            seed=seed,
        )

    model_out = _strip_private(model_m)
    model_out["checkpoint"] = jsonable(checkpoint_meta or {})
    model_out["predictions_file"] = os.path.abspath(predictions_file)
    write_json(os.path.join(out_dir, "metrics.json"), model_out)
    write_json(
        os.path.join(out_dir, "metrics_at_best_loss.json"),
        {
            **(checkpoint_meta or {}),
            "selection_rule": "lowest validation loss",
            "metrics": model_out,
        },
    )
    if checkpoint_meta and checkpoint_meta.get("best_r2_epoch"):
        write_json(
            os.path.join(out_dir, "metrics_at_best_r2.json"),
            {
                **(checkpoint_meta or {}),
                "selection_rule": "highest validation R² (diagnostic only)",
                "metrics": model_out,
            },
        )
    if clim_m is not None:
        write_json(
            os.path.join(out_dir, "climatology_metrics.json"),
            _strip_private(clim_m),
        )

    # ── Tables ──
    ps = M.per_site_metrics(valid, obs_col, pred_col)
    ps.to_csv(os.path.join(out_dir, "per_site_metrics.csv"), index=False)
    psy = M.per_siteyear_metrics(valid, obs_col, pred_col)
    psy.to_csv(os.path.join(out_dir, "per_siteyear_metrics.csv"), index=False)
    M.rmse_by_doy(valid, obs_col, pred_col, clim_col).to_csv(
        os.path.join(out_dir, "rmse_by_doy.csv"), index=False
    )

    if write_phenology:
        ph_valid = valid
        n_sy = valid.groupby(["site_id", "year"], sort=False).ngroups
        if pheno_max_siteyears and n_sy > pheno_max_siteyears:
            sites = np.array(sorted(valid["site_id"].unique()))
            keep_n = max(
                1,
                int(len(sites) * pheno_max_siteyears / max(n_sy, 1)),
            )
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(sites, size=min(keep_n, len(sites)), replace=False))
            ph_valid = valid[valid["site_id"].isin(keep)]
            if logger:
                logger.warning(
                    f"phenology metrics on {len(keep):,} / {len(sites):,} sites "
                    f"(cap {pheno_max_siteyears:,} site-years)"
                )
        ph = M.phenology_metrics(ph_valid, obs_col, pred_col)
        ph.to_csv(os.path.join(out_dir, "pheno_metrics.csv"), index=False)

    rows = [M.table_row("model", model_out)]
    if clim_m is not None:
        rows.append(M.table_row("climatology", _strip_private(clim_m)))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False)

    # ── Predictions file inside the bundle ──
    _link_or_copy(predictions_file, os.path.join(out_dir, "predictions.csv"))

    if run_metadata:
        write_json(os.path.join(out_dir, metadata_name), run_metadata)

    if logger:
        r2 = model_out.get("overall_r2", float("nan"))
        rmse = model_out.get("rmse", float("nan"))
        logger.success(f"Bundle        : {out_dir}  (R²={r2:+.4f}, RMSE={rmse:.4f})")
    return model_out
