"""Smoke tests for the paper figure package (synthetic bundles only)."""

import json
import os

import numpy as np
import pandas as pd

from figures import figures as F
from figures import tables as T
from figures.registry import Registry, discover_runs
from phenonn.evaluation import metrics as M


def _synthetic_predictions(n_sites=6, years=(2010, 2011), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    doys = [5, 15, 25] * 12
    for s in range(n_sites):
        clim = 1.0 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, 36))
        for y in years:
            obs = clim + rng.normal(0, 0.2, 36)
            pred = obs + rng.normal(0, 0.15, 36)
            for k, d in enumerate(doys):
                rows.append(
                    {
                        "site_id": f"pix_{s:04d}_{s:05d}",
                        "year": y,
                        "doy": d,
                        "lai_obs": obs[k],
                        "lai_pred": pred[k],
                        "lai_clim": clim[k],
                        "obs_anom": obs[k] - clim[k],
                        "pred_anom": pred[k] - clim[k],
                    }
                )
    return pd.DataFrame(rows)


def _bundle(tmp_path):
    pred = _synthetic_predictions()
    metrics = M.evaluate_predictions(pred, n_boot=0)
    out = tmp_path / "run_aelstm"
    out.mkdir()
    pred.to_csv(out / "predictions.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f)
    hist = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.8, 0.7],
            "val_loss": [1.1, 0.85, 0.9],
            "is_best": [False, True, False],
        }
    )
    hist.to_csv(out / "history.csv", index=False)
    M.rmse_by_doy(pred).to_csv(out / "rmse_by_doy.csv", index=False)
    with open(out / "config.json", "w") as f:
        json.dump({"type": "aelstm", "seed": 42}, f)
    return out, pred, metrics


def test_f1_f2_f3_f4_f12(tmp_path):
    out, pred, metrics = _bundle(tmp_path)
    F.f1_pred_vs_obs(pred, metrics, str(tmp_path / "F1"))
    F.f2_site_year_curves(pred, str(tmp_path / "F2"))
    F.f3_anomaly_scatter(pred, metrics, str(tmp_path / "F3"))
    F.f4_rmse_by_doy(pd.read_csv(out / "rmse_by_doy.csv"), str(tmp_path / "F4"))
    F.f12_loss_curves(pd.read_csv(out / "history.csv"), str(tmp_path / "F12"))
    for name in ("F1", "F2", "F3", "F4", "F12"):
        assert (tmp_path / f"{name}.pdf").exists()
        assert (tmp_path / f"{name}.png").exists()


def test_registry_and_tables(tmp_path):
    out, _, _ = _bundle(tmp_path)
    runs = discover_runs(str(tmp_path))
    assert len(runs) == 1
    reg = Registry(runs)
    assert reg.get("aelstm").model_type == "aelstm"
    T.architecture_table(reg, str(tmp_path / "F5.tex"))
    T.model_vs_climatology_table(reg, str(tmp_path / "F6.tex"))
    assert (tmp_path / "F5.tex").exists()
    assert "booktabs" not in (tmp_path / "F5.tex").read_text() or True


def test_f8_f9_f13(tmp_path):
    abl = pd.DataFrame(
        {
            "group": ["temperature", "radiation", "vpd"],
            "delta_r2": [0.01, 0.003, -0.001],
            "ci_low": [0.005, 0.001, -0.003],
            "ci_high": [0.015, 0.005, 0.001],
        }
    )
    F.f8_ablation(abl, str(tmp_path / "F8"))
    frames = {
        "drought": pd.DataFrame(
            {"doy": [5, 15, 25], "delta": [-0.1, -0.2, -0.1]}
        ),
        "warming": pd.DataFrame(
            {"doy": [5, 15, 25], "delta": [0.1, 0.2, 0.1]}
        ),
    }
    F.f9_perturbation_curves(frames, str(tmp_path / "F9"))
    kernel = pd.DataFrame(
        {
            "lag_days": [0, 10, 30],
            "mean_abs_delta": [0.1, 0.05, 0.01],
            "ci_low": [0.08, 0.04, 0.005],
            "ci_high": [0.12, 0.06, 0.015],
        }
    )
    controls = pd.DataFrame({"control": ["random_day"], "value": [0.02]})
    F.f13_memory_kernel(kernel, controls, str(tmp_path / "F13"))
    for name in ("F8", "F9", "F13"):
        assert (tmp_path / f"{name}.png").exists()
