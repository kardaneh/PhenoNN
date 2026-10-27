"""Unit tests for :mod:`phenonn.evaluation.metrics`."""

import datetime

import numpy as np
import pandas as pd

from phenonn.evaluation import metrics as M


def _frame(site_means=(1.0, 3.0), n_years=2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # Canonical 36-dekad calendar: days 5/15/25 of every month.
    doys = [
        datetime.date(2001, month, day).timetuple().tm_yday
        for month in range(1, 13)
        for day in (5, 15, 25)
    ]
    for s, mu in enumerate(site_means):
        clim = mu + np.sin(np.linspace(0, 2 * np.pi, 36))
        for y in range(2010, 2010 + n_years):
            obs = clim + rng.normal(0, 0.2, 36)
            pred = obs + rng.normal(0, noise, 36)
            for k, d in enumerate(doys):
                rows.append(
                    {
                        "site_id": f"s{s}",
                        "year": y,
                        "doy": d,
                        "lai_obs": obs[k],
                        "lai_pred": pred[k],
                        "lai_clim": clim[k],
                    }
                )
    return pd.DataFrame(rows)


def test_perfect_prediction():
    df = _frame(noise=0.0)
    m = M.evaluate_predictions(df, n_boot=0)
    assert abs(m["overall_r2"] - 1.0) < 1e-10
    assert m["rmse"] < 1e-10
    assert abs(m["bias"]) < 1e-10
    assert abs(m["slope"] - 1.0) < 1e-10
    assert abs(m["centered_r2"] - 1.0) < 1e-10
    assert abs(m["pearson_r2"] - 1.0) < 1e-10
    assert m["per_site"]["pct_gt0"] == 100.0


def test_centred_r2_removes_site_means():
    # Between-site offset only: pooled R² is poor, centred R² is perfect.
    df = _frame(noise=0.0)
    df["lai_pred"] = df["lai_pred"] + df["site_id"].map({"s0": -1.0, "s1": 1.0})
    m = M.evaluate_predictions(df, n_boot=0)
    assert m["overall_r2"] < 0.9
    assert m["centered_r2"] > 0.999


def test_constant_observation_is_nan():
    df = _frame(noise=0.1)
    df["lai_obs"] = 1.0
    m = M.evaluate_predictions(df, n_boot=0)
    assert np.isnan(m["overall_r2"])
    assert np.isnan(m["centered_r2"])
    assert m["rmse"] >= 0


def test_anomaly_metrics():
    df = _frame(noise=0.0)
    df["lai_pred"] = df["lai_clim"]
    m = M.evaluate_predictions(df, n_boot=0)
    assert m["anomaly"] is not None
    # A climatology prediction has zero anomaly skill on noisy observations.
    assert m["anomaly"]["overall_r2"] <= 1e-6


def test_per_site_and_doy_tables():
    df = _frame(noise=0.1)
    ps = M.per_site_metrics(df)
    assert {"site_id", "n", "r2", "rmse", "mae", "bias"}.issubset(ps.columns)
    assert len(ps) == 2
    psy = M.per_siteyear_metrics(df)
    assert len(psy) == 4
    rd = M.rmse_by_doy(df)
    assert len(rd) == 36
    assert "rmse_clim" in rd.columns


def test_bootstrap_is_reproducible_and_ordered():
    df = _frame(noise=0.15, n_years=3)
    a = M.evaluate_predictions(df, n_boot=40, seed=7)["ci95"]
    b = M.evaluate_predictions(df, n_boot=40, seed=7)["ci95"]
    assert a == b
    lo, hi = a["overall_r2"]
    assert lo <= hi
    m = M.evaluate_predictions(df, n_boot=0)
    assert lo <= m["overall_r2"] + 1e-6
    assert m["overall_r2"] - 1e-6 <= hi


def test_phenology_sos_eos():
    doy = np.array([5, 15, 25, 35, 45, 55, 65, 75])
    lai = np.array([0.0, 0.0, 0.2, 1.0, 2.0, 1.0, 0.2, 0.0])
    out = M.sos_eos_curve(doy, lai, fractions=(0.25,))
    assert out["sos_25"] == 35.0
    assert out["eos_25"] == 55.0
    assert out["los_25"] == 20.0


def test_jsonable_sanitises_non_finite(tmp_path):
    import json

    from phenonn.evaluation.reproducibility import jsonable, write_json

    payload = {"a": float("nan"), "b": np.float32("inf"), "c": [1.0, np.nan]}
    clean = jsonable(payload)
    assert clean["a"] is None
    assert clean["b"] is None
    assert clean["c"][1] is None

    path = tmp_path / "meta.json"
    write_json(str(path), payload)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    with open(path) as f:
        assert json.load(f)["a"] is None

