# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Smoke tests for the paper figure package (synthetic bundles only)."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from analysis.figures import figures as F
from analysis.figures import tables as T
from analysis.figures.registry import Registry, discover_runs
from phenonn.evaluation import metrics as M


def _synthetic_predictions(n_sites=6, years=(2010, 2011), seed=0):
    """Build a small predictions frame with observations and climatology."""
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


def _bundle(root):
    """Write the minimal artefact set the figures consume."""
    pred = _synthetic_predictions()
    metrics = M.evaluate_predictions(pred, n_boot=0)
    out = Path(root) / "run_aelstm"
    out.mkdir()
    pred.to_csv(out / "predictions.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f)

    clim = pred.copy()
    clim["lai_pred"] = clim["lai_clim"]
    clim_metrics = M.evaluate_predictions(clim, n_boot=0)
    with open(out / "climatology_metrics.json", "w") as f:
        json.dump(
            {k: v for k, v in clim_metrics.items() if not k.startswith("_")},
            f,
        )

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
        json.dump({"type": "aelstm", "seed": 42, "protocol": "B"}, f)
    with open(out / "run_metadata.json", "w") as f:
        json.dump(
            {
                "stage": "train",
                "protocol": "B",
                "best_epoch": 2,
                "train_years": "1992-2006",
                "val_years": "2007-2009",
                "test_years": "2010-2019",
            },
            f,
        )
    with open(out / "predict_metadata.json", "w") as f:
        json.dump(
            {
                "stage": "predict",
                "predict_sites": "test",
                "predict_years": [2010, 2011],
                "n_sites": 6,
                "n_points": len(pred),
            },
            f,
        )
    return out, pred, metrics


class TestFigureFunctions(unittest.TestCase):
    """F1-F4 and F12 from a synthetic bundle."""

    def test_f1_f2_f3_f4_f12(self):
        """Each figure writes both a PDF and a PNG."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, pred, metrics = _bundle(root)
            F.plot_paper_pred_vs_obs(pred, metrics, str(root / "F1"))
            F.plot_paper_site_year_curves(pred, str(root / "F2"))
            F.plot_paper_anomaly_scatter(pred, metrics, str(root / "F3"))
            F.plot_paper_rmse_by_doy(
                pd.read_csv(out / "rmse_by_doy.csv"), str(root / "F4")
            )
            F.plot_paper_loss_curves(
                pd.read_csv(out / "history.csv"), str(root / "F12")
            )
            for name in ("F1", "F2", "F3", "F4", "F12"):
                with self.subTest(name=name):
                    self.assertTrue((root / f"{name}.pdf").exists())
                    self.assertTrue((root / f"{name}.png").exists())

    def test_f2_scores_and_composition(self):
        """F2 writes the score box and the top-PFT composition."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, pred, _ = _bundle(root)
            site = pred["site_id"].iloc[0]
            year = int(pred["year"].iloc[0])
            rows = []
            for pft in range(3):
                for doy in (5, 15, 25):
                    rows.append(
                        {
                            "site_id": site,
                            "year": year,
                            "pft": pft,
                            "pft_name": f"PFT{pft + 1}",
                            "frac": 0.4 - 0.1 * pft,
                            "doy": doy,
                        }
                    )
            F.plot_paper_site_year_curves(
                pred,
                str(root / "F2"),
                choices=[(site, year)],
                pft_df=pd.DataFrame(rows),
            )
            self.assertTrue((root / "F2.png").exists())

    def test_f12_multi_architecture(self):
        """F12 overlays architectures and averages multiple seeds."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hist = pd.DataFrame(
                {
                    "epoch": [1, 2, 3],
                    "train_loss": [1.0, 0.8, 0.7],
                    "val_loss": [1.1, 0.85, 0.9],
                    "is_best": [False, True, False],
                }
            )
            F.plot_paper_loss_curves(
                {"lstm": [hist], "aelstm": [hist, hist]},
                str(root / "F12"),
                early_stop_epochs={"lstm": 3},
            )
            self.assertTrue((root / "F12.png").exists())
            self.assertTrue((root / "F12.pdf").exists())

    def test_f8_f9_f13(self):
        """Ablation, perturbation and memory-kernel figures render."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            abl = pd.DataFrame(
                {
                    "group": ["temperature", "radiation", "vpd"],
                    "delta_r2": [0.01, 0.003, -0.001],
                    "ci_low": [0.005, 0.001, -0.003],
                    "ci_high": [0.015, 0.005, 0.001],
                }
            )
            F.plot_paper_ablation(abl, str(root / "F8"))
            frames = {
                "drought": pd.DataFrame(
                    {"doy": [5, 15, 25], "delta": [-0.1, -0.2, -0.1]}
                ),
                "warming": pd.DataFrame({"doy": [5, 15, 25], "delta": [0.1, 0.2, 0.1]}),
            }
            F.plot_paper_perturbation_curves(frames, str(root / "F9"))
            kernel = pd.DataFrame(
                {
                    "lag_days": [0, 10, 30],
                    "mean_abs_delta": [0.1, 0.05, 0.01],
                    "ci_low": [0.08, 0.04, 0.005],
                    "ci_high": [0.12, 0.06, 0.015],
                }
            )
            controls = pd.DataFrame({"control": ["random_day"], "value": [0.02]})
            F.plot_paper_memory_kernel(kernel, controls, str(root / "F13"))
            for name in ("F8", "F9", "F13"):
                with self.subTest(name=name):
                    self.assertTrue((root / f"{name}.png").exists())

    def test_f8_label_schema_and_f13_controls(self):
        """The aggregate's ``label`` column and all F13 controls work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            abl = pd.DataFrame(
                {
                    "label": ["temperature", "radiation"],
                    "delta_r2": [0.01, -0.002],
                }
            )
            F.plot_paper_ablation(abl, str(root / "F8"))
            table = T.ablation_table(abl, str(root / "F8.tex"))
            self.assertIn("Ablated group", table.columns)

            kernel = pd.DataFrame(
                {
                    "lag_days": [0, 10, 30],
                    "mean_abs_delta": [0.1, 0.05, 0.01],
                    "ci_low": [0.08, 0.04, 0.005],
                    "ci_high": [0.12, 0.06, 0.015],
                }
            )
            controls = pd.DataFrame(
                {
                    "control": [
                        "noop",
                        "random_day",
                        "random_day_p95",
                        "linearity",
                    ],
                    "value": [0.0, 0.02, 0.04, 2.0],
                }
            )
            F.plot_paper_memory_kernel(kernel, controls, str(root / "F13"))
            self.assertTrue((root / "F8.png").exists())
            self.assertTrue((root / "F13.png").exists())

    def test_f10_warming_and_f11_pft(self):
        """Warming/CO2 and per-PFT figures plus the F10 table render."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = pd.DataFrame(
                {
                    "scenario": ["+2C", "+2C+CO2"],
                    "delta_t": [2.0, 2.0],
                    "co2_mode": ["absolute", "doubling"],
                    "co2": ["+100 ppm", "2x"],
                    "mean_delta_lai": [0.1, 0.15],
                    "median_dsos_10": [-3.0, -2.0],
                    "median_deos_10": [2.0, 3.0],
                    "median_dlos_10": [5.0, 5.0],
                }
            )
            F.plot_paper_warming_co2(warm, str(root / "F10"))
            T.warming_co2_table(warm, str(root / "F10.tex"))

            rows = []
            for pft in range(15):
                for doy in (5, 15, 25):
                    rows.append(
                        {
                            "site_id": "pix_0001_00001",
                            "year": 2010,
                            "pft": pft,
                            "pft_name": f"PFT{pft + 1}",
                            "frac": 1.0 / 15.0,
                            "doy": doy,
                            "lai_pure": float(pft + 1) / 15.0,
                            "lai_pred": 0.5,
                            "lai_obs": 0.5,
                        }
                    )
            F.plot_paper_pft_curves(pd.DataFrame(rows), str(root / "F11"))

            for name in ("F10", "F11"):
                with self.subTest(name=name):
                    self.assertTrue((root / f"{name}.png").exists())
            self.assertTrue((root / "F10.tex").exists())


class TestRegistryAndTables(unittest.TestCase):
    """Run discovery and LaTeX tables."""

    def test_registry_and_tables(self):
        """A single run is discovered and both tables are written."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _bundle(root)
            runs = discover_runs(str(root))
            self.assertEqual(len(runs), 1)
            reg = Registry(runs)
            self.assertEqual(reg.get("aelstm").model_type, "aelstm")
            T.architecture_table(reg, str(root / "F5.tex"))
            T.model_vs_climatology_table(reg, str(root / "F6.tex"))
            T.anomaly_table(reg, str(root / "F7.tex"))
            self.assertTrue((root / "F5.tex").exists())
            self.assertTrue((root / "F6.tex").exists())
            self.assertTrue((root / "F7.tex").exists())

    def test_history_only_run_is_discovered(self):
        """A training-only run is available for F12 before prediction."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, _, _ = _bundle(root)
            (out / "metrics.json").unlink()
            (out / "climatology_metrics.json").unlink()

            runs = discover_runs(str(root))
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.model_type, "aelstm")
            self.assertFalse(run.metrics)
            self.assertFalse(run.history().empty)

    def test_climatology_row_and_predict_metadata(self):
        """F6 includes climatology and the bundle records its split."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _bundle(root)
            run = discover_runs(str(root))[0]
            self.assertEqual(run.prediction_sites, "test")
            self.assertEqual(run.prediction_years, [2010, 2011])
            self.assertEqual(run.best_epoch, 2)
            self.assertTrue(run.climatology)

            table = T.model_vs_climatology_table(Registry([run]), str(root / "F6.tex"))
            self.assertIn("Climatology", table["Model"].tolist())
            self.assertIn("Median site R²", table.columns)

    def test_architecture_seed_aggregation(self):
        """F5 collapses multiple seeds of one architecture into one row."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, _, _ = _bundle(root)
            second = root / "run_aelstm_seed43"
            shutil.copytree(out, second)
            with open(second / "config.json") as f:
                config = json.load(f)
            config["seed"] = 43
            with open(second / "config.json", "w") as f:
                json.dump(config, f)

            reg = Registry(discover_runs(str(root)))
            table = T.architecture_table(reg, str(root / "F5.tex"))
            self.assertEqual(len(table), 1)
            self.assertEqual(table.iloc[0]["Seeds"], "2")
            self.assertIn("$\\pm$", table.iloc[0]["R²"])

    def test_legacy_protocol_is_normalised(self):
        """Pre-protocol and uppercase metadata degrade gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, _, _ = _bundle(root)
            meta = out / "run_metadata.json"

            with open(meta, "w") as f:
                json.dump({"protocol": "NONE"}, f)
            self.assertEqual(discover_runs(str(root))[0].protocol, "none")

            with open(meta, "w") as f:
                json.dump({}, f)
            with open(out / "config.json", "w") as f:
                json.dump({"type": "aelstm", "seed": 42}, f)
            self.assertIsNone(discover_runs(str(root))[0].protocol)


class TestCompatibilityAliases(unittest.TestCase):
    """The old F-number names remain callable aliases."""

    def test_aliases_resolve(self):
        """Every deprecated alias points at its plot_paper_* replacement."""
        pairs = {
            "f1_pred_vs_obs": "plot_paper_pred_vs_obs",
            "f2_site_year_curves": "plot_paper_site_year_curves",
            "f3_anomaly_scatter": "plot_paper_anomaly_scatter",
            "f4_rmse_by_doy": "plot_paper_rmse_by_doy",
            "f8_ablation": "plot_paper_ablation",
            "f9_perturbation_curves": "plot_paper_perturbation_curves",
            "f10_warming_co2": "plot_paper_warming_co2",
            "f11_pft_curves": "plot_paper_pft_curves",
            "f12_loss_curves": "plot_paper_loss_curves",
            "f13_memory_kernel": "plot_paper_memory_kernel",
        }
        for old, new in pairs.items():
            with self.subTest(old=old):
                self.assertIs(getattr(F, old), getattr(F, new))


class TestMakeAll(unittest.TestCase):
    """``make_all`` can run optional figures without a prediction bundle."""

    def test_optional_input_without_registry(self):
        """F13 is produced even when ``--runs_dir`` has no ``metrics.json``."""
        from analysis.figures import make_all

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            mem = root / "mem"
            out = root / "out"
            runs.mkdir()
            mem.mkdir()
            pd.DataFrame(
                {
                    "lag_days": [0, 10, 30],
                    "mean_abs_delta": [0.1, 0.05, 0.01],
                    "ci_low": [0.08, 0.04, 0.005],
                    "ci_high": [0.12, 0.06, 0.015],
                }
            ).to_csv(mem / "memory_kernel.csv", index=False)
            pd.DataFrame(
                {"control": ["noop", "random_day"], "value": [0.0, 0.02]}
            ).to_csv(mem / "memory_controls.csv", index=False)

            argv = sys.argv
            sys.argv = [
                "make_all",
                "--runs_dir",
                str(runs),
                "--out_dir",
                str(out),
                "--figures",
                "F13",
                "--memory_dir",
                str(mem),
            ]
            try:
                make_all.main()
            finally:
                sys.argv = argv

            self.assertTrue((out / "F13_memory_kernel.png").exists())
            self.assertTrue((out / "F13_memory_kernel.pdf").exists())


class TestAnnotationsOutsideAxes(unittest.TestCase):
    """Metric boxes and scores are never drawn inside the plotting area."""

    def _capture(self, func, *args, **kwargs):
        captured = []
        with mock.patch.object(F, "savefig", lambda fig, out: captured.append(fig)):
            func(*args, **kwargs)
        return captured[0]

    def test_f1_metrics_are_a_footnote(self):
        """F1 puts R²/RMSE/... below the axes, not on the hexbin."""
        pred = pd.DataFrame(
            {"lai_obs": np.linspace(0, 6, 200), "lai_pred": np.linspace(0, 6, 200)}
        )
        metrics = {
            "overall_r2": 0.9,
            "rmse": 0.4,
            "mae": 0.3,
            "bias": -0.01,
            "slope": 0.9,
            "n_points": 200,
        }
        fig = self._capture(F.plot_paper_pred_vs_obs, pred, metrics, "F1")
        self.assertTrue(all(len(ax.texts) == 0 for ax in fig.axes))
        self.assertGreaterEqual(len(fig.texts), 1)

    def test_f2_composition_and_scores_are_in_the_title(self):
        """F2 keeps the PFT composition and scores out of the panels."""
        doy = np.arange(5, 366, 10)
        pred = pd.DataFrame(
            {
                "site_id": np.repeat(["pix_a", "pix_b"], doy.size),
                "year": np.repeat([2012, 2013], doy.size),
                "doy": np.tile(doy, 2),
                "lai_obs": 2.0 + np.sin(np.tile(doy, 2) / 58.0),
                "lai_pred": 2.0 + np.sin(np.tile(doy, 2) / 58.0 + 0.1),
                "lai_clim": 2.0 + np.sin(np.tile(doy, 2) / 58.0 + 0.2),
                "pft_dom_name": np.repeat(["grass", "crop"], doy.size),
            }
        )
        fig = self._capture(
            F.plot_paper_site_year_curves,
            pred,
            "F2",
            choices=[("pix_a", 2012), ("pix_b", 2013)],
        )
        self.assertTrue(all(len(ax.texts) == 0 for ax in fig.axes))
        self.assertTrue(all("\n" in ax.get_title() for ax in fig.axes))

    def test_f13_linearity_is_a_footnote(self):
        """F13 moves the linearity ratio out of the axes."""
        kernel = pd.DataFrame(
            {
                "lag_days": np.arange(0, 40, 5),
                "mean_abs_delta": np.linspace(0.3, 0.0, 8),
                "ci_low": np.linspace(0.2, 0.0, 8),
                "ci_high": np.linspace(0.4, 0.0, 8),
            }
        )
        controls = pd.DataFrame(
            {"control": ["linearity", "noop"], "value": [0.5, 0.01]}
        )
        fig = self._capture(F.plot_paper_memory_kernel, kernel, controls, "F13")
        self.assertTrue(all(len(ax.texts) == 0 for ax in fig.axes))
        self.assertGreaterEqual(len(fig.texts), 1)


class TestLegendStyle(unittest.TestCase):
    """Legends are outside the axes and use at least two columns."""

    def test_linestyle_cycle_with_markers(self):
        """markers=True must not pass None to mpltex's itertools.product."""
        from analysis.figures.ipsl_aid import MARKERS, linestyle_cycle

        cycle = linestyle_cycle(markers=True)
        styles = [next(cycle) for _ in range(2 * len(MARKERS))]
        self.assertTrue({s["marker"] for s in styles}.issubset(set(MARKERS)))
        self.assertGreaterEqual(len({s["marker"] for s in styles}), 4)
        self.assertGreaterEqual(len({s["linestyle"] for s in styles}), 3)

    def test_next_style_overrides_colour_and_marker(self):
        """The palette colour and marker win over the mpltex defaults."""
        from analysis.figures.ipsl_aid import linestyle_cycle, next_style

        style = next_style(linestyle_cycle(), color="#123456", marker="s")
        self.assertEqual(style["color"], "#123456")
        self.assertEqual(style["marker"], "s")
        self.assertIn("linestyle", style)

    def test_figure_legend_outside_and_multicolumn(self):
        """A figure legend must be multi-column and clear of the axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from analysis.figures.ipsl_aid import collect_handles, figure_legend

        fig, ax = plt.subplots(layout="constrained")
        ax.plot([0, 1], [0, 1], label="a")
        ax.plot([0, 1], [1, 0], "--", label="b")
        handles, labels = collect_handles([ax])
        legend = figure_legend(fig, handles, labels, ncol=3)
        fig.canvas.draw()

        self.assertIsNotNone(legend)
        self.assertGreaterEqual(legend._ncols, 2)
        self.assertFalse(legend.get_window_extent().overlaps(ax.get_window_extent()))
        self.assertIsNone(figure_legend(fig, [], []))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
