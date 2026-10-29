# Copyright 2026 CNRS
# Authors: Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
PhenoNN installation smoke test.

Run after `uv sync` (or `pip install -e .`) to verify that the package
and its dependencies import cleanly.

Usage
-----
    python test_phenonn_installation.py
"""

import importlib
import os
import subprocess
import sys

from phenonn.utils.logger import Logger


# ── Result bookkeeping ──────────────────────────────────────────────────────
#
# The Logger prints to console and to a file, but we also need to decide the
# process exit code from the tests' outcomes. We collect them in a dict that
# the summary reads at the end.

RESULTS = {
    "dependencies": False,
    "phenonn modules": False,
    "public API": False,
    "model factory": False,
    "CLI": False,
}


def _mark(group: str, ok: bool) -> None:
    """Record the outcome of one test group (AND-aggregated per group)."""
    RESULTS[group] = RESULTS[group] and ok if RESULTS[group] else ok


# ── Import helpers ──────────────────────────────────────────────────────────


def try_import(logger: Logger, dotted: str) -> bool:
    """Import `dotted`, return True on success, log the outcome."""
    try:
        importlib.import_module(dotted)
        logger.success(f"{dotted}")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{dotted}: {type(exc).__name__}: {exc}")
        return False


# ── Test definitions ────────────────────────────────────────────────────────


DEPENDENCIES = [
    "numpy",
    "pandas",
    "xarray",
    "netCDF4",
    "torch",
    "matplotlib",
    "rich",
    "tqdm",
    "scipy",
    "mpltex",
    "sklearn",
]

PHENONN_MODULES = [
    "phenonn",
    "phenonn.data",
    "phenonn.data.lai_dataset",
    "phenonn.models",
    "phenonn.models.rnn",
    "phenonn.models.aelstm",
    "phenonn.models.bitransformer",
    "phenonn.models.transformer",
    "phenonn.training",
    "phenonn.training.train_full_ram",
    "phenonn.prediction",
    "phenonn.prediction.predict",
    "phenonn.utils",
    "phenonn.utils.config",
    "phenonn.utils.wrappers",
    "phenonn.utils.model_loader",
    "phenonn.utils.evaluater",
    "phenonn.utils.diagnostics",
    "phenonn.utils.logger",
    "phenonn.utils.utils",
]

PHENONN_PUBLIC_NAMES = [
    "RamLAIDataset",
    "RNN_LSTM",
    "AELSTM",
    "BiTransformerV2",
    "EncoderTorch",
    "CausalSparseTransformer",
    "SinusoidalPositionalEncoding",
    "train",
    "load_shared",
    "train_one_config",
    "predict",
    "__version__",
]


# ── Test groups ─────────────────────────────────────────────────────────────


def test_environment(logger: Logger) -> None:
    logger.start_task("Python Environment")
    logger.info(f"version    : {sys.version.split()[0]}")
    logger.info(f"executable : {sys.executable}")


def test_dependencies(logger: Logger) -> bool:
    logger.start_task("Core Dependencies")
    ok = True
    for name in DEPENDENCIES:
        ok &= try_import(logger, name)
    return ok


def test_phenonn_modules(logger: Logger) -> bool:
    logger.start_task("PhenoNN Modules")
    ok = True
    for name in PHENONN_MODULES:
        ok &= try_import(logger, name)
    return ok


def test_public_api(logger: Logger) -> bool:
    logger.start_task("PhenoNN Public API")
    try:
        import phenonn
    except Exception as exc:  # noqa: BLE001
        logger.error(f"cannot import phenonn: {type(exc).__name__}: {exc}")
        return False

    ok = True
    for name in PHENONN_PUBLIC_NAMES:
        if hasattr(phenonn, name):
            logger.success(f"phenonn.{name}")
        else:
            logger.error(f"phenonn.{name} is missing")
            ok = False

    declared = set(getattr(phenonn, "__all__", []))
    for name in PHENONN_PUBLIC_NAMES:
        if name not in declared:
            logger.warning(f"{name} not listed in phenonn.__all__")
            ok = False
    return ok


def test_model_factory(logger: Logger) -> bool:
    """Build one model of each kind the .sh files use, on a tiny input."""
    logger.start_task("Model Factory", description="build_model / build_model_pft")
    try:
        import torch
        from types import SimpleNamespace
        from phenonn.utils.config import FEATURE_CHANNELS
        from phenonn.utils.model_loader import build_model

    except Exception as exc:  # noqa: BLE001
        logger.error(f"cannot import model_loader: {type(exc).__name__}: {exc}")
        return False

    ok = True
    for typ in ("lstm", "aelstm", "bitransformer_v2"):
        args = SimpleNamespace(
            type=typ,
            seq_length=365,
            hidden_size=16,
            num_layers=2,
            d_model=None,
            nhead=4,
            forward_expansion=2,
            dropout1=0.0,
            dropout2=0.0,
            dropout_att=0.0,
            dropout_trans=0.0,
            dropout_encoder=0.0,
            feed_forward_trans1=2,
            feed_forward_trans2=2,
            feed_forward_encoder=2,
            n_attn_blocks=1,
            stress_dim=4,
            num_layers1=2,
            d_layers=2,
            embed_size=32,
            daily_lai=False,
            pft_mixing=False,
            pft_meteo_only=False,
        )
        try:
            model = build_model(args)
            x = torch.zeros(1, FEATURE_CHANNELS, 365)
            y = model(x)
            assert y.shape[0] == 1, y.shape
            logger.step(typ, f"built, output shape {tuple(y.shape)}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"build_model('{typ}'): {type(exc).__name__}: {exc}")
            ok = False
    return ok


def test_cli(logger: Logger) -> bool:
    logger.start_task("CLI")
    ok = True
    for argv in (["--version"], ["--help"]):
        command = [sys.executable, "-m", "phenonn", *argv]
        logger.info(f"running: {' '.join(command)}")
        try:
            res = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"timeout after 120 s: {' '.join(command)}")
            ok = False
            continue
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{' '.join(command)}: {exc}")
            ok = False
            continue
        if res.returncode == 0:
            # The banner is printed first; take the last non-empty line of stdout.
            lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
            last = lines[-1] if lines else ""
            logger.step(f"phenonn {' '.join(argv)}", f"exit 0 — {last}")
        else:
            logger.error(
                f"phenonn {' '.join(argv)} exited {res.returncode}: "
                f"{res.stderr.strip()}"
            )
            ok = False
    return ok


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    log_file = "test_phenonn_installation.log"
    logger = Logger(
        console_output=True,
        file_output=True,
        log_file=log_file,
        record=False,
    )
    logger.show_header("PhenoNN Installation Test Suite")

    test_environment(logger)
    _mark("dependencies", test_dependencies(logger))
    _mark("phenonn modules", test_phenonn_modules(logger))
    _mark("public API", test_public_api(logger))
    _mark("model factory", test_model_factory(logger))
    _mark("CLI", test_cli(logger))

    # ── Summary ──
    logger.start_task("Summary")
    for name, ok in RESULTS.items():
        if ok:
            logger.success(name)
        else:
            logger.error(name)

    all_ok = all(RESULTS.values())
    if all_ok:
        logger.success("All checks passed — PhenoNN is ready to use.")
    else:
        logger.warning("Some checks failed — see the messages above.")
        logger.info("Troubleshooting:")
        logger.info("  - uv sync")
        logger.info("  - python -c 'import phenonn; print(phenonn.__file__)'")

    logger.info(f"log file written to {os.path.abspath(log_file)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
