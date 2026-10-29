# Copyright 2026 CNRS
# Authors: Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
PhenoNN datagen installation smoke test.

Run after `uv sync` (or `pip install -e .`) to verify that the
`phenonn.datagen` subpackage and its CLI scripts import cleanly.

Usage
-----
    python test_datagen_installation.py
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from phenonn.utils.logger import Logger


# ── Result bookkeeping ──────────────────────────────────────────────────────
#
# The Logger prints to console and to a file, but we also need to decide the
# process exit code from the tests' outcomes. We collect them in a dict that
# the summary reads at the end.

RESULTS = {
    "shared modules": False,
    "datagen modules": False,
    "public API": False,
    "CLI": False,
    "data root": False,
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


SHARED_MODULES = [
    "phenonn.datagen",
    "phenonn.datagen.config",
    "phenonn.datagen._ncwrite",
]

DATAGEN_MODULES = [
    "phenonn.datagen.compute_valid_pixels",
    "phenonn.datagen.select_pixels",
    "phenonn.datagen.rebalance_pixels",
    "phenonn.datagen.rebalance_spatial",
    "phenonn.datagen.make_selected_pixels_01",
    "phenonn.datagen.build_daily_dataset_pixelset",
    "phenonn.datagen.build_daily_dataset_era5idx",
    "phenonn.datagen.build_pixelset_targets",
    "phenonn.datagen.interpolate_lai_daily_linear",
    "phenonn.datagen.add_daylength",
    "phenonn.datagen.add_soil_moisture_proxy",
    "phenonn.datagen.add_pheno_thermal",
    "phenonn.datagen.create_stats_file",
    "phenonn.datagen.diagnose_selected_pixels",
    "phenonn.datagen.diagnose_selected_pixels_bis",
    "phenonn.datagen.pft_diagnostics",
]

DATAGEN_CLI = [
    "compute_valid_pixels",
    "select_pixels",
    "rebalance_pixels",
    "rebalance_spatial",
    "make_selected_pixels_01",
    "build_daily_dataset_pixelset",
    "build_daily_dataset_era5idx",
    "build_pixelset_targets",
    "interpolate_lai_daily_linear",
    "add_daylength",
    "add_soil_moisture_proxy",
    "add_pheno_thermal",
    "create_stats_file",
    "diagnose_selected_pixels",
    "diagnose_selected_pixels_bis",
    "pft_diagnostics",
]

DATAGEN_PUBLIC_NAMES = [
    "N_PFT",
    "N_DEKAD_YEAR",
    "METEO_BASE",
    "DYNAMIC_FEATURES",
    "PFT_COLS",
    "PFT_NAMES",
    "FEATURES_FNAME",
    "TARGETS_FNAME",
    "PFT_FNAME",
    "VALID_FNAME",
    "SELECTED_PIXELS_FNAME",
]

# Canonical data root: the final `pixelset1/` deliverable lives under it.
# Override with PHENONN_DATA_ROOT when the data lives elsewhere.
DATA_ROOT = Path(
    os.environ.get(
        "PHENONN_DATA_ROOT",
        Path(__file__).resolve().parents[2],
    )
)

# The nine items a completed `pixelset1/` is expected to contain.
# `valid_pixels/` and `all_pixels_selected.nc` are NOT here on purpose:
# they are intermediate products of Steps 1–2 and are not part of the
# deliverable. See the README "What is not in pixelset1/" note.
PIXELSET1_CONTENTS = [
    DATA_ROOT / "pixelset1" / "CO2_1700_2023_TRENDYv2024.txt",
    DATA_ROOT / "pixelset1" / "era5_pixelset",
    DATA_ROOT / "pixelset1" / "LAI_pixelset",
    DATA_ROOT / "pixelset1" / "LAI_daily_linear",
    DATA_ROOT / "pixelset1" / "PFT_pixelset",
    DATA_ROOT / "pixelset1" / "norm_stats_1992_2019.json",
    DATA_ROOT / "pixelset1" / "selected_pixels005_1.nc",
    DATA_ROOT / "pixelset1" / "selected_pixels005_balanced.nc",
    DATA_ROOT / "pixelset1" / "selected_pixels01_1.nc",
]


# ── Test groups ─────────────────────────────────────────────────────────────


def test_environment(logger: Logger) -> None:
    logger.start_task("Python Environment")
    logger.info(f"version    : {sys.version.split()[0]}")
    logger.info(f"executable : {sys.executable}")


def test_shared_modules(logger: Logger) -> bool:
    logger.start_task("Shared Modules", description="config / _ncwrite / __init__")
    ok = True
    for name in SHARED_MODULES:
        ok &= try_import(logger, name)
    return ok


def test_datagen_modules(logger: Logger) -> bool:
    logger.start_task("Datagen Modules")
    ok = True
    for name in DATAGEN_MODULES:
        ok &= try_import(logger, name)
    return ok


def test_public_api(logger: Logger) -> bool:
    logger.start_task("phenonn.datagen Public API")
    try:
        import phenonn.datagen as dg
    except Exception as exc:  # noqa: BLE001
        logger.error(f"cannot import phenonn.datagen: {type(exc).__name__}: {exc}")
        return False

    ok = True
    for name in DATAGEN_PUBLIC_NAMES:
        if hasattr(dg, name):
            logger.success(f"phenonn.datagen.{name}")
        else:
            logger.error(f"phenonn.datagen.{name} is missing")
            ok = False

    declared = set(getattr(dg, "__all__", []))
    for name in DATAGEN_PUBLIC_NAMES:
        if name not in declared:
            logger.warning(f"{name} not listed in phenonn.datagen.__all__")
            ok = False
    return ok


def test_cli(logger: Logger) -> bool:
    logger.start_task("CLI", description="python -m phenonn.datagen.<script> --help")
    ok = True
    for name in DATAGEN_CLI:
        try:
            res = subprocess.run(
                [sys.executable, "-m", f"phenonn.datagen.{name}", "--help"],
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"phenonn.datagen.{name} --help: {exc}")
            ok = False
            continue
        if res.returncode == 0:
            # The usage block is printed first; take the first non-empty line.
            lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
            first = lines[0] if lines else ""
            logger.step(f"phenonn.datagen.{name}", f"exit 0 — {first}")
        else:
            logger.error(
                f"phenonn.datagen.{name} exited {res.returncode}: "
                f"{res.stderr.strip()}"
            )
            ok = False
    return ok


def test_data_root(logger: Logger) -> bool:
    """Informational: report whether `pixelset1/` is reachable and populated."""
    logger.start_task("Data Root", description=str(DATA_ROOT))
    ok = True

    if not DATA_ROOT.exists():
        logger.warning(f"data root not found (skipped): {DATA_ROOT}")
        return True

    logger.success(f"data root present: {DATA_ROOT}")
    for path in PIXELSET1_CONTENTS:
        if path.exists():
            logger.step(path.name, "present")
        else:
            logger.warning(f"missing: {path}")
    return ok


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    log_file = "test_datagen_installation.log"
    logger = Logger(
        console_output=True,
        file_output=True,
        log_file=log_file,
        record=False,
    )
    logger.show_header("PhenoNN datagen Installation Test Suite")

    test_environment(logger)
    _mark("shared modules", test_shared_modules(logger))
    _mark("datagen modules", test_datagen_modules(logger))
    _mark("public API", test_public_api(logger))
    _mark("CLI", test_cli(logger))

    # The data root is informational only: it must not gate the exit code,
    # because the test is meant to be runnable on a build node where the data
    # has not been staged yet. We record it but always leave it as True for
    # the purpose of the summary.
    test_data_root(logger)
    _mark("data root", True)

    # ── Summary ──
    logger.start_task("Summary")
    for name, ok in RESULTS.items():
        if ok:
            logger.success(name)
        else:
            logger.error(name)

    all_ok = all(RESULTS.values())
    if all_ok:
        logger.success("All checks passed — phenonn.datagen is ready to use.")
    else:
        logger.warning("Some checks failed — see the messages above.")
        logger.info("Troubleshooting:")
        logger.info("  - uv sync")
        logger.info(
            "  - python -c 'import phenonn.datagen; " "print(phenonn.datagen.__file__)'"
        )
        logger.info(
            "  - grep -rn '^from \\(config\\|_ncwrite\\|dataset_05\\)' "
            "phenonn/datagen/*.py"
        )

    logger.info(f"log file written to {os.path.abspath(log_file)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
