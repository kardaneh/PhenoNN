# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN command-line interface.

Usage
-----
    phenonn train       Train a model (RAM working-set trainer)
    phenonn predict     Run inference with a trained checkpoint
    phenonn --version   Print the package version
    phenonn --help      Print this help

Every subcommand forwards the remaining arguments to the underlying
``python -m`` entry point. For the full list of flags, run, e.g.:

    phenonn train --help
    phenonn predict --help

Equivalent direct invocations (used by the SLURM job scripts):

    python -m phenonn.training.train_full_ram  <flags>
    python -m phenonn.prediction.predict       <flags>
"""

import sys

from .version import __version__


_HELP = f"""\
PhenoNN version {__version__}

Usage: phenonn <command> [options]

Commands:
  train       Train a model (RAM working-set trainer, pixelset NetCDF)
  predict     Run inference with a trained checkpoint

Run 'phenonn <command> --help' for the full list of options.
"""


def _dispatch(module_path: str, argv: list) -> int:
    """Import `module_path`, run its `main()` with `argv` as sys.argv.

    `argv` must already include the program name at index 0, matching
    the layout the underlying argparse parsers expect.
    """
    import importlib

    module = importlib.import_module(module_path)
    original_argv = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = original_argv
    return 0


def _dispatch_help(module_path: str, argv: list) -> None:
    """Print a subcommand's help without importing the heavy entry point."""
    import importlib

    parse_args = importlib.import_module(module_path).parse_args
    parse_args(argv)


def main() -> int:
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"phenonn {__version__}")
        return 0

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_HELP)
        return 0

    command = sys.argv[1]
    subcommand_argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        if any(arg in ("-h", "--help") for arg in subcommand_argv[1:]):
            _dispatch_help("phenonn.training.args", subcommand_argv[1:])
        return _dispatch("phenonn.training.train_full_ram", subcommand_argv)
    if command == "predict":
        if any(arg in ("-h", "--help") for arg in subcommand_argv[1:]):
            _dispatch_help("phenonn.prediction.args", subcommand_argv[1:])
        return _dispatch("phenonn.prediction.predict", subcommand_argv)

    print(f"Unknown command: {command}", file=sys.stderr)
    print(_HELP, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
