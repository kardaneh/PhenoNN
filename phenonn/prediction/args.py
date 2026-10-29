"""PhenoNN prediction CLI arguments.

This module is dependency-light: it imports argparse and the protocol
normaliser only, so ``phenonn predict --help`` does not import torch.
"""

from __future__ import annotations

import argparse

from phenonn.evaluation.leakage import normalise_protocol


def parse_args(argv=None):
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description="PhenoNN inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features_dir", default="")
    p.add_argument("--target_dir", default="")
    p.add_argument("--pft_dir", default="")
    p.add_argument(
        "--parent_map",
        default="",
        help="Optional selected_pixels01_1.nc. When set, ERA5 features are "
        "read from the deduplicated 0.1° pixelset via each 0.05° site's "
        "parent cell, instead of a 0.05°-indexed feature file. "
        "Must match the --parent_map used at training.",
    )
    p.add_argument(
        "--predict_sites",
        default="val",
        choices=["val", "train", "all", "grid", "test"],
    )
    p.add_argument(
        "--selected_pixels",
        default="",
        help="Path to a selected_pixels*.nc. Overrides --sites and " "--predict_sites.",
    )
    p.add_argument(
        "--sites",
        default="",
        help="Comma-separated explicit site IDs — overrides --predict_sites.",
    )
    p.add_argument(
        "--n_predict_sites",
        type=int,
        default=0,
        help="If > 0, random subsample of this many sites.",
    )
    p.add_argument(
        "--predict_years",
        default="",
        help="'2015-2018', '2015,2016' or 'all'. Empty → val_years from the ckpt.",
    )
    p.add_argument("--row_min", type=int, default=-1)
    p.add_argument("--row_max", type=int, default=-1)
    p.add_argument("--col_min", type=int, default=-1)
    p.add_argument("--col_max", type=int, default=-1)
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scatter_years",
        action="store_true",
        help="One pred-vs-obs scatter per evaluated year, in a "
        "`scatter_year/` sub-folder next to --output_csv.",
    )
    p.add_argument(
        "--n_curves",
        type=int,
        default=None,
        help="Number of per-site curves to plot in *_lai_curves_all.png. "
        "Default: 100. 0 = plot every site that produced predictions.",
    )
    p.add_argument(
        "--ablate_features",
        default="",
        help="Comma list of features the model was TRAINED without. "
        "Normally read from the checkpoint.",
    )
    p.add_argument(
        "--pft_min_frac",
        type=float,
        default=0.05,
        help="Show PFTs with fraction ≥ this threshold above each curve "
        "subplot (default: 0.05 = 5%%).",
    )
    # ── Paper-protocol bundle options ──
    p.add_argument(
        "--clim_years",
        default="",
        help="Years used for the anomaly climatology, e.g. '1992-2009'. "
        "Empty -> train_years stored in the checkpoint.",
    )
    p.add_argument(
        "--no_climatology",
        action="store_true",
        help="Skip the anomaly climatology and the anomaly metrics "
        "(quick diagnostics only; paper runs keep it on).",
    )
    p.add_argument(
        "--clim_target_dir",
        default="",
        help="Folder of LAI_dekadal_{Y}.nc used to build the climatology. "
        "Empty -> --target_dir / checkpoint target_dir.",
    )
    p.add_argument(
        "--bundle_dir",
        default="",
        help="Directory for metrics.json / per-site tables. "
        "Empty -> directory of --output_csv.",
    )
    p.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap draws over sites for the 95%% CIs (0 disables).",
    )
    p.add_argument(
        "--protocol",
        default="none",
        type=normalise_protocol,
        choices=["A", "B", "none"],
        help="Split protocol used for leakage checks: A = reproduction "
        "(biased), B = unbiased paper split, none = ad-hoc.",
    )
    p.add_argument(
        "--test_years",
        default="",
        help="Test years, e.g. '2010-2019'. Used by the leakage checks.",
    )
    p.add_argument(
        "--allow_leakage",
        action="store_true",
        help="Downgrade leakage errors to warnings (not for paper runs).",
    )
    p.add_argument(
        "--no_pft_columns",
        action="store_true",
        help="Do not attach pft_dom / bare_frac / purity to the predictions.",
    )
    p.add_argument(
        "--no_phenology_metrics",
        action="store_true",
        help="Skip pheno_metrics.csv (slow on >200k site-years).",
    )
    p.add_argument(
        "--pheno_max_siteyears",
        type=int,
        default=0,
        help="If > 0, compute phenology on a deterministic subsample of "
        "site-years larger than this.",
    )
    args = p.parse_args(argv)
    if args.n_curves is None:
        args.n_curves = 100
        args._n_curves_defaulted = True
    return args
