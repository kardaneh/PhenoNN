"""
Shared naming / feature list for the feature-ablation study.

The non-PFT features (everything before the PFT block in the canonical feature
tensor) are the ones we ablate — one run each — plus one baseline run. Keeping
the list and the experiment names here means gen_jobs.py (which writes the
SLURM scripts) and aggregate.py (which reads the results back) can never drift.
"""

from phenonn.utils.config import ALL_FEATURES, PFT_START

# Canonical order is [dynamic, cyclic, co2, pft]; PFT_START is where the PFT
# block begins, so ALL_FEATURES[:PFT_START] is exactly the non-PFT set.
NON_PFT_FEATURES = list(ALL_FEATURES[:PFT_START])

# Experiment folder names under --output_dir (runs/<exp>/checkpoints/best_model.pth).
BASELINE_EXP = "abl_baseline"


def exp_name(feature: str) -> str:
    """Experiment name for the run that ablates `feature`."""
    return f"abl_{feature}"
