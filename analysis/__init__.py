# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Post-training analyses for PhenoNN.

Sub-packages and studies (all run as
``python -m analysis.<study>.<script>``):

- ``diagnostics``: one-shot data/model diagnostics and baseline CSVs.
- ``figures``: paper figures F1–F13 and LaTeX tables from run bundles.
- ``feature_ablation``: leave-one-feature-out and grouped channel ablations.
- ``perturbation``: ceteris-paribus weather perturbation scenarios.
- ``pft_linearity``: linearity of a no-mixing model in PFT fractions.
- ``year_generalization``: chronological, random, and cold-to-hot year splits.
- ``memory_analysis``: memory kernel (M1) plus the remaining study protocol.

``analysis/_checkpoint_utils`` holds the shared checkpoint helpers and
``analysis/_cluster.sh`` holds the shared cluster and environment helpers.

Nothing is re-exported here. The scripts share entry-point names (main /
parse_args) and pull heavy optional dependencies, so eager aggregation is
avoided.
"""

__all__ = []
