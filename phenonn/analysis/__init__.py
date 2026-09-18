# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN post-training analyses: baselines, climatology comparison, phenology
dates, stress sensitivity, grid/PFT diagnostics.

Each module is a standalone CLI (``python -m phenonn.analysis.<module>``);
nothing is re-exported here — the scripts share entry-point names (main /
parse_args) and pull heavy optional deps, so eager aggregation is avoided.
"""

__all__ = []
