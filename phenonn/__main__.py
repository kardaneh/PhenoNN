# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Allow running the PhenoNN package as a module with: python -m phenonn
"""

import sys
from phenonn.cli import main

if __name__ == "__main__":
    sys.exit(main())
