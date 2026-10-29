# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Version information for PhenoNN.
"""

__version__ = "0.1.0"


def get_version():
    """Return the version string."""
    return __version__


def get_versions():
    """Return a dictionary with version information."""
    return {
        "version": __version__,
        "author": "Kazem Ardaneh",
        "email": "kazem.arrdaneh@gmail.com",
        "url": "https://github.com/kardaneh/PhenoNN",
    }
