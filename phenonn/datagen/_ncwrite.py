# Copyright 2026 CNRS
# Authors: Stefan Barbu, Kazem Ardaneh, Esteban Carlin
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Write an xarray Dataset to NetCDF from a *fresh* subprocess.

Why this exists
---------------
On some environments (the IPSL "spirit" cluster `.venv_ERA5`) the pip wheels
of ``netCDF4`` and ``h5py`` each bundle their OWN libhdf5. Once h5py is loaded
in a process — e.g. to read the ORCHIDEE PFTmap via ``engine="h5netcdf"`` or to
read THEIA H5 tiles via ``h5py.File`` — a subsequent ``netCDF4`` write in the
SAME interpreter corrupts the global HDF5 state and dies with
``RuntimeError: NetCDF: HDF error`` (inside ``Variable._put``).

Running the write in a **spawned** interpreter that never imports h5py
sidesteps the conflict. A plain ``multiprocessing`` fork would NOT help: the
child would inherit the parent's already-loaded (poisoned) HDF5 library, so we
launch a brand-new ``sys.executable`` via ``subprocess``.

The Dataset is materialised (``.load()``) and pickled to a temp file next to
the output (same filesystem, room on /data), then the child unpickles and
writes it. ``import xarray`` does not pull in h5py, and the write uses
``engine="netcdf4"``, so the child stays h5py-free.
"""

import os
import pickle
import subprocess
import sys
import tempfile


# Child program: load the pickled (ds, path, encoding, engine) and write.
# Imports only pickle + (via unpickle) xarray + (via to_netcdf) netCDF4.
_CHILD = (
    "import pickle, sys\n"
    "with open(sys.argv[1], 'rb') as fh:\n"
    "    ds, out_path, encoding, engine = pickle.load(fh)\n"
    "ds.to_netcdf(out_path, engine=engine, encoding=encoding)\n"
)


def to_netcdf_subprocess(ds, out_path, encoding=None, engine="netcdf4"):
    """
    Write ``ds`` to ``out_path`` via a fresh h5py-free subprocess.

    Parameters
    ----------
    ds : xarray.Dataset
        Materialised in-process with ``.load()`` before pickling.
    out_path : str or os.PathLike
    encoding : dict or None
        Same ``encoding`` dict you would pass to ``Dataset.to_netcdf``.
    engine : str
        NetCDF engine for the child (default ``"netcdf4"``).

    Raises
    ------
    RuntimeError
        If the child process exits non-zero (its stderr is included).
    """
    out_path = os.fspath(out_path)
    ds = ds.load()

    tmp_dir = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(suffix=".pkl", dir=tmp_dir)
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(
                (ds, out_path, encoding, engine), fh, protocol=pickle.HIGHEST_PROTOCOL
            )
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, tmp],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"subprocess NetCDF write failed (rc={r.returncode}) for "
                f"{out_path}:\n{r.stderr.strip()}"
            )
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
