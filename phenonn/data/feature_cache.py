# Copyright 2026 CNRS
# Authors: Esteban Carlin, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Optional on-disk cache for the raw daily feature arrays.

The RAM trainer re-reads and re-decompresses the same ERA5 pixelset cubes
on every job. When several runs share a site set and a year range (another
architecture, another normalisation profile, a resume, a prediction), the
cache turns a 50-minute NetCDF pass into a sequential read of `.npy` files.

Design rules:

- **Raw only.** Nothing normalised is stored. One entry therefore serves
  every ``norm_stats`` file and every model.
- **Derived and disposable.** The cache is never an input of a run. A miss
  rebuilds from the NetCDF files, and deleting the directory is safe.
- **Keyed on every input.** The directory name is a hash of the feature
  files (size and mtime), the parent map, the year list, the site list and
  the channel list. A different seed is a different site list and therefore
  a different entry.
- **Off by default.** ``feature_cache="off"`` leaves the historical
  behaviour untouched.
- **Atomic.** Entries are built in ``<key>.tmp`` and renamed on success, so
  an interrupted job never leaves a half-valid entry behind.

Storage location, in order: ``--cache_dir``, ``$PHENONN_CACHE_DIR``,
``./cache`` when it exists, else ``~/.cache/phenonn``.

Administration::

    python -m phenonn.data.feature_cache list
    python -m phenonn.data.feature_cache rm --key <key>
    python -m phenonn.data.feature_cache rm --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np

from phenonn.utils.config import DYNAMIC_FEATURES, FEATURES_FNAME

try:  # POSIX only; the cache stays single-process on other platforms.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

CACHE_VERSION = 1
MODES = ("off", "read", "write", "auto")
MARKER = "COMPLETE"


def default_cache_dir() -> str:
    """Resolve the cache root when ``--cache_dir`` is empty.

    Returns
    -------
    str
        ``$PHENONN_CACHE_DIR`` when set, else ``./cache`` when that
        directory already exists, else ``~/.cache/phenonn``.
    """
    env = os.environ.get("PHENONN_CACHE_DIR", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    local = os.path.abspath("cache")
    if os.path.isdir(local):
        return local
    return os.path.join(os.path.expanduser("~"), ".cache", "phenonn")


def features_root(root: str) -> str:
    """Return the directory that holds the feature entries.

    Parameters
    ----------
    root : str
        Cache root, as returned by :func:`default_cache_dir`.

    Returns
    -------
    str
        ``<root>/features``.
    """
    return os.path.join(os.path.abspath(root), "features")


def _file_tag(path: str) -> str:
    """Return ``size:mtime`` for a source file, or ``missing``.

    Parameters
    ----------
    path : str
        File to stat.

    Returns
    -------
    str
        Compact fingerprint component.
    """
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return f"{st.st_size}:{int(st.st_mtime)}"


def feature_key(
    features_dir: str,
    years: Iterable[int],
    site_ids: Iterable[str],
    parent_map_path: str = "",
) -> str:
    """Build the cache key for one raw feature set.

    Parameters
    ----------
    features_dir : str
        Folder of ``ERA5_daily_pixelset_{Y}.nc`` files.
    years : iterable of int
        Feature years that will be read.
    site_ids : iterable of str
        Requested target sites, in any order.
    parent_map_path : str, optional
        File the parent map was read from; empty when there is none.

    Returns
    -------
    str
        Short hexadecimal key.
    """
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}|".encode())
    h.update(os.path.realpath(features_dir).encode())
    h.update(b"|")
    h.update(",".join(DYNAMIC_FEATURES).encode())
    h.update(b"|")
    for y in sorted({int(y) for y in years}):
        path = os.path.join(features_dir, FEATURES_FNAME.format(year=y))
        h.update(f"{y}:{_file_tag(path)}|".encode())
    h.update(b"parent:")
    h.update((_file_tag(parent_map_path) if parent_map_path else "none").encode())
    h.update(b"|")
    sites = "\n".join(sorted({str(s) for s in site_ids}))
    h.update(hashlib.sha1(sites.encode()).hexdigest().encode())
    return h.hexdigest()[:16]


class _CacheLock:
    """Advisory lock so two jobs cannot build the same entry at once."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "w")
        if fcntl is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc_info):
        if self._fh is not None:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False


class FeatureCache:
    """One feature-cache entry, addressed by ``root`` and ``key``.

    Parameters
    ----------
    root : str
        Cache root.
    key : str
        Value returned by :func:`feature_key`.
    """

    def __init__(self, root: str, key: str) -> None:
        self.root = os.path.abspath(root)
        self.key = key
        self.path = os.path.join(features_root(self.root), key)
        self.tmp = self.path + ".tmp"
        self._years: Dict[str, dict] = {}

    # ── State ──────────────────────────────────────────────────────────────

    def exists(self) -> bool:
        """Return True when a complete, readable entry is present.

        Returns
        -------
        bool
            ``MARKER`` and ``meta.json`` present and every year file there.
        """
        if not os.path.exists(os.path.join(self.path, MARKER)):
            return False
        try:
            meta = self.meta()
        except (OSError, ValueError):
            return False
        return all(os.path.exists(self._year_path(y)) for y in meta.get("years", {}))

    def meta(self) -> dict:
        """Return the entry metadata.

        Returns
        -------
        dict
            ``{"version", "key", "years": {Y: {"t", "n_sites", "sites"}}}``.
        """
        with open(os.path.join(self.path, "meta.json")) as fh:
            return json.load(fh)

    def _year_path(self, year: int) -> str:
        return os.path.join(self.path, f"f{int(year)}.npy")

    # ── Read ───────────────────────────────────────────────────────────────

    def read_year(self, year: int) -> np.ndarray:
        """Load one year of raw features.

        Parameters
        ----------
        year : int
            Feature year.

        Returns
        -------
        np.ndarray
            Writable ``(n_sites, n_days, n_dynamic)`` float32 copy. The
            caller may normalise it in place without touching the cache.
        """
        arr = np.load(self._year_path(year), mmap_mode="r")
        out = np.array(arr, dtype=np.float32, copy=True)
        del arr
        return out

    # ── Write ──────────────────────────────────────────────────────────────

    def lock(self) -> _CacheLock:
        """Return the advisory build lock for this entry."""
        return _CacheLock(os.path.join(self.root, "locks", self.key + ".lock"))

    def begin(self, years: Dict[int, dict]) -> None:
        """Start an atomic build.

        Parameters
        ----------
        years : dict
            ``{year: {"t": n_days, "n_sites": n, "sites": [...]}}``.
        """
        if os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp, exist_ok=True)
        self._years = {str(int(y)): info for y, info in years.items()}

    def write_year(self, year: int, array: np.ndarray) -> None:
        """Store one raw year array.

        Parameters
        ----------
        year : int
        array : np.ndarray
            ``(n_sites, n_days, n_dynamic)``, float32.
        """
        np.save(
            os.path.join(self.tmp, f"f{int(year)}.npy"),
            np.ascontiguousarray(array, dtype=np.float32),
            allow_pickle=False,
        )

    def commit(self) -> None:
        """Write metadata and atomically publish the entry."""
        payload = {
            "version": CACHE_VERSION,
            "key": self.key,
            "years": self._years,
        }
        with open(os.path.join(self.tmp, "meta.json"), "w") as fh:
            json.dump(payload, fh)
        with open(os.path.join(self.tmp, MARKER), "w") as fh:
            fh.write("complete\n")
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.rename(self.tmp, self.path)

    def abort(self) -> None:
        """Remove a partially written build."""
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def expected_bytes(years: Dict[int, dict]) -> int:
        """Return the on-disk size of an entry in bytes.

        Parameters
        ----------
        years : dict
            Same payload as :meth:`begin`.

        Returns
        -------
        int
            Sum of ``n_sites * n_days * n_dynamic * 4`` over the years.
        """
        n_dyn = len(DYNAMIC_FEATURES)
        return sum(
            int(info["n_sites"]) * int(info["t"]) * n_dyn * 4 for info in years.values()
        )


# ── Administration ──────────────────────────────────────────────────────────


def _human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"  # pragma: no cover - unreachable


def list_entries(root: Optional[str] = None) -> List[dict]:
    """List the complete entries under a cache root.

    Parameters
    ----------
    root : str, optional
        Cache root; defaults to :func:`default_cache_dir`.

    Returns
    -------
    list of dict
        ``key``, ``path``, ``size`` in bytes, ``years``, ``n_sites`` and
        ``source_mtime`` (the newest source file recorded in the key).
    """
    base = features_root(root or default_cache_dir())
    entries: List[dict] = []
    if not os.path.isdir(base):
        return entries
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, MARKER)):
            continue
        try:
            with open(os.path.join(path, "meta.json")) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        size = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )
        years = meta.get("years", {})
        entries.append(
            {
                "key": name,
                "path": path,
                "size": size,
                "years": sorted(int(y) for y in years),
                "n_sites": max(
                    (int(info["n_sites"]) for info in years.values()), default=0
                ),
                "source_mtime": max(
                    (
                        os.path.getmtime(os.path.join(path, f))
                        for f in os.listdir(path)
                        if f.startswith("f")
                    ),
                    default=0.0,
                ),
            }
        )
    return entries


def remove_entry(
    key: Optional[str] = None,
    root: Optional[str] = None,
    remove_all: bool = False,
) -> List[str]:
    """Delete one entry, or every entry under a root.

    Parameters
    ----------
    key : str, optional
        Entry key. Ignored when ``remove_all`` is True.
    root : str, optional
        Cache root; defaults to :func:`default_cache_dir`.
    remove_all : bool, optional
        Delete every complete entry.

    Returns
    -------
    list of str
        Paths removed.
    """
    base = features_root(root or default_cache_dir())
    targets: List[str] = []
    if remove_all:
        if os.path.isdir(base):
            targets = [
                os.path.join(base, name)
                for name in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, name))
            ]
    elif key:
        targets = [os.path.join(base, key)]
    removed = []
    for path in targets:
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(path)
    return removed


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point for cache administration.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phenonn.data.feature_cache",
        description="List or delete raw feature-cache entries.",
    )
    parser.add_argument("command", choices=("list", "rm"))
    parser.add_argument("--root", default="", help="Cache root.")
    parser.add_argument("--key", default="", help="Entry key (rm only).")
    parser.add_argument(
        "--all", action="store_true", dest="remove_all", help="Delete every entry."
    )
    args = parser.parse_args(argv)
    root = args.root or default_cache_dir()

    if args.command == "list":
        entries = list_entries(root)
        if not entries:
            print(f"no cache entries under {features_root(root)}")
            return 0
        total = 0
        for entry in entries:
            total += entry["size"]
            years = entry["years"]
            span = f"{years[0]}-{years[-1]}" if years else "no years"
            print(
                f"{entry['key']}  {_human_size(entry['size']):>10}  "
                f"{span:>9}  {entry['n_sites']:,} sites  {entry['path']}"
            )
        print(f"{len(entries)} entries, {_human_size(total)}")
        return 0

    if not args.key and not args.remove_all:
        parser.error("rm needs --key or --all")
    removed = remove_entry(args.key, root, args.remove_all)
    if not removed:
        print("nothing removed")
        return 1
    for path in removed:
        print(f"removed {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
