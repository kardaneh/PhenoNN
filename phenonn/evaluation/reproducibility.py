"""Reproducibility metadata for training and inference bundles.

Every run writes a ``run_metadata.json`` recording the resolved configuration,
the git commit, and the SHA-256 of every input file it touched.  The figure
package reads the same file so a reported number can always be traced back to
a commit and a dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

__all__ = [
    "sha256_file",
    "git_commit",
    "git_dirty",
    "collect_run_metadata",
    "write_json",
    "jsonable",
]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, or an empty string when it does not exist."""
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_commit(cwd: Optional[str] = None) -> str:
    """Return the current commit hash, or ``""`` outside a git repository."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return ""


def git_dirty(cwd: Optional[str] = None) -> bool:
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def jsonable(value):
    """Recursively convert numpy / torch scalars and paths to JSON values.

    Non-finite floats (NaN, ±inf) become ``None`` so the files are strict
    JSON and can be read by any downstream tool.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def collect_run_metadata(
    args,
    data_paths: Optional[Iterable[str]] = None,
    extra: Optional[Dict] = None,
    cwd: Optional[str] = None,
) -> Dict:
    """Build the JSON-serialisable metadata block for one run.

    Parameters
    ----------
    args : argparse.Namespace or dict
        Resolved CLI configuration.
    data_paths : iterable of str, optional
        Files whose SHA-256 should be recorded.
    extra : dict, optional
        Additional keys (for example ``protocol`` or ``clim_years``).
    cwd : str, optional
        Repository root used for the git probe.
    """
    cfg = vars(args) if hasattr(args, "__dict__") else dict(args)
    # Private, runtime-only attributes (for example the in-memory site pool)
    # must never end up in the metadata file.
    cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    paths = sorted({os.fspath(p) for p in (data_paths or []) if p})
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(cwd),
        "git_dirty": git_dirty(cwd),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": jsonable(cfg),
        "data_files": {p: sha256_file(p) for p in paths},
    }
    if extra:
        meta.update(jsonable(extra))
    return meta


def write_json(path: str, payload) -> None:
    """Write ``payload`` as pretty JSON, creating the parent directory."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")
