"""Run registry for the paper figures.

A *run* is a directory that contains ``metrics.json`` (written by
``predict.py``) and optionally ``history.csv``, ``predictions.csv``,
``run_metadata.json`` and ``config.json``.  The registry scans a tree, loads
the small metadata files, and gives the figure functions a single place to
look up a model by name.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import pandas as pd

__all__ = ["Run", "Registry", "load_json", "discover_runs"]


def load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@dataclass
class Run:
    """One prediction bundle on disk."""

    name: str
    directory: str
    metrics: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    prediction_metadata: Dict = field(default_factory=dict)

    @property
    def model_type(self) -> str:
        for src in (self.config, self.metadata.get("config", {}), self.metrics):
            v = src.get("type") or src.get("model_type")
            if v:
                return str(v).lower()
        return ""

    @property
    def model_label(self) -> str:
        """Human-facing architecture name used by the paper tables."""
        model = self.model_type
        if model in ("bitransformer_v2", "bittransformer_v2"):
            return "bitransformer"
        return model or self.name

    @property
    def seed(self):
        for src in (self.config, self.metadata.get("config", {})):
            if src.get("seed") is not None:
                return src.get("seed")
        return None

    @property
    def best_epoch(self):
        """Best validation epoch, if the training metadata recorded it."""
        for src in (self.metadata, self.metrics, self.config):
            value = src.get("best_epoch")
            if value is not None:
                return value
        return None

    @property
    def prediction_sites(self):
        """Site pool used to write ``predictions.csv``, if recorded."""
        return self.prediction_metadata.get("predict_sites")

    @property
    def prediction_years(self):
        """Years used to write ``predictions.csv``, if recorded."""
        return self.prediction_metadata.get("predict_years")

    @property
    def climatology(self) -> Dict:
        """Metrics of the training-year climatology for this bundle."""
        return load_json(self.file("climatology_metrics.json"))

    @property
    def protocol(self):
        """Canonical split protocol, or ``None`` for pre-protocol runs."""
        raw = self.metadata.get("protocol") or self.config.get("protocol")
        if raw is None:
            return None
        # Imported lazily so the figure package stays usable without the
        # training stack.
        from phenonn.evaluation.leakage import normalise_protocol

        return normalise_protocol(raw)

    def file(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def has(self, name: str) -> bool:
        return os.path.exists(self.file(name))

    def predictions(self) -> pd.DataFrame:
        path = self.file("predictions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def history(self) -> pd.DataFrame:
        return pd.read_csv(self.file("history.csv"))


def discover_runs(root: str) -> List[Run]:
    """Recursively find directories containing ``metrics.json``."""
    runs: List[Run] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "metrics.json" not in filenames:
            continue
        name = os.path.basename(dirpath.rstrip(os.sep))
        metrics = load_json(os.path.join(dirpath, "metrics.json"))
        metadata = load_json(os.path.join(dirpath, "run_metadata.json"))
        config = load_json(os.path.join(dirpath, "config.json"))
        prediction_metadata = load_json(os.path.join(dirpath, "predict_metadata.json"))
        runs.append(
            Run(
                name=name,
                directory=dirpath,
                metrics=metrics,
                metadata=metadata,
                config=config,
                prediction_metadata=prediction_metadata,
            )
        )
    runs.sort(key=lambda r: (r.model_type, r.name))
    return runs


class Registry:
    """A thin, name-addressable collection of runs."""

    def __init__(self, runs: Iterable[Run]):
        self.runs = list(runs)
        self._by_name = {r.name: r for r in self.runs}

    @classmethod
    def from_dir(cls, root: str) -> "Registry":
        return cls(discover_runs(root))

    def __len__(self):
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def get(self, name: str) -> Run:
        if name in self._by_name:
            return self._by_name[name]
        matches = [r for r in self.runs if name in r.name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"no run matching {name!r} in {len(self.runs)} runs")
        raise KeyError(f"{name!r} is ambiguous: {[r.name for r in matches]}")

    def by_model(self, model: str) -> List[Run]:
        model = model.lower()
        return [r for r in self.runs if r.model_type == model]

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.runs:
            rows.append(
                {
                    "name": r.name,
                    "model_type": r.model_type,
                    "seed": r.seed,
                    "protocol": r.protocol,
                    "directory": r.directory,
                    "overall_r2": r.metrics.get("overall_r2"),
                    "rmse": r.metrics.get("rmse"),
                    "mae": r.metrics.get("mae"),
                    "bias": r.metrics.get("bias"),
                    "n_points": r.metrics.get("n_points"),
                }
            )
        return pd.DataFrame(rows)
