"""Split- and leakage-validation helpers.

The paper protocol has two splits (``PROTOCOL.md`` §2):

``A`` — reproduction: train 1992–2009, validation and test 2010–2019.
``B`` — paper: train 1992–2006, validation 2007–2009, test 2010–2019.

These helpers are intentionally strict: a run that would leak test years into
normalisation statistics, the anomaly climatology or early stopping raises
:class:`LeakageError` before any compute is spent.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = [
    "LeakageError",
    "PROTOCOLS",
    "normalise_protocol",
    "parse_years",
    "overlap",
    "validate_split",
    "validate_normalisation_years",
    "validate_climatology_years",
    "reconcile_resumed_clim_years",
]

PROTOCOLS = ("A", "B", "none")


class LeakageError(ValueError):
    """Raised when a configuration leaks validation or test years."""


def normalise_protocol(protocol) -> str:
    """Return the canonical name of a split protocol.

    The three accepted protocols are defined in ``PROTOCOL.md``:

    ``A``
        Reproduction split: train 1992--2009, validation and test
        2010--2019.  Validation and test overlap by design because the
        report used one decade for both early stopping and reporting.
        This is a reproduction split, not an unbiased headline.
    ``B``
        Paper split: train 1992--2006, validation 2007--2009, test
        2010--2019.  The three sets are disjoint, so test metrics are
        unbiased.
    ``none``
        Ad-hoc split.  The leakage checks still run, but no protocol
        specific overlap rule is applied.

    Parameters
    ----------
    protocol : str or None
        Protocol name.  Matching is case-insensitive and surrounding
        whitespace is ignored.  ``None`` and the empty string map to
        ``"none"``.

    Returns
    -------
    str
        ``"A"``, ``"B"`` or ``"none"``.

    Raises
    ------
    ValueError
        If ``protocol`` is not one of the accepted names.
    """
    if protocol is None:
        return "none"
    value = str(protocol).strip()
    if not value:
        return "none"
    lowered = value.lower()
    if lowered == "none":
        return "none"
    if lowered in ("a", "b"):
        return lowered.upper()
    raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")


def parse_years(spec) -> list:
    """Parse ``"1992-2009"``, ``"1992,2000"``, a list, or ``None``."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple, set, range)):
        return sorted(int(y) for y in spec)
    s = str(spec).strip()
    if not s:
        return []
    if s.lower() == "all":
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def overlap(a: Iterable[int], b: Iterable[int]) -> list:
    """Sorted intersection of two year collections."""
    return sorted(set(int(y) for y in a) & set(int(y) for y in b))


def _fmt(years: Sequence[int]) -> str:
    years = sorted(set(int(y) for y in years))
    if not years:
        return "∅"
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}–{years[-1]}"


def validate_split(
    train: Sequence[int],
    val: Sequence[int],
    test: Optional[Sequence[int]] = None,
    protocol: str = "none",
    *,
    strict: bool = True,
) -> list:
    """Validate a train/validation/test split and return warnings.

    Parameters
    ----------
    train, val : sequence of int
    test : sequence of int, optional
        Required for ``protocol="B"``.  For ``protocol="A"`` it must equal the
        validation decade, because the report used the same decade for early
        stopping and reporting.
    protocol : {"A", "B", "none"}
        Case-insensitive; see :func:`normalise_protocol`.
    strict : bool
        Raise :class:`LeakageError` on violations.  When False, return the
        messages instead.

    Returns
    -------
    list of str
        Warnings that do not invalidate the split (for example Protocol A's
        deliberate validation/test overlap).
    """
    protocol = normalise_protocol(protocol)

    problems = []
    warnings = []
    train = sorted(set(int(y) for y in train))
    val = sorted(set(int(y) for y in val))
    test = sorted(set(int(y) for y in test)) if test else []

    if not train:
        problems.append("training years are empty")
    if not val:
        problems.append("validation years are empty")
    if overlap(train, val):
        problems.append(f"train ∩ val = {_fmt(overlap(train, val))} (must be empty)")

    if protocol == "B":
        if not test:
            problems.append("protocol B requires explicit test years")
        if overlap(train, test):
            problems.append(f"train ∩ test = {_fmt(overlap(train, test))}")
        if overlap(val, test):
            problems.append(f"val ∩ test = {_fmt(overlap(val, test))}")
    elif protocol == "A":
        if test and sorted(set(test)) != val:
            problems.append(
                "protocol A reports on the validation decade; test years "
                f"({_fmt(test)}) must equal val years ({_fmt(val)})"
            )
        if test and overlap(val, test):
            warnings.append(
                "protocol A: validation and test overlap by design "
                f"({_fmt(overlap(val, test))}); this is a reproduction split, "
                "not a paper headline"
            )
    else:
        if test and overlap(train, test):
            problems.append(f"train ∩ test = {_fmt(overlap(train, test))}")
        if test and overlap(val, test):
            problems.append(f"val ∩ test = {_fmt(overlap(val, test))}")

    if problems:
        msg = "split validation failed:\n  - " + "\n  - ".join(problems)
        if strict:
            raise LeakageError(msg)
        warnings.extend(problems)
    return warnings


def validate_normalisation_years(
    norm_years: Sequence[int],
    train_years: Sequence[int],
    *,
    strict: bool = True,
) -> list:
    """Ensure normalisation statistics were built on training years only."""
    norm_years = sorted(set(int(y) for y in norm_years))
    train_years = sorted(set(int(y) for y in train_years))
    if not norm_years:
        return []
    leaked = sorted(set(norm_years) - set(train_years))
    if leaked:
        msg = (
            "normalisation statistics include non-training years: "
            f"{_fmt(leaked)} (train = {_fmt(train_years)})"
        )
        if strict:
            raise LeakageError(msg)
        return [msg]
    return []


def validate_climatology_years(
    clim_years: Sequence[int],
    train_years: Sequence[int],
    predict_years: Optional[Sequence[int]] = None,
    *,
    strict: bool = True,
) -> list:
    """Ensure the anomaly climatology uses training years only.

    ``predict_years`` is checked for overlap separately: predicting on training
    years with a training-year climatology is expected and therefore only a
    warning.
    """
    clim_years = sorted(set(int(y) for y in clim_years))
    train_years = sorted(set(int(y) for y in train_years))
    warnings = []
    if not clim_years:
        return warnings
    leaked = sorted(set(clim_years) - set(train_years))
    if leaked:
        msg = (
            "anomaly climatology includes non-training years: "
            f"{_fmt(leaked)} (train = {_fmt(train_years)})"
        )
        if strict:
            raise LeakageError(msg)
        warnings.append(msg)
    if predict_years:
        both = overlap(clim_years, predict_years)
        if both:
            warnings.append(
                "anomaly climatology overlaps prediction years "
                f"({_fmt(both)}); expected for train-set diagnostics, invalid "
                "for reported test metrics"
            )
    return warnings


def reconcile_resumed_clim_years(args, explicit, logger) -> None:
    """Drop a stored climatology window that would leak non-training years.

    Checkpoints written before the paper protocol stored the old
    ``--clim_years 1992-2010`` default.  Resuming such a run would merge that
    value back over the new empty default and then fail the leakage check,
    because 2010 is a validation/test year in Protocol A.  Only values that
    came from the checkpoint are reconciled; an explicit CLI value is left
    alone so the strict check can reject it.

    Parameters
    ----------
    args : argparse.Namespace
        Resolved arguments; ``clim_years`` is updated in place.
    explicit : set of str
        Names the user wrote on the command line.
    logger : object
        Logger with a ``warning`` method.
    """
    if not getattr(args, "anomaly_mode", False):
        return
    if "clim_years" in explicit:
        return
    stored = parse_years(getattr(args, "clim_years", ""))
    if not stored:
        return
    train = parse_years(getattr(args, "train_years", ""))
    leaked = sorted(set(stored) - set(train))
    if not leaked:
        return
    fallback = getattr(args, "train_years", "")
    logger.warning(
        "Stored --clim_years "
        f"{min(stored)}-{max(stored)} includes non-training years "
        f"{min(leaked)}-{max(leaked)}; rebuilding the anomaly climatology from "
        f"--train_years ({fallback}) to avoid leakage."
    )
    args.clim_years = fallback
