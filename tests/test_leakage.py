"""Unit tests for :mod:`phenonn.evaluation.leakage`."""

import pytest

from phenonn.evaluation.leakage import (
    LeakageError,
    parse_years,
    validate_climatology_years,
    validate_normalisation_years,
    validate_split,
)


def test_parse_years():
    assert parse_years("1992-2009") == list(range(1992, 2010))
    assert parse_years("2010,2012") == [2010, 2012]
    assert parse_years("") == []
    assert parse_years([2000, 1999]) == [1999, 2000]


def test_protocol_b_valid():
    validate_split(
        list(range(1992, 2007)),
        [2007, 2008, 2009],
        list(range(2010, 2020)),
        protocol="B",
    )


def test_protocol_b_rejects_train_val_overlap():
    with pytest.raises(LeakageError, match="train ∩ val"):
        validate_split([1992, 1993], [1993, 1994], [2010], protocol="B")


def test_protocol_b_rejects_val_test_overlap():
    with pytest.raises(LeakageError, match="val ∩ test"):
        validate_split([1992], [2007], [2007, 2010], protocol="B")


def test_protocol_b_requires_test():
    with pytest.raises(LeakageError, match="requires explicit test"):
        validate_split([1992], [2007], [], protocol="B")


def test_protocol_a_allows_same_decade():
    warnings = validate_split([1992], [2010], [2010], protocol="A")
    assert any("overlap by design" in w for w in warnings)


def test_protocol_a_rejects_different_test_decade():
    with pytest.raises(LeakageError, match="must equal val years"):
        validate_split([1992], [2010], [2011], protocol="A")


def test_norm_stats_must_be_training_years():
    validate_normalisation_years([1992, 2000], [1992, 2000, 2001])
    with pytest.raises(LeakageError, match="non-training years"):
        validate_normalisation_years([1992, 2015], [1992, 2000])


def test_climatology_must_be_training_years():
    validate_climatology_years([1992, 2000], [1992, 2000, 2001], [2010])
    with pytest.raises(LeakageError, match="non-training years"):
        validate_climatology_years([1992, 2010], [1992, 2000], [2010])


def test_climatology_overlap_with_prediction_is_warning_only():
    warnings = validate_climatology_years([1992], [1992, 2000], [1992])
    assert any("overlaps prediction years" in w for w in warnings)
