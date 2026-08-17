"""Focused tests for lightweight training target and feature policy."""

import pytest

from src.features.feature_contract import MODEL_FEATURE_COLUMNS
import src.train.training_contract as training_contract


def test_target_mode_constants_and_collections_preserve_order():
    assert training_contract.TARGET_MODE_EXCESS_RETURN == "excess_return"
    assert (
        training_contract.TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM
        == "cross_sectional_top_bottom"
    )
    assert (
        training_contract.TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG
        == "cross_sectional_rank_ndcg"
    )
    assert training_contract.TARGET_MODES == (
        "excess_return",
        "cross_sectional_top_bottom",
        "cross_sectional_rank_ndcg",
    )
    assert training_contract.WALK_FORWARD_TARGET_MODES == (
        "excess_return",
        "cross_sectional_rank_ndcg",
    )


@pytest.mark.parametrize("target_mode", training_contract.TARGET_MODES)
def test_validate_target_mode_accepts_supported_modes(target_mode):
    assert training_contract.validate_target_mode(target_mode) is None


def test_validate_target_mode_rejects_unknown_target_mode():
    with pytest.raises(ValueError) as exc_info:
        training_contract.validate_target_mode("bad_mode")

    assert str(exc_info.value) == (
        "Unsupported target_mode='bad_mode'. Expected one of "
        "excess_return, cross_sectional_top_bottom, cross_sectional_rank_ndcg."
    )


@pytest.mark.parametrize("target_mode", training_contract.WALK_FORWARD_TARGET_MODES)
def test_validate_walk_forward_target_mode_accepts_supported_modes(target_mode):
    assert training_contract.validate_walk_forward_target_mode(target_mode) is None


def test_validate_walk_forward_target_mode_rejects_unsupported_mode():
    with pytest.raises(ValueError) as exc_info:
        training_contract.validate_walk_forward_target_mode(
            "cross_sectional_top_bottom"
        )

    assert str(exc_info.value) == (
        "Unsupported walk-forward target_mode='cross_sectional_top_bottom'. "
        "Expected one of excess_return, cross_sectional_rank_ndcg."
    )


def test_resolve_model_feature_columns_defaults_to_a_copy_of_canonical_order():
    feature_columns = training_contract.resolve_model_feature_columns()

    assert feature_columns == MODEL_FEATURE_COLUMNS
    assert feature_columns is not MODEL_FEATURE_COLUMNS


def test_resolve_model_feature_columns_preserves_requested_feature_order():
    feature_columns = training_contract.resolve_model_feature_columns(
        ["momentum_10d", "volatility"]
    )

    assert feature_columns == ["momentum_10d", "volatility"]


def test_resolve_model_feature_columns_rejects_empty_override():
    with pytest.raises(ValueError) as exc_info:
        training_contract.resolve_model_feature_columns([])

    assert str(exc_info.value) == (
        "feature_columns_override must contain at least one feature."
    )


def test_resolve_model_feature_columns_rejects_unknown_features():
    with pytest.raises(ValueError) as exc_info:
        training_contract.resolve_model_feature_columns(["momentum_10d", "targetReturns"])

    assert str(exc_info.value) == (
        "feature_columns_override may only include MODEL_FEATURE_COLUMNS. "
        "Invalid entries: ['targetReturns']"
    )


def test_resolve_model_feature_columns_rejects_duplicate_features():
    with pytest.raises(ValueError) as exc_info:
        training_contract.resolve_model_feature_columns(
            ["volatility", "momentum_10d", "volatility"]
        )

    assert str(exc_info.value) == (
        "feature_columns_override contains duplicate features: ['volatility']"
    )
