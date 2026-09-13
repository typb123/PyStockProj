"""Feature contract tests for model-feature ordering and validation guarantees."""

import pytest

from src.features.feature_contract import (
    FORWARD_RETURN_METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_GROUPS,
    NOMINAL_PRICE_SCALE_SENSITIVE_FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    RANKING_TARGET_COLUMNS,
    SPLIT_METADATA_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    validate_feature_contract,
)


EXPECTED_MODEL_FEATURE_COLUMNS = [
    "open_to_close",
    "high_to_close",
    "low_to_close",
    "Volume",
    "sma_5_to_close",
    "sma_10_to_close",
    "sma_20_to_close",
    "dailyReturn",
    "volatility",
    "rsi",
    "macd_to_close",
    "signal_line_to_close",
    "macd_histogram_to_close",
    "vma_10",
    "vma_20",
    "tenkan_sen_to_close",
    "kijun_sen_to_close",
    "senkou_span_a_to_close",
    "senkou_span_b_to_close",
    "chikou_lag_close_26_to_close",
    "chikou_return_26",
    "chikou_above_lag_26",
    "bb_middle_to_close",
    "bb_upper_to_close",
    "bb_lower_to_close",
    "bb_std_to_close",
    "atr_to_close",
    "stoch_k",
    "stoch_d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_50d",
    "relative_momentum_5d",
    "relative_momentum_10d",
    "relative_momentum_20d",
    "relative_momentum_50d",
]


def test_model_feature_columns_preserve_expected_order():
    assert MODEL_FEATURE_COLUMNS == EXPECTED_MODEL_FEATURE_COLUMNS
    assert len(MODEL_FEATURE_COLUMNS) == 37
    assert "rolling_signed_volume_20d" not in MODEL_FEATURE_COLUMNS


def test_model_feature_contract_has_no_duplicates_or_target_overlap():
    validate_feature_contract()
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS))
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(TARGET_COLUMNS)
    assert set(TARGET_COLUMNS).issubset(NON_FEATURE_COLUMNS)


def test_model_feature_contract_excludes_nominal_price_scale_sensitive_columns():
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(
        NOMINAL_PRICE_SCALE_SENSITIVE_FEATURE_COLUMNS
    )


def test_ranking_target_columns_are_metadata_not_model_features():
    assert RANKING_TARGET_COLUMNS == [
        "excess_return_rank_pct_by_date",
        "top_quintile_target",
        "ranking_train_sample",
    ]
    assert set(RANKING_TARGET_COLUMNS).issubset(NON_FEATURE_COLUMNS)
    assert set(RANKING_TARGET_COLUMNS).issubset(SPLIT_METADATA_COLUMNS)
    assert set(RANKING_TARGET_COLUMNS).isdisjoint(TARGET_COLUMNS)
    assert set(RANKING_TARGET_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)


def test_forward_return_endpoint_columns_are_metadata_not_model_features():
    assert FORWARD_RETURN_METADATA_COLUMNS == [
        "forward_end_date",
        "benchmark_forward_end_date",
    ]
    assert set(FORWARD_RETURN_METADATA_COLUMNS).issubset(TARGET_COLUMNS)
    assert set(FORWARD_RETURN_METADATA_COLUMNS).issubset(NON_FEATURE_COLUMNS)
    assert set(FORWARD_RETURN_METADATA_COLUMNS).issubset(SPLIT_METADATA_COLUMNS)
    assert set(FORWARD_RETURN_METADATA_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)


def test_spy_relative_momentum_features_are_present():
    assert SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS == [
        "relative_momentum_5d",
        "relative_momentum_10d",
        "relative_momentum_20d",
        "relative_momentum_50d",
    ]
    for column in SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS:
        assert column in MODEL_FEATURE_COLUMNS


def test_feature_contract_validation_rejects_duplicate_features():
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["duplicate"] = ["open_to_close"]

    with pytest.raises(ValueError, match="Duplicate model feature columns"):
        validate_feature_contract(feature_groups=feature_groups)


def test_feature_contract_validation_rejects_target_overlap():
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["bad_target_feature"] = ["targetReturns"]

    with pytest.raises(ValueError, match="Target columns must not overlap"):
        validate_feature_contract(feature_groups=feature_groups)


def test_feature_contract_validation_rejects_nominal_price_scale_sensitive_column():
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["bad_nominal_price"] = ["Close"]

    with pytest.raises(ValueError, match="nominal price-scale-sensitive"):
        validate_feature_contract(feature_groups=feature_groups)


@pytest.mark.parametrize("ranking_column", RANKING_TARGET_COLUMNS)
def test_feature_contract_validation_rejects_ranking_target_overlap(ranking_column):
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["bad_ranking_target_feature"] = [ranking_column]

    with pytest.raises(ValueError, match="Target columns must not overlap"):
        validate_feature_contract(feature_groups=feature_groups)


@pytest.mark.parametrize("endpoint_column", FORWARD_RETURN_METADATA_COLUMNS)
def test_feature_contract_validation_rejects_endpoint_metadata_overlap(endpoint_column):
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["bad_endpoint_metadata_feature"] = [endpoint_column]

    with pytest.raises(ValueError, match="Target columns must not overlap"):
        validate_feature_contract(feature_groups=feature_groups)


def test_feature_contract_validation_rejects_empty_groups():
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["empty"] = []

    with pytest.raises(ValueError, match="must not be empty"):
        validate_feature_contract(feature_groups=feature_groups)
