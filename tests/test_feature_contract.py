import pytest

from src.features.feature_contract import (
    MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_GROUPS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    validate_feature_contract,
)


EXPECTED_MODEL_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "5_day_avg",
    "10_day_avg",
    "20_day_avg",
    "dailyReturn",
    "volatility",
    "rsi",
    "macd",
    "signalLine",
    "macdHistogram",
    "obv",
    "vma_10",
    "vma_20",
    "tenkan_sen",
    "kijun_sen",
    "senkou_span_a",
    "senkou_span_b",
    "chikou_lag_close_26",
    "chikou_return_26",
    "chikou_above_lag_26",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "BB_Std",
    "ATR",
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


def test_model_feature_contract_has_no_duplicates_or_target_overlap():
    validate_feature_contract()
    assert len(MODEL_FEATURE_COLUMNS) == len(set(MODEL_FEATURE_COLUMNS))
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(TARGET_COLUMNS)


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
    feature_groups["duplicate"] = ["Open"]

    with pytest.raises(ValueError, match="Duplicate model feature columns"):
        validate_feature_contract(feature_groups=feature_groups)


def test_feature_contract_validation_rejects_target_overlap():
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["bad_target_feature"] = ["targetReturns"]

    with pytest.raises(ValueError, match="Target columns must not overlap"):
        validate_feature_contract(feature_groups=feature_groups)


def test_feature_contract_validation_rejects_empty_groups():
    feature_groups = dict(MODEL_FEATURE_GROUPS)
    feature_groups["empty"] = []

    with pytest.raises(ValueError, match="must not be empty"):
        validate_feature_contract(feature_groups=feature_groups)
