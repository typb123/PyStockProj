"""Column ownership for model features, targets, and evaluation metadata."""

RAW_PRICE_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

TREND_FEATURE_COLUMNS = [
    "5_day_avg",
    "10_day_avg",
    "20_day_avg",
]

RETURN_RISK_FEATURE_COLUMNS = [
    "dailyReturn",
    "volatility",
]

MOMENTUM_OSCILLATOR_FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "signalLine",
    "macdHistogram",
]

VOLUME_INDICATOR_FEATURE_COLUMNS = [
    "obv",
    "vma_10",
    "vma_20",
]

ICHIMOKU_FEATURE_COLUMNS = [
    "tenkan_sen",
    "kijun_sen",
    "senkou_span_a",
    "senkou_span_b",
    "chikou_lag_close_26",
    "chikou_return_26",
    "chikou_above_lag_26",
]

BOLLINGER_FEATURE_COLUMNS = [
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "BB_Std",
]

ATR_FEATURE_COLUMNS = [
    "ATR",
]

STOCHASTIC_FEATURE_COLUMNS = [
    "stoch_k",
    "stoch_d",
]

ABSOLUTE_MOMENTUM_FEATURE_COLUMNS = [
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_50d",
]

SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS = [
    "relative_momentum_5d",
    "relative_momentum_10d",
    "relative_momentum_20d",
    "relative_momentum_50d",
]

MODEL_FEATURE_GROUPS = {
    "raw_price": RAW_PRICE_FEATURE_COLUMNS,
    "trend": TREND_FEATURE_COLUMNS,
    "return_risk": RETURN_RISK_FEATURE_COLUMNS,
    "momentum_oscillator": MOMENTUM_OSCILLATOR_FEATURE_COLUMNS,
    "volume_indicator": VOLUME_INDICATOR_FEATURE_COLUMNS,
    "ichimoku": ICHIMOKU_FEATURE_COLUMNS,
    "bollinger": BOLLINGER_FEATURE_COLUMNS,
    "atr": ATR_FEATURE_COLUMNS,
    "stochastic": STOCHASTIC_FEATURE_COLUMNS,
    "absolute_momentum": ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    "spy_relative_momentum": SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
}

TARGET_COLUMNS = [
    "raw_forward_return",
    "benchmark_forward_return",
    "excess_forward_return",
    "targetReturns",
    "beat_benchmark_target",
]

FORWARD_RETURN_METADATA_COLUMNS = [
    "forward_end_date",
    "benchmark_forward_end_date",
]

TARGET_COLUMNS = TARGET_COLUMNS + FORWARD_RETURN_METADATA_COLUMNS

RANKING_TARGET_COLUMNS = [
    "excess_return_rank_pct_by_date",
    "top_quintile_target",
    "ranking_train_sample",
]

NON_FEATURE_COLUMNS = TARGET_COLUMNS + RANKING_TARGET_COLUMNS

SPLIT_METADATA_COLUMNS = [
    "_source_index",
    "Ticker",
    "prediction_date",
    "dailyReturn",
    "raw_forward_return",
    "benchmark_forward_return",
    "excess_forward_return",
    "beat_benchmark_target",
    *FORWARD_RETURN_METADATA_COLUMNS,
]


def flatten_feature_groups(feature_groups: dict[str, list[str]]) -> list[str]:
    """Flatten ordered feature groups into the final model feature order."""
    return [
        column
        for feature_columns in feature_groups.values()
        for column in feature_columns
    ]


MODEL_FEATURE_COLUMNS = flatten_feature_groups(MODEL_FEATURE_GROUPS)

BASE_MODEL_FEATURE_COLUMNS = flatten_feature_groups(
    {
        group_name: feature_columns
        for group_name, feature_columns in MODEL_FEATURE_GROUPS.items()
        if group_name != "spy_relative_momentum"
    }
)


def validate_feature_contract(
    feature_groups: dict[str, list[str]] = MODEL_FEATURE_GROUPS,
    target_columns: list[str] = NON_FEATURE_COLUMNS,
) -> None:
    """Fail fast if feature groups are empty, duplicated, or overlap targets."""
    empty_groups = [
        group_name
        for group_name, feature_columns in feature_groups.items()
        if not feature_columns
    ]
    if empty_groups:
        raise ValueError(f"Feature groups must not be empty: {empty_groups}")

    model_features = flatten_feature_groups(feature_groups)
    duplicate_features = sorted(
        {
            feature
            for feature in model_features
            if model_features.count(feature) > 1
        }
    )
    if duplicate_features:
        raise ValueError(f"Duplicate model feature columns: {duplicate_features}")

    target_overlap = sorted(set(model_features).intersection(target_columns))
    if target_overlap:
        raise ValueError(
            f"Target columns must not overlap model features: {target_overlap}"
        )


validate_feature_contract()
