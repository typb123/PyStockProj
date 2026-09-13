"""Technical indicator tests for past-looking features used by the model contract."""

import numpy as np
import pandas as pd

from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    BASE_MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from src.data.data_prep import add_benchmark_relative_momentum
from src.data.technical_indicators import calculate_data


def test_chikou_features_preserve_lagged_close_timing_without_nominal_scale():
    close = np.arange(100.0, 140.0)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(1000.0, 1040.0),
        }
    )

    result = calculate_data(df)

    expected_return = (df.loc[26, "Close"] - df.loc[0, "Close"]) / df.loc[0, "Close"]

    assert np.isclose(
        result.loc[26, "chikou_lag_close_26_to_close"],
        df.loc[0, "Close"] / df.loc[26, "Close"] - 1,
    )
    assert result.loc[26, "chikou_return_26"] == expected_return
    assert result.loc[26, "chikou_above_lag_26"] == int(df.loc[26, "Close"] > df.loc[0, "Close"])
    assert "chikou_span" not in result.columns
    assert "chikou_span" not in BASE_MODEL_FEATURE_COLUMNS
    assert "chikou_lag_close_26_to_close" in BASE_MODEL_FEATURE_COLUMNS
    assert "chikou_return_26" in BASE_MODEL_FEATURE_COLUMNS
    assert "chikou_above_lag_26" in BASE_MODEL_FEATURE_COLUMNS


def test_calculate_data_creates_trailing_momentum_columns():
    close = np.arange(100.0, 160.0)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(1000.0, 1060.0),
        }
    )

    result = calculate_data(df)

    for column in ["momentum_5d", "momentum_10d", "momentum_20d", "momentum_50d"]:
        assert column in result.columns
        assert column in ABSOLUTE_MOMENTUM_FEATURE_COLUMNS
        assert column in MODEL_FEATURE_COLUMNS


def test_trailing_momentum_columns_are_past_looking():
    close = np.array([100.0 * (1.1 ** index) for index in range(60)])
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(1000.0, 1060.0),
        }
    )

    result = calculate_data(df)

    assert np.isnan(result.loc[4, "momentum_5d"])
    expected_5d = close[5] / close[0] - 1
    expected_50d = close[50] / close[0] - 1
    assert np.isclose(result.loc[5, "momentum_5d"], expected_5d)
    assert np.isclose(result.loc[50, "momentum_50d"], expected_50d)


def test_calculate_data_does_not_generate_retired_signed_volume_features():
    close = np.arange(100.0, 130.0)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.arange(1_000.0, 1_030.0),
        }
    )

    result = calculate_data(df)

    assert "rolling_signed_volume_20d" not in result.columns
    assert "obv" not in result.columns


def test_base_model_features_match_after_history_start_boundary_has_warmed_up():
    """A suffix with enough context must match a longer history at one date.

    The long warmup lets recursive EMA features converge before the target date.
    """
    periods = 520
    steps = np.arange(periods, dtype=float)
    close = 100.0 + 0.08 * steps + 3.0 * np.sin(steps / 5.0) + np.cos(steps / 13.0)
    volume = 1_000_000.0 + 1_000.0 * steps + 10_000.0 * (steps % 7)
    history = pd.DataFrame(
        {
            "Open": close * (0.997 + 0.001 * np.cos(steps / 9.0)),
            "High": close * 1.015,
            "Low": close * 0.985,
            "Close": close,
            "Volume": volume,
        },
        index=pd.bdate_range("2020-01-02", periods=periods),
    )
    short_history = history.iloc[-260:]
    target_date = history.index[-1]

    long_features = calculate_data(history)
    short_features = calculate_data(short_history)

    np.testing.assert_allclose(
        long_features.loc[target_date, BASE_MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
        short_features.loc[target_date, BASE_MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
        equal_nan=True,
    )

    assert "rolling_signed_volume_20d" not in long_features.columns
    assert "rolling_signed_volume_20d" not in short_features.columns


def test_remediated_model_features_are_invariant_to_common_price_scaling():
    index = np.arange(90, dtype=float)
    close = 100.0 + index + 3.0 * np.sin(index / 3.0)
    raw = pd.DataFrame(
        {
            "Open": close * (0.995 + 0.001 * np.cos(index)),
            "High": close * 1.015,
            "Low": close * 0.985,
            "Close": close,
            "Volume": 1_000_000.0 + index * 1000.0,
        }
    )
    scaled = raw.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 7.5

    def model_feature_frame(ohlcv):
        stock = calculate_data(ohlcv)
        spy_ohlcv = ohlcv.copy()
        spy_ohlcv[["Open", "High", "Low", "Close"]] *= 1.5
        spy = calculate_data(spy_ohlcv)
        dates = pd.bdate_range("2024-01-01", periods=len(ohlcv))
        stock["Ticker"] = "AAA"
        stock["prediction_date"] = dates
        spy["Ticker"] = "SPY"
        spy["prediction_date"] = dates
        return add_benchmark_relative_momentum(
            pd.concat([stock, spy], ignore_index=True)
        )

    baseline_features = model_feature_frame(raw).query("Ticker == 'AAA'")[
        MODEL_FEATURE_COLUMNS
    ]
    scaled_features = model_feature_frame(scaled).query("Ticker == 'AAA'")[
        MODEL_FEATURE_COLUMNS
    ]

    np.testing.assert_allclose(
        scaled_features.to_numpy(dtype=float),
        baseline_features.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True,
    )
