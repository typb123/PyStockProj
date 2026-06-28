"""Technical indicator tests for past-looking features used by the model contract."""

import numpy as np
import pandas as pd

from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    BASE_MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from src.data.technical_indicators import calculate_data


def test_chikou_features_use_lagged_close_only():
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

    assert result.loc[26, "chikou_lag_close_26"] == df.loc[0, "Close"]
    assert result.loc[26, "chikou_return_26"] == expected_return
    assert result.loc[26, "chikou_above_lag_26"] == int(df.loc[26, "Close"] > df.loc[0, "Close"])
    assert "chikou_span" not in result.columns
    assert "chikou_span" not in BASE_MODEL_FEATURE_COLUMNS
    assert "chikou_lag_close_26" in BASE_MODEL_FEATURE_COLUMNS
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
