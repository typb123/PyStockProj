import numpy as np
import pandas as pd

from src.config import REQUIRED_COLUMNS
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
    assert "chikou_span" not in REQUIRED_COLUMNS
    assert "chikou_lag_close_26" in REQUIRED_COLUMNS
    assert "chikou_return_26" in REQUIRED_COLUMNS
    assert "chikou_above_lag_26" in REQUIRED_COLUMNS
