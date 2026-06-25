import pandas as pd

import src.train.trainer as trainer


def test_prepare_data_parallel_passes_period_to_fetch_stock_data(monkeypatch):
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return pd.DataFrame(
            {
                "Close": [100.0],
                "Ticker": [ticker],
            }
        )

    def fake_calculate_data(df):
        return df

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(trainer, "calculate_data", fake_calculate_data)

    result = trainer.prepare_data_parallel(["AAPL", "MSFT"], period="1y")

    assert calls == [("AAPL", "1y"), ("MSFT", "1y")]
    assert result.shape[0] == 2
    assert set(result["Ticker"]) == {"AAPL", "MSFT"}


def test_validate_input_data_does_not_fill_missing_values():
    data = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "BBB", "BBB"],
            "Close": [100.0, None, 500.0, 600.0],
            "Volume": [1000.0, 1100.0, None, 1300.0],
        }
    )

    result = trainer.validate_input_data(data)

    assert pd.isna(result.loc[1, "Close"])
    assert pd.isna(result.loc[2, "Volume"])
    assert result.loc[2, "Close"] == 500.0
    assert result.loc[1, "Volume"] == 1100.0


def test_model_metadata_does_not_include_linear_regression_prediction():
    metadata = trainer.build_model_metadata(
        linear_features=["Close"],
        classifier_features=["Close", "Volume"],
        regressor_features=["Close", "Volume"],
        prediction_days=5,
    )

    assert "LinearRegression_Prediction" not in metadata["classifier_features"]
    assert "LinearRegression_Prediction" not in metadata["regressor_features"]
    assert "LinearRegression_Prediction" not in metadata["classifier_feature_names"]
    assert "LinearRegression_Prediction" not in metadata["regressor_feature_names"]
