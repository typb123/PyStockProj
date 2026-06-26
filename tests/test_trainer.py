import logging
import pandas as pd
import numpy as np

import src.train.trainer as trainer


def test_prepare_data_parallel_passes_period_to_fetch_stock_data(monkeypatch):
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return pd.DataFrame(
            {
                "Close": [100.0],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )

    def fake_calculate_data(df):
        return df

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(trainer, "calculate_data", fake_calculate_data)

    result = trainer.prepare_data_parallel(["AAPL", "MSFT"], period="1y")

    assert calls == [("AAPL", "1y"), ("MSFT", "1y")]
    assert result.shape[0] == 2
    assert set(result["Ticker"]) == {"AAPL", "MSFT"}
    assert "prediction_date" in result.columns
    assert set(result["prediction_date"]) == {pd.Timestamp("2024-01-02")}


def test_prepare_data_parallel_summarizes_skipped_tickers(monkeypatch, caplog):
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        if ticker == "BAD":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "Close": [100.0],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )

    def fake_calculate_data(df):
        return df

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(trainer, "calculate_data", fake_calculate_data)

    with caplog.at_level(logging.INFO):
        result = trainer.prepare_data_parallel(["AAPL", "BAD", "MSFT"], period="1y")

    assert calls == [("AAPL", "1y"), ("BAD", "1y"), ("MSFT", "1y")]
    assert set(result["Ticker"]) == {"AAPL", "MSFT"}
    assert "Fetched valid data for 2 of 3 requested tickers." in caplog.text
    assert "Skipped 1 tickers with no usable data: ['BAD']" in caplog.text
    assert "No data returned for BAD. Skipping..." not in caplog.text


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


def test_log_xgboost_test_report_uses_explicit_direction_labels(monkeypatch):
    captured = {}

    def fake_build_classification_report(y_true, y_pred):
        captured["y_true"] = np.asarray(y_true)
        captured["y_pred"] = np.asarray(y_pred)
        return {"accuracy": 1.0}

    def fake_build_top_n_ranked_selection_report(split_metadata, predicted_returns):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(predicted_returns)
        return {"top_5": {"selected_row_count": 1}}

    monkeypatch.setattr(
        trainer,
        "build_classification_report",
        fake_build_classification_report,
    )
    monkeypatch.setattr(
        trainer,
        "build_top_n_ranked_selection_report",
        fake_build_top_n_ranked_selection_report,
    )

    y_test = np.array([0.10, -0.20, 0.30])
    direction_y_test = np.array([0, 1, 0])
    classifier_predictions = np.array([0, 1, 1])
    test_split_metadata = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "prediction_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02"]
            ),
            "raw_forward_return": [0.11, -0.18, 0.28],
            "benchmark_forward_return": [0.01, 0.02, -0.02],
            "excess_forward_return": y_test,
            "beat_benchmark_target": direction_y_test,
        }
    )
    regressor_predictions = np.array([0.08, -0.15, 0.20])

    trainer.log_xgboost_test_report(
        y_train=np.array([0.01, -0.02, 0.03]),
        y_val=np.array([0.02, -0.01, 0.04]),
        y_test=y_test,
        direction_y_test=direction_y_test,
        test_split_metadata=test_split_metadata,
        classifier_predictions=classifier_predictions,
        classifier_validation_probability_up=np.array([0.55, 0.45, 0.65]),
        classifier_probability_up=np.array([0.60, 0.40, 0.70]),
        regressor_predictions=regressor_predictions,
    )

    np.testing.assert_array_equal(captured["y_true"], direction_y_test)
    np.testing.assert_array_equal(captured["y_pred"], classifier_predictions)
    pd.testing.assert_frame_equal(captured["split_metadata"], test_split_metadata)
    np.testing.assert_array_equal(captured["ranked_predictions"], regressor_predictions)
    assert not np.array_equal(captured["y_true"], (y_test > 0).astype(int))
