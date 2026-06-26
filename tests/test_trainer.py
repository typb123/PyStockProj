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


def test_validate_input_data_logs_nan_details_at_debug(caplog):
    data = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA"],
            "Close": [100.0, None],
        }
    )

    with caplog.at_level(logging.INFO):
        trainer.validate_input_data(data)

    assert "Data validation complete." in caplog.text
    assert "NaN values before preparation" not in caplog.text
    assert "Column Close:" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        trainer.validate_input_data(data)

    assert "NaN values before preparation: 1" in caplog.text
    assert "Column Close: 1 NaN values" in caplog.text


def test_log_feature_importances_summarizes_info_and_keeps_full_debug(caplog):
    feature_importances = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "importance": [0.33333, 0.22222, 0.11111],
        }
    )

    with caplog.at_level(logging.INFO):
        trainer.log_feature_importances("Test Model", feature_importances, top_n=2)

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert info_messages == [
        "Top 2 Feature Importances for Test Model: {'f1': 0.3333, 'f2': 0.2222}"
    ]
    assert all(record.levelno != logging.DEBUG for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        trainer.log_feature_importances("Test Model", feature_importances, top_n=2)

    assert "Feature Importances for Test Model:" in caplog.text
    assert "f1: 0.3333" in caplog.text
    assert "f2: 0.2222" in caplog.text
    assert "f3: 0.1111" in caplog.text


def test_format_top_n_ranked_selection_summary_formats_percentages():
    report = {
        "top_5": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.0123,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_selected_excess_return_vs_benchmark": 0.008,
            },
            "universe": {
                "average_excess_return_vs_benchmark": 0.0015,
            },
        }
    }

    summary = trainer.format_top_n_ranked_selection_summary(report)

    assert summary == (
        "XGBoost Top-N Ranked Selection Summary:\n"
        "top_5:\n"
        "  model excess=1.23%, random excess=0.40%, "
        "momentum excess=0.80%, universe excess=0.15%\n"
        "  model minus random=0.83%, model minus momentum=0.43%, "
        "model beat rate=52.88%"
    )


def test_format_top_n_ranked_selection_summary_handles_unavailable_momentum():
    report = {
        "top_10": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.002,
            },
            "momentum_baseline": {
                "available": False,
                "average_selected_excess_return_vs_benchmark": np.nan,
            },
            "universe": {
                "average_excess_return_vs_benchmark": -0.001,
            },
        }
    }

    summary = trainer.format_top_n_ranked_selection_summary(report)

    assert "momentum excess=unavailable" in summary
    assert "model minus momentum=unavailable" in summary
    assert "universe excess=-0.10%" in summary


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


def test_log_xgboost_test_report_logs_top_n_summary_at_info_and_full_report_at_debug(
    monkeypatch,
    caplog,
):
    top_n_report = {
        "top_5": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.0123,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_selected_excess_return_vs_benchmark": 0.008,
            },
            "universe": {
                "average_excess_return_vs_benchmark": 0.0015,
            },
        }
    }

    monkeypatch.setattr(
        trainer,
        "build_classification_report",
        lambda *args, **kwargs: {"accuracy": 1.0},
    )
    monkeypatch.setattr(
        trainer,
        "build_regression_report",
        lambda *args, **kwargs: {"mse": 0.1},
    )
    monkeypatch.setattr(
        trainer,
        "build_actual_return_baseline_report",
        lambda *args, **kwargs: {"count": 1},
    )
    monkeypatch.setattr(
        trainer,
        "build_trading_relevance_report",
        lambda *args, **kwargs: {"relevance": 1},
    )
    monkeypatch.setattr(
        trainer,
        "build_probability_summary",
        lambda *args, **kwargs: {"count": 1},
    )
    monkeypatch.setattr(
        trainer,
        "build_probability_tail_report",
        lambda *args, **kwargs: {"top": 1},
    )
    monkeypatch.setattr(
        trainer,
        "build_probability_threshold_report",
        lambda *args, **kwargs: {0.5: {"selected_count": 1}},
    )
    monkeypatch.setattr(
        trainer,
        "build_validation_selected_threshold_report",
        lambda *args, **kwargs: {"selected_threshold": 0.5},
    )
    monkeypatch.setattr(
        trainer,
        "build_predicted_return_quantile_report",
        lambda *args, **kwargs: {"top_10_pct": {}},
    )
    monkeypatch.setattr(
        trainer,
        "build_return_correlation_report",
        lambda *args, **kwargs: {"pearson": 0.0},
    )
    monkeypatch.setattr(
        trainer,
        "build_combined_signal_report",
        lambda *args, **kwargs: {"selected_count": 0},
    )
    monkeypatch.setattr(
        trainer,
        "build_top_n_ranked_selection_report",
        lambda *args, **kwargs: top_n_report,
    )

    with caplog.at_level(logging.INFO):
        trainer.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=pd.DataFrame(),
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=np.array([0.03]),
        )

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert any(
        message.startswith("XGBoost Top-N Ranked Selection Summary:")
        for message in info_messages
    )
    assert not any(
        message.startswith("XGBoost Top-N Ranked Selection Report:")
        for message in info_messages
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        trainer.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=pd.DataFrame(),
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=np.array([0.03]),
        )

    assert "XGBoost Top-N Ranked Selection Report:" in caplog.text
