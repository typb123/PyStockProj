import logging
import pandas as pd
import numpy as np
import pytest

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
                "momentum_score_column": "momentum_10d",
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
        "momentum_10d excess=0.80%, universe excess=0.15%\n"
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


def test_format_top_n_basket_backtest_summary_formats_normal_report():
    report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "momentum_score_column": "momentum_10d",
                "average_basket_excess_return": 0.008,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
            },
        }
    }

    summary = trainer.format_top_n_basket_backtest_summary(report)

    assert summary == (
        "XGBoost Top-N Basket Backtest Summary:\n"
        "top_5:\n"
        "  model raw=1.23%, model excess=1.00%, random excess=0.40%, "
        "momentum_10d excess=0.80%, universe excess=0.15%, benchmark raw=0.23%\n"
        "  model minus random=0.60%, model minus momentum=0.20%, "
        "beat benchmark rate=52.88%"
    )


def test_format_top_n_basket_backtest_summary_handles_unavailable_momentum():
    report = {
        "top_10": {
            "model": {
                "average_basket_raw_return": 0.012,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.002,
            },
            "momentum_baseline": {
                "available": False,
                "average_basket_excess_return": np.nan,
            },
            "universe": {
                "average_basket_excess_return": -0.001,
            },
            "benchmark": {
                "average_basket_raw_return": 0.003,
            },
        }
    }

    summary = trainer.format_top_n_basket_backtest_summary(report)

    assert "momentum excess=unavailable" in summary
    assert "model minus momentum=unavailable" in summary
    assert "universe excess=-0.10%" in summary
    assert "benchmark raw=0.30%" in summary


def test_format_horizon_comparison_summary_formats_basket_metrics():
    horizon_reports = {
        5: {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.01},
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": True,
                        "average_basket_excess_return": 0.003,
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.004},
                }
            }
        },
        10: {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": -0.01},
                    "random_baseline": {"average_basket_excess_return": 0.001},
                    "momentum_baseline": {
                        "available": False,
                        "average_basket_excess_return": np.nan,
                    },
                    "universe": {"average_basket_excess_return": -0.002},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            }
        },
    }

    summary = trainer.format_horizon_comparison_summary(horizon_reports)

    assert summary == (
        "Horizon Comparison Summary:\n"
        "5d:\n"
        "  top_5 model excess=1.00%, random=0.20%, momentum=0.30%, "
        "universe=0.10%, benchmark raw=0.40%\n"
        "10d:\n"
        "  top_5 model excess=-1.00%, random=0.10%, momentum=unavailable, "
        "universe=-0.20%, benchmark raw=0.30%"
    )


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
    assert metadata["prediction_days"] == 5


def test_parse_args_defaults_to_ten_prediction_days():
    args = trainer.parse_args([])

    assert args.prediction_days == 10
    assert args.horizons == [10]
    assert args.all_horizons is False


def test_parse_args_accepts_prediction_days_long_and_short_flags():
    long_args = trainer.parse_args(["--prediction-days", "5"])
    short_args = trainer.parse_args(["-d", "5"])

    assert long_args.prediction_days == 5
    assert long_args.horizons == [5]
    assert short_args.prediction_days == 5
    assert short_args.horizons == [5]


def test_parse_args_rejects_invalid_prediction_days_values():
    for argv in [
        ["--prediction-days", "0"],
        ["--prediction-days", "-1"],
        ["--prediction-days", "abc"],
    ]:
        with pytest.raises(SystemExit):
            trainer.parse_args(argv)


def test_parse_args_all_horizons_resolves_research_horizons():
    args = trainer.parse_args(["--all-horizons"])

    assert args.all_horizons is True
    assert args.horizons == [5, 10, 20, 50]


def test_parse_args_rejects_all_horizons_with_prediction_days():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--all-horizons", "--prediction-days", "5"])


def test_build_horizon_model_paths_uses_horizon_specific_directory():
    paths = trainer.build_horizon_model_paths(20)

    assert paths["model_metadata"] == "models/horizon_20/model_metadata.pkl"
    assert paths["classifier"] == "models/horizon_20/xgboost_classifier.json"
    assert paths["regressor"] == "models/horizon_20/xgboost_regressor.json"


def test_save_horizon_model_artifacts_writes_horizon_specific_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FakeXgbModel:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as model_file:
                model_file.write("model")

    saved_paths = trainer.save_horizon_model_artifacts(
        5,
        linear_model={"linear": True},
        scaler_lr={"scaler": True},
        data_preparator={"preparator": True},
        all_features=["Close"],
        model_metadata={"prediction_days": 5},
        classifier=FakeXgbModel(),
        regressor=FakeXgbModel(),
    )

    assert saved_paths["model_metadata"] == "models/horizon_5/model_metadata.pkl"
    assert (tmp_path / "models/horizon_5/model_metadata.pkl").exists()
    assert (tmp_path / "models/horizon_5/xgboost_classifier.json").exists()
    assert not (tmp_path / "models/model_metadata.pkl").exists()


def test_save_horizon_model_artifacts_preserves_legacy_paths_for_default_horizon(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    class FakeXgbModel:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as model_file:
                model_file.write("model")

    trainer.save_horizon_model_artifacts(
        10,
        linear_model={"linear": True},
        scaler_lr={"scaler": True},
        data_preparator={"preparator": True},
        all_features=["Close"],
        model_metadata={"prediction_days": 10},
        classifier=FakeXgbModel(),
        regressor=FakeXgbModel(),
    )

    assert (tmp_path / "models/horizon_10/model_metadata.pkl").exists()
    assert (tmp_path / "models/model_metadata.pkl").exists()


def test_main_trains_all_horizons_and_logs_comparison(monkeypatch, caplog):
    trained_horizons = []

    monkeypatch.setattr(
        trainer,
        "prepare_data_parallel",
        lambda tickers, period="5y": pd.DataFrame({"Close": [1.0]}),
    )

    def fake_train_models(data, prediction_days):
        trained_horizons.append(prediction_days)
        return {
            "prediction_days": prediction_days,
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": prediction_days / 1000},
                    "random_baseline": {"average_basket_excess_return": 0.001},
                    "momentum_baseline": {
                        "available": True,
                        "average_basket_excess_return": 0.002,
                    },
                    "universe": {"average_basket_excess_return": 0.0},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    with caplog.at_level(logging.INFO):
        trainer.main(["--all-horizons"])

    assert trained_horizons == [5, 10, 20, 50]
    assert "Horizon Comparison Summary:" in caplog.text
    assert "50d:" in caplog.text


def test_log_xgboost_test_report_uses_explicit_direction_labels(monkeypatch):
    captured = {}

    def fake_build_classification_report(y_true, y_pred):
        captured["y_true"] = np.asarray(y_true)
        captured["y_pred"] = np.asarray(y_pred)
        return {"accuracy": 1.0}

    def fake_build_top_n_ranked_selection_report(
        split_metadata,
        predicted_returns,
        prediction_days=None,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(predicted_returns)
        captured["prediction_days"] = prediction_days
        return {"top_5": {"selected_row_count": 1}}

    def fake_build_top_n_basket_backtest_report(
        split_metadata,
        predicted_returns,
        prediction_days=None,
    ):
        captured["basket_split_metadata"] = split_metadata
        captured["basket_ranked_predictions"] = np.asarray(predicted_returns)
        captured["basket_prediction_days"] = prediction_days
        return {"top_5": {"model": {}}}

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
    monkeypatch.setattr(
        trainer,
        "build_top_n_basket_backtest_report",
        fake_build_top_n_basket_backtest_report,
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
    assert captured["prediction_days"] == 10
    pd.testing.assert_frame_equal(
        captured["basket_split_metadata"],
        test_split_metadata,
    )
    np.testing.assert_array_equal(
        captured["basket_ranked_predictions"],
        regressor_predictions,
    )
    assert captured["basket_prediction_days"] == 10
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
    basket_report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.008,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
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
    monkeypatch.setattr(
        trainer,
        "build_top_n_basket_backtest_report",
        lambda *args, **kwargs: basket_report,
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
    assert any(
        message.startswith("XGBoost Top-N Basket Backtest Summary:")
        for message in info_messages
    )
    assert not any(
        message.startswith("XGBoost Top-N Ranked Selection Report:")
        for message in info_messages
    )
    assert not any(
        message.startswith("XGBoost Top-N Basket Backtest Report:")
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
    assert "XGBoost Top-N Basket Backtest Report:" in caplog.text


def test_log_xgboost_test_report_logs_basket_summary_at_info_and_full_report_at_debug(
    monkeypatch,
    caplog,
):
    captured = {}
    basket_report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.008,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
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
        lambda *args, **kwargs: {"top_5": {"model": {}}},
    )

    def fake_build_top_n_basket_backtest_report(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(ranked_predictions)
        captured["prediction_days"] = prediction_days
        return basket_report

    monkeypatch.setattr(
        trainer,
        "build_top_n_basket_backtest_report",
        fake_build_top_n_basket_backtest_report,
    )

    test_split_metadata = pd.DataFrame({"Ticker": ["AAA"]})
    regressor_predictions = np.array([0.03])

    with caplog.at_level(logging.INFO):
        trainer.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=test_split_metadata,
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=regressor_predictions,
        )

    pd.testing.assert_frame_equal(captured["split_metadata"], test_split_metadata)
    np.testing.assert_array_equal(captured["ranked_predictions"], regressor_predictions)
    assert captured["prediction_days"] == 10
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert any(
        message.startswith("XGBoost Top-N Basket Backtest Summary:")
        for message in info_messages
    )
    assert not any(
        message.startswith("XGBoost Top-N Basket Backtest Report:")
        for message in info_messages
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        trainer.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=test_split_metadata,
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=regressor_predictions,
        )

    assert "XGBoost Top-N Basket Backtest Report:" in caplog.text
