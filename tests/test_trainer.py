import logging
import pandas as pd
import numpy as np
import pytest

from src.config import (
    MODEL_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)
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

    result = trainer.prepare_data_parallel(
        ["AAPL", "MSFT"],
        period="1y",
        use_cache=False,
    )

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
        result = trainer.prepare_data_parallel(
            ["AAPL", "BAD", "MSFT"],
            period="1y",
            use_cache=False,
        )

    assert calls == [("AAPL", "1y"), ("BAD", "1y"), ("MSFT", "1y")]
    assert set(result["Ticker"]) == {"AAPL", "MSFT"}
    assert "Fetched valid data for 2 of 3 requested tickers." in caplog.text
    assert "Skipped 1 tickers with no usable data: ['BAD']" in caplog.text
    assert "No data returned for BAD. Skipping..." not in caplog.text


def test_empty_fetch_returns_without_normalizing_or_writing_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    empty_data = pd.DataFrame()
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return empty_data

    def fake_normalize_raw_ohlcv_index(data):
        raise AssertionError("empty data should not be normalized")

    def fake_write_yfinance_cache(data, ticker, period, cache_dir=None):
        raise AssertionError("empty data should not be cached")

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(
        trainer,
        "normalize_raw_ohlcv_index",
        fake_normalize_raw_ohlcv_index,
    )
    monkeypatch.setattr(trainer, "write_yfinance_cache", fake_write_yfinance_cache)

    result = trainer.fetch_raw_ticker_data("BAD", period="1y", use_cache=True)

    assert calls == [("BAD", "1y")]
    assert result.empty


def test_cache_hit_loads_cached_data_without_fetching_yfinance(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = trainer.get_yfinance_cache_path("AAPL", "1y")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_data = pd.DataFrame(
        {"Close": [100.0], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    cached_data.index.name = "Date"
    cached_data.to_csv(cache_path)

    def fake_fetch_stock_data(ticker, period="5y"):
        raise AssertionError("yfinance fetch should not be called on cache hit")

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)

    result = trainer.fetch_raw_ticker_data("AAPL", period="1y", use_cache=True)

    pd.testing.assert_frame_equal(result, cached_data)


def test_cache_hit_normalizes_timezone_aware_index(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = trainer.get_yfinance_cache_path("AAPL", "1y")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "Date,Close\n2024-03-11 00:00:00-04:00,100.0\n2024-03-08 00:00:00-05:00,99.0\n",
        encoding="utf-8",
    )

    def fake_fetch_stock_data(ticker, period="5y"):
        raise AssertionError("yfinance fetch should not be called on cache hit")

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)

    result = trainer.fetch_raw_ticker_data("AAPL", period="1y", use_cache=True)

    expected = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-03-08", "2024-03-11"]),
    )
    expected.index.name = "Date"
    pd.testing.assert_frame_equal(result, expected)
    assert result.index.tz is None


def test_cache_miss_fetches_data_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    fetched_data = pd.DataFrame(
        {"Close": [100.0], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    fetched_data.index.name = "Date"
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return fetched_data

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)

    result = trainer.fetch_raw_ticker_data("AAPL", period="1y", use_cache=True)

    assert calls == [("AAPL", "1y")]
    pd.testing.assert_frame_equal(result, fetched_data)
    cache_path = trainer.get_yfinance_cache_path("AAPL", "1y")
    assert cache_path.exists()
    cached_result = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(cached_result, fetched_data)


def test_cache_miss_normalizes_fetched_timezone_aware_index_before_caching(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    fetched_data = pd.DataFrame(
        {"Close": [100.0, 99.0]},
        index=pd.Index(
            [
                "2024-03-11 00:00:00-04:00",
                "2024-03-08 00:00:00-05:00",
            ],
            name="Date",
        ),
    )

    def fake_fetch_stock_data(ticker, period="5y"):
        return fetched_data

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)

    result = trainer.fetch_raw_ticker_data("AAPL", period="1y", use_cache=True)

    expected = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-03-08", "2024-03-11"]),
    )
    expected.index.name = "Date"
    pd.testing.assert_frame_equal(result, expected)
    assert result.index.tz is None

    cache_path = trainer.get_yfinance_cache_path("AAPL", "1y")
    cached_result = pd.read_csv(cache_path, index_col=0)
    cached_result = trainer.normalize_raw_ohlcv_index(cached_result)
    pd.testing.assert_frame_equal(cached_result, expected)


def test_normalize_raw_ohlcv_index_handles_mixed_dst_offsets():
    data = pd.DataFrame(
        {"Close": [100.0, 99.0]},
        index=[
            "2024-03-11 00:00:00-04:00",
            "2024-03-08 00:00:00-05:00",
        ],
    )

    result = trainer.normalize_raw_ohlcv_index(data)

    expected = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-03-08", "2024-03-11"]),
    )
    expected.index.name = "Date"
    pd.testing.assert_frame_equal(result, expected)
    assert result.index.tz is None


def test_no_cache_bypasses_cache_read_and_write(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = trainer.get_yfinance_cache_path("AAPL", "1y")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_data = pd.DataFrame(
        {"Close": [50.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    cached_data.index.name = "Date"
    cached_data.to_csv(cache_path)
    fetched_data = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    fetched_data.index.name = "Date"
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return fetched_data

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)

    result = trainer.fetch_raw_ticker_data("AAPL", period="1y", use_cache=False)

    assert calls == [("AAPL", "1y")]
    pd.testing.assert_frame_equal(result, fetched_data)
    unchanged_cache = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(unchanged_cache, cached_data)


def test_cache_read_failure_falls_back_to_fetch(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(trainer, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = trainer.get_yfinance_cache_path("AAPL", "1y")
    cache_path.write_text("not,a,valid,ohlcv\n1,2", encoding="utf-8")
    fetched_data = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.to_datetime(["2024-01-03"]),
    )
    fetched_data.index.name = "Date"

    def fake_read_csv(*args, **kwargs):
        raise ValueError("broken cache")

    def fake_fetch_stock_data(ticker, period="5y"):
        return fetched_data

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)

    with caplog.at_level(logging.WARNING):
        result = trainer.fetch_raw_ticker_data("AAPL", period="1y", use_cache=True)

    pd.testing.assert_frame_equal(result, fetched_data)
    assert "Failed to read YFinance cache for AAPL period=1y" in caplog.text


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
            "relative_momentum_baseline": {
                "available": True,
                "relative_momentum_score_column": "relative_momentum_10d",
                "average_selected_excess_return_vs_benchmark": 0.009,
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
        "momentum_10d excess=0.80%, relative_momentum_10d excess=0.90%, "
        "universe excess=0.15%\n"
        "  model minus random=0.83%, model minus momentum=0.43%, "
        "model minus relative momentum=0.33%, model beat rate=52.88%"
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
    assert "relative_momentum excess=n/a" in summary
    assert "model minus relative momentum=n/a" in summary
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
            "relative_momentum_baseline": {
                "available": True,
                "relative_momentum_score_column": "relative_momentum_10d",
                "average_basket_excess_return": 0.009,
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
        "momentum_10d excess=0.80%, relative_momentum_10d excess=0.90%, "
        "universe excess=0.15%, benchmark raw=0.23%\n"
        "  model minus random=0.60%, model minus momentum=0.20%, "
        "model minus relative momentum=0.10%, beat benchmark rate=52.88%"
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
    assert "relative_momentum excess=n/a" in summary
    assert "model minus relative momentum=n/a" in summary
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
                        "momentum_score_column": "momentum_5d",
                        "average_basket_excess_return": 0.003,
                    },
                    "relative_momentum_baseline": {
                        "available": True,
                        "relative_momentum_score_column": "relative_momentum_5d",
                        "average_basket_excess_return": 0.004,
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
                        "momentum_score_column": "momentum_10d",
                        "average_basket_excess_return": np.nan,
                    },
                    "relative_momentum_baseline": {
                        "available": False,
                        "relative_momentum_score_column": "relative_momentum_10d",
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
        "  top_5 model excess=1.00%, random=0.20%, momentum_5d=0.30%, "
        "relative_momentum_5d=0.40%, "
        "universe=0.10%, benchmark raw=0.40%\n"
        "10d:\n"
        "  top_5 model excess=-1.00%, random=0.10%, momentum_10d=unavailable, "
        "relative_momentum_10d=unavailable, "
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


def test_train_models_metadata_includes_momentum_features_and_excludes_targets(
    monkeypatch,
):
    captured = {}
    target_columns = {
        "raw_forward_return",
        "benchmark_forward_return",
        "excess_forward_return",
        "targetReturns",
        "beat_benchmark_target",
    }

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)
            self.scalar = None

        def prepare_for_train(self, data, prediction_days, test_size):
            rows = 4
            feature_count = len(MODEL_FEATURE_COLUMNS)
            split_metadata = pd.DataFrame(
                {
                    "Ticker": ["AAA"] * rows,
                    "prediction_date": pd.date_range("2024-01-01", periods=rows),
                    "dailyReturn": [0.01] * rows,
                    "raw_forward_return": [0.02] * rows,
                    "benchmark_forward_return": [0.01] * rows,
                    "excess_forward_return": [0.01] * rows,
                    "beat_benchmark_target": [1] * rows,
                }
            )
            return {
                "x_train": np.ones((rows, feature_count)),
                "x_val": np.ones((rows, feature_count)),
                "x_test": np.ones((rows, feature_count)),
                "y_train": np.array([0.01, 0.02, -0.01, 0.03]),
                "y_val": np.array([0.01, -0.02, 0.02, 0.03]),
                "y_test": np.array([0.02, -0.01, 0.01, 0.03]),
                "direction_y_train": np.array([1, 1, 0, 1]),
                "direction_y_val": np.array([1, 0, 1, 1]),
                "direction_y_test": np.array([1, 0, 1, 1]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {"test": split_metadata},
            }

    class FakeLinearRegression:
        def fit(self, x, y):
            self.coef_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.zeros(len(x))

    class FakeXgbModel:
        def __init__(self, **kwargs):
            self.feature_importances_ = np.array([])

        def fit(self, x, y, eval_set=None, verbose=False):
            self.feature_importances_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.ones(len(x), dtype=int)

        def predict_proba(self, x):
            return np.column_stack([np.zeros(len(x)), np.ones(len(x))])

        def save_model(self, path):
            pass

    def fake_save_horizon_model_artifacts(
        prediction_days,
        linear_model,
        scaler_lr,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
        regressor,
    ):
        captured["all_features"] = all_features
        captured["model_metadata"] = model_metadata
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(trainer, "LinearRegression", FakeLinearRegression)
    monkeypatch.setattr(trainer, "XGBClassifier", FakeXgbModel)
    monkeypatch.setattr(trainer, "XGBRegressor", FakeXgbModel)
    monkeypatch.setattr(trainer, "evaluate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer,
        "log_xgboost_test_report",
        lambda *args, **kwargs: {"basket_backtest": {}},
    )
    monkeypatch.setattr(
        trainer, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        trainer,
        "save_horizon_model_artifacts",
        fake_save_horizon_model_artifacts,
    )

    trainer.train_models(pd.DataFrame({"Close": [1.0]}), prediction_days=10)

    metadata = captured["model_metadata"]
    for column in MOMENTUM_FEATURE_COLUMNS + RELATIVE_MOMENTUM_FEATURE_COLUMNS:
        assert column in captured["all_features"]
        assert column in metadata["classifier_features"]
        assert column in metadata["regressor_features"]

    assert metadata["classifier_features"] == MODEL_FEATURE_COLUMNS
    assert metadata["regressor_features"] == MODEL_FEATURE_COLUMNS
    assert target_columns.isdisjoint(captured["all_features"])
    assert target_columns.isdisjoint(metadata["classifier_features"])
    assert target_columns.isdisjoint(metadata["regressor_features"])


def test_parse_args_defaults_to_ten_prediction_days():
    args = trainer.parse_args([])

    assert args.prediction_days == 10
    assert args.horizons == [10]
    assert args.all_horizons is False
    assert args.period == "5y"
    assert args.no_cache is False


def test_parse_args_accepts_period():
    args = trainer.parse_args(["--period", "10y"])

    assert args.period == "10y"


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


def test_save_horizon_model_artifacts_writes_horizon_specific_paths(
    tmp_path, monkeypatch
):
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


def test_main_trains_all_horizons_and_logs_comparison(monkeypatch, caplog, capsys):
    trained_horizons = []
    prepare_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)

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
                        "momentum_score_column": f"momentum_{prediction_days}d",
                        "average_basket_excess_return": 0.002,
                    },
                    "relative_momentum_baseline": {
                        "available": True,
                        "relative_momentum_score_column": (
                            f"relative_momentum_{prediction_days}d"
                        ),
                        "average_basket_excess_return": 0.003,
                    },
                    "universe": {"average_basket_excess_return": 0.0},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    with caplog.at_level(logging.INFO):
        trainer.main(["--all-horizons", "--period", "10y"])

    assert prepare_calls == [(trainer.TRAINING_TICKERS, "10y", True)]
    assert trained_horizons == [5, 10, 20, 50]
    assert "Selected YFinance period: 10y" in caplog.text
    assert "Raw YFinance OHLCV cache enabled: True" in caplog.text
    assert "Horizon Comparison Summary:" in caplog.text
    assert "50d:" in caplog.text
    output = capsys.readouterr().out
    assert "Horizon Comparison Summary:" in output
    assert "50d:" in output
    assert "XGBoost Beat-Benchmark Classification Report" not in output


def test_main_single_horizon_prints_top_n_basket_summary(monkeypatch, capsys):
    trained_horizons = []
    prepare_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)

    def fake_train_models(data, prediction_days):
        trained_horizons.append(prediction_days)
        return {
            "prediction_days": prediction_days,
            "basket_backtest": {
                "top_5": {
                    "model": {
                        "average_basket_raw_return": 0.02,
                        "average_basket_excess_return": 0.01,
                        "beat_benchmark_rate": 0.55,
                    },
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": True,
                        "momentum_score_column": "momentum_10d",
                        "average_basket_excess_return": 0.004,
                    },
                    "relative_momentum_baseline": {
                        "available": True,
                        "relative_momentum_score_column": "relative_momentum_10d",
                        "average_basket_excess_return": 0.005,
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    trainer.main(["--prediction-days", "10", "--no-cache"])

    assert prepare_calls == [(trainer.TRAINING_TICKERS, "5y", False)]
    assert trained_horizons == [10]
    output = capsys.readouterr().out
    assert "XGBoost Top-N Basket Backtest Summary:" in output
    assert "top_5:" in output
    assert "Horizon Comparison Summary:" not in output
    assert "XGBoost Beat-Benchmark Classification Report" not in output


def test_log_xgboost_test_report_uses_explicit_direction_labels(monkeypatch):
    captured = {}

    def fake_build_classification_report(y_true, y_pred):
        captured["y_true"] = np.asarray(y_true)
        captured["y_pred"] = np.asarray(y_pred)
        return {"accuracy": 1.0}

    def fake_build_top_n_selection_reports(
        split_metadata,
        predicted_returns,
        prediction_days=None,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(predicted_returns)
        captured["prediction_days"] = prediction_days
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 1}},
            "basket_backtest": {"top_5": {"model": {}}},
        }

    monkeypatch.setattr(
        trainer,
        "build_classification_report",
        fake_build_classification_report,
    )
    monkeypatch.setattr(
        trainer,
        "build_top_n_selection_reports",
        fake_build_top_n_selection_reports,
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
        "build_top_n_selection_reports",
        lambda *args, **kwargs: {
            "ranked_selection": top_n_report,
            "basket_backtest": basket_report,
        },
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
    def fake_build_top_n_selection_reports(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(ranked_predictions)
        captured["prediction_days"] = prediction_days
        return {
            "ranked_selection": {"top_5": {"model": {}}},
            "basket_backtest": basket_report,
        }

    monkeypatch.setattr(
        trainer,
        "build_top_n_selection_reports",
        fake_build_top_n_selection_reports,
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
