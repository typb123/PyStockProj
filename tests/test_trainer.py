"""Trainer tests for caching, metadata, CLI validation, and report plumbing."""

import logging
import pandas as pd
import numpy as np
import pytest

from src.config import (
    DEFAULT_TRAINING_UNIVERSE,
    LARGE_MEGA_CAP_STOCKS,
    TRAINING_TICKERS,
    TRAINING_UNIVERSES,
    XG_PARAMS_REGRESSOR,
    get_training_tickers,
)
from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
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


def test_prepare_data_parallel_fetches_spy_when_in_universe(monkeypatch):
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2024-01-02"]),
        )

    monkeypatch.setattr(trainer, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(trainer, "calculate_data", lambda df: df)

    result = trainer.prepare_data_parallel(
        ["SPY", "AAPL"],
        period="1y",
        use_cache=False,
    )

    assert calls == [("SPY", "1y"), ("AAPL", "1y")]
    assert set(result["Ticker"]) == {"SPY", "AAPL"}


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


def test_format_top_n_ranked_selection_summary_accepts_custom_title():
    report = {
        "top_5": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.002,
            },
            "momentum_baseline": {
                "available": True,
                "average_selected_excess_return_vs_benchmark": 0.004,
            },
            "universe": {
                "average_excess_return_vs_benchmark": 0.001,
            },
        }
    }

    summary = trainer.format_top_n_ranked_selection_summary(
        report,
        title="Custom Ranked Title:",
    )

    assert summary.startswith("Custom Ranked Title:\ntop_5:")
    assert "XGBoost Top-N Ranked Selection Summary:" not in summary


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


def test_format_top_n_basket_backtest_summary_formats_bootstrap_ci_line():
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
                "average_basket_excess_return": 0.008,
            },
            "relative_momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.009,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
            },
            "bootstrap_confidence_intervals": {
                "model_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": 0.0052,
                    "ci_upper": 0.0178,
                    "confidence_level": 0.95,
                },
                "model_minus_momentum_baseline_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": -0.0008,
                    "ci_upper": 0.0083,
                    "confidence_level": 0.95,
                },
                "model_minus_relative_momentum_baseline_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": -0.001,
                    "ci_upper": 0.006,
                    "confidence_level": 0.95,
                },
                "model_minus_universe_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": 0.001,
                    "ci_upper": 0.012,
                    "confidence_level": 0.95,
                },
            },
        }
    }

    summary = trainer.format_top_n_basket_backtest_summary(report)

    assert "model excess 95% CI=[0.52%, 1.78%]" in summary
    assert "model minus momentum 95% CI=[-0.08%, 0.83%]" in summary
    assert "model minus relative momentum 95% CI=[-0.10%, 0.60%]" in summary
    assert "model minus universe 95% CI=[0.10%, 1.20%]" in summary


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


def test_format_top_n_basket_backtest_summary_accepts_custom_title():
    report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.012,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.002,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.004,
            },
            "universe": {
                "average_basket_excess_return": 0.001,
            },
            "benchmark": {
                "average_basket_raw_return": 0.003,
            },
        }
    }

    summary = trainer.format_top_n_basket_backtest_summary(
        report,
        title="Custom Basket Title:",
    )

    assert summary.startswith("Custom Basket Title:\ntop_5:")
    assert "XGBoost Top-N Basket Backtest Summary:" not in summary


def test_format_top_n_basket_backtest_by_year_summary_formats_year_rows():
    report = {
        "top_5": {
            "2022": {
                "model": {
                    "average_basket_excess_return": 0.015,
                    "beat_benchmark_rate": 0.75,
                    "evaluated_dates": 4,
                },
                "random_baseline": {
                    "average_basket_excess_return": 0.004,
                },
                "momentum_baseline": {
                    "available": True,
                    "momentum_score_column": "momentum_10d",
                    "average_basket_excess_return": 0.010,
                },
                "relative_momentum_baseline": {
                    "available": True,
                    "relative_momentum_score_column": "relative_momentum_10d",
                    "average_basket_excess_return": 0.011,
                },
                "universe": {
                    "average_basket_excess_return": 0.002,
                },
            },
            "2023": {
                "model": {
                    "average_basket_excess_return": -0.001,
                    "beat_benchmark_rate": 0.40,
                    "evaluated_dates": 5,
                },
                "random_baseline": {
                    "average_basket_excess_return": 0.003,
                },
                "momentum_baseline": {
                    "available": False,
                    "average_basket_excess_return": np.nan,
                },
                "relative_momentum_baseline": {
                    "available": False,
                    "average_basket_excess_return": np.nan,
                },
                "universe": {
                    "average_basket_excess_return": 0.001,
                },
            },
        }
    }

    summary = trainer.format_top_n_basket_backtest_by_year_summary(report)

    assert summary == (
        "XGBoost Top-N Basket Backtest By-Year Summary:\n"
        "2022 top_5: model excess=1.50%, random excess=0.40%, "
        "momentum_10d excess=1.00%, relative_momentum_10d excess=1.10%, "
        "universe excess=0.20%, model minus random=1.10%, "
        "model minus momentum=0.50%, model minus relative momentum=0.40%, beat benchmark rate=75.00%, "
        "evaluated dates=4\n"
        "2023 top_5: model excess=-0.10%, random excess=0.30%, "
        "momentum excess=unavailable, relative_momentum excess=unavailable, "
        "universe excess=0.10%, model minus random=-0.40%, "
        "model minus momentum=unavailable, model minus relative momentum=unavailable, beat benchmark rate=40.00%, "
        "evaluated dates=5"
    )


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
    assert metadata["target_type"] == "spy_relative_excess_forward_return"
    assert metadata["benchmark_ticker"] == "SPY"
    assert metadata["regressor_target"] == "targetReturns"
    assert metadata["classifier_target"] == "beat_benchmark_target"


def test_default_training_universe_is_large_mega_cap_stocks():
    assert DEFAULT_TRAINING_UNIVERSE == "large_mega_cap_stocks"


def test_training_tickers_match_default_large_mega_cap_universe():
    assert TRAINING_TICKERS == LARGE_MEGA_CAP_STOCKS
    assert TRAINING_TICKERS == get_training_tickers()


def test_named_training_universes_have_no_duplicates():
    for universe_name, tickers in TRAINING_UNIVERSES.items():
        assert len(tickers) == len(set(tickers)), universe_name


def test_named_training_universes_include_spy_benchmark():
    for universe_name, tickers in TRAINING_UNIVERSES.items():
        assert "SPY" in tickers, universe_name


def test_get_training_tickers_returns_copy():
    tickers = get_training_tickers("large_mega_cap_stocks")
    tickers.append("SHOULD_NOT_MUTATE")

    assert "SHOULD_NOT_MUTATE" not in get_training_tickers("large_mega_cap_stocks")
    assert "SHOULD_NOT_MUTATE" not in TRAINING_UNIVERSES["large_mega_cap_stocks"]


def test_get_training_tickers_rejects_unknown_universe_with_valid_choices():
    with pytest.raises(ValueError) as exc_info:
        get_training_tickers("bad_name")

    message = str(exc_info.value)
    assert "bad_name" in message
    for universe_name in TRAINING_UNIVERSES:
        assert universe_name in message


def test_xgboost_regressor_candidates_include_default_baseline():
    candidates = trainer.build_xgboost_regressor_candidate_configs()

    assert candidates[0]["candidate_id"] == 0
    assert candidates[0]["candidate_name"] == "candidate_0_baseline"
    assert candidates[0]["params"] == XG_PARAMS_REGRESSOR
    assert len(candidates) >= 2


def test_validation_top_n_selection_chooses_best_candidate_and_preserves_splits(
    monkeypatch,
):
    captured_fit_calls = []

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.array([1.0])

        def fit(self, x, y, eval_set=None, verbose=False):
            captured_fit_calls.append(
                {
                    "x": x.copy(),
                    "y": np.asarray(y).copy(),
                    "eval_x": eval_set[0][0].copy(),
                    "eval_y": np.asarray(eval_set[0][1]).copy(),
                    "verbose": verbose,
                }
            )
            return self

        def predict(self, x):
            return np.full(len(x), self.params["validation_score"])

    def fake_model_only_report(split_metadata, ranked_predictions, top_n_values):
        score = float(np.asarray(ranked_predictions)[0])
        return {
            f"top_{top_n}": {
                "model": {"average_basket_excess_return": score + top_n / 10000}
            }
            for top_n in top_n_values
        }

    monkeypatch.setattr(trainer, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        trainer,
        "build_model_only_top_n_basket_backtest_report",
        fake_model_only_report,
    )

    x_train = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    x_val = pd.DataFrame({"feature_a": [5.0, 6.0], "feature_b": [7.0, 8.0]})
    y_train = np.array([0.01, -0.02])
    y_val = np.array([0.03, 0.04])
    validation_metadata = pd.DataFrame({"Ticker": ["AAA", "BBB"]})
    candidates = [
        {
            "candidate_id": 0,
            "candidate_name": "candidate_0_baseline",
            "params": {"validation_score": 0.01},
        },
        {
            "candidate_id": 1,
            "candidate_name": "candidate_1_better",
            "params": {"validation_score": 0.03},
        },
    ]

    selected_model, report = trainer.select_xgboost_regressor_by_validation_top_n(
        x_train,
        y_train,
        x_val,
        y_val,
        validation_metadata,
        candidate_configs=candidates,
    )

    assert selected_model.params == {"validation_score": 0.03}
    assert report["selection_metric"] == "validation_top_n_mean_excess_return"
    assert report["selected_candidate_id"] == 1
    assert report["candidates"][0]["selected"] is False
    assert report["candidates"][1]["selected"] is True
    assert report["candidates"][1]["available_bucket_count"] == 3
    assert "unavailable or NaN buckets are ignored" in report["selection_bucket_policy"]
    assert len(captured_fit_calls) == 2
    for fit_call in captured_fit_calls:
        pd.testing.assert_frame_equal(fit_call["x"], x_train)
        pd.testing.assert_frame_equal(fit_call["eval_x"], x_val)
        np.testing.assert_array_equal(fit_call["y"], y_train)
        np.testing.assert_array_equal(fit_call["eval_y"], y_val)
        assert fit_call["verbose"] is False


def test_format_xgboost_regressor_validation_selection_report_is_readable():
    report = {
        "candidates": [
            {
                "candidate_name": "candidate_0_baseline",
                "params": {"max_depth": 5},
                "validation_top_n_mean_excess_return": 0.01,
                "validation_top_n_basket_excess_returns": {
                    "top_5": 0.02,
                    "top_10": 0.01,
                    "top_20": 0.00,
                },
                "selected": False,
            },
            {
                "candidate_name": "candidate_1_better",
                "params": {"max_depth": 4},
                "validation_top_n_mean_excess_return": 0.03,
                "validation_top_n_basket_excess_returns": {
                    "top_5": 0.04,
                    "top_10": 0.03,
                    "top_20": 0.02,
                },
                "selected": True,
            },
        ],
        "selected_candidate_name": "candidate_1_better",
        "selected_validation_top_n_mean_excess_return": 0.03,
    }

    summary = trainer.format_xgboost_regressor_validation_selection_report(report)

    assert summary.startswith("XGBoost Regressor Validation Top-N Selection Report:")
    assert "candidate_0_baseline: score=1.00%" in summary
    assert "candidate_1_better selected: score=3.00%" in summary
    assert "top_5=4.00%, top_10=3.00%, top_20=2.00%" in summary
    assert "selected=candidate_1_better score=3.00%" in summary


def test_train_models_uses_selected_regressor_predictions_for_final_report(
    monkeypatch,
):
    rows = 4
    feature_count = len(MODEL_FEATURE_COLUMNS)
    y_train = np.array([0.01, 0.02, -0.01, 0.03])
    y_val = np.array([0.01, -0.02, 0.02, 0.03])
    captured = {}

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)
            self.scalar = None

        def prepare_for_train(self, data, prediction_days, test_size):
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
                "x_val": np.ones((rows, feature_count)) * 2,
                "x_test": np.ones((rows, feature_count)) * 3,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": np.array([0.02, -0.01, 0.01, 0.03]),
                "direction_y_train": np.array([1, 1, 0, 1]),
                "direction_y_val": np.array([1, 0, 1, 1]),
                "direction_y_test": np.array([1, 0, 1, 1]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "val": split_metadata.copy(),
                    "test": split_metadata.copy(),
                },
            }

    class FakeLinearRegression:
        def fit(self, x, y):
            self.coef_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.zeros(len(x))

    class FakeClassifier:
        def __init__(self, **params):
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

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.array([])

        def fit(self, x, y, eval_set=None, verbose=False):
            captured.setdefault("regressor_fit_y", []).append(np.asarray(y).copy())
            self.feature_importances_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.full(len(x), self.params["prediction"])

        def save_model(self, path):
            pass

    candidates = [
        {
            "candidate_id": 0,
            "candidate_name": "candidate_0_baseline",
            "params": {"prediction": 0.01},
        },
        {
            "candidate_id": 1,
            "candidate_name": "candidate_1_selected",
            "params": {"prediction": 0.04},
        },
    ]

    def fake_model_only_report(split_metadata, ranked_predictions, top_n_values):
        score = float(np.asarray(ranked_predictions)[0])
        return {
            f"top_{top_n}": {"model": {"average_basket_excess_return": score}}
            for top_n in top_n_values
        }

    def fake_log_xgboost_test_report(*args, **kwargs):
        captured["final_regressor_predictions"] = np.asarray(args[8])
        captured["selection_report"] = kwargs["regressor_validation_selection_report"]
        return {
            "regressor_validation_selection": captured["selection_report"],
            "basket_backtest": {},
        }

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(trainer, "LinearRegression", FakeLinearRegression)
    monkeypatch.setattr(trainer, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(trainer, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        trainer,
        "build_xgboost_regressor_candidate_configs",
        lambda base_params: candidates,
    )
    monkeypatch.setattr(
        trainer,
        "build_model_only_top_n_basket_backtest_report",
        fake_model_only_report,
    )
    monkeypatch.setattr(trainer, "evaluate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer, "log_xgboost_test_report", fake_log_xgboost_test_report
    )
    monkeypatch.setattr(
        trainer, "log_feature_importances", lambda *args, **kwargs: None
    )

    def fake_save_horizon_model_artifacts(*args, **kwargs):
        captured["saved_model_metadata"] = args[5]
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(
        trainer,
        "save_horizon_model_artifacts",
        fake_save_horizon_model_artifacts,
    )

    report = trainer.train_models(pd.DataFrame({"Close": [1.0]}), prediction_days=10)

    np.testing.assert_array_equal(captured["regressor_fit_y"][0], y_train)
    np.testing.assert_array_equal(
        captured["final_regressor_predictions"], [0.04] * rows
    )
    assert captured["selection_report"]["selected_candidate_id"] == 1
    assert report["regressor_validation_selection"]["selected_candidate_id"] == 1
    assert (
        captured["saved_model_metadata"]["regressor_validation_selection"][
            "selected_candidate_id"
        ]
        == 1
    )
    assert (
        captured["saved_model_metadata"]["xgboost_regressor_selected_candidate_name"]
        == "candidate_1_selected"
    )
    assert captured["saved_model_metadata"]["xgboost_regressor_selected_params"] == {
        "prediction": 0.04
    }


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
                "split_metadata": {
                    "val": split_metadata.copy(),
                    "test": split_metadata,
                },
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
    for column in (
        ABSOLUTE_MOMENTUM_FEATURE_COLUMNS + SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS
    ):
        assert column in captured["all_features"]
        assert column in metadata["classifier_features"]
        assert column in metadata["regressor_features"]

    assert metadata["classifier_features"] == MODEL_FEATURE_COLUMNS
    assert metadata["regressor_features"] == MODEL_FEATURE_COLUMNS
    assert metadata["target_type"] == "spy_relative_excess_forward_return"
    assert metadata["benchmark_ticker"] == "SPY"
    assert metadata["regressor_target"] == "targetReturns"
    assert metadata["classifier_target"] == "beat_benchmark_target"
    assert target_columns.isdisjoint(captured["all_features"])
    assert target_columns.isdisjoint(metadata["classifier_features"])
    assert target_columns.isdisjoint(metadata["regressor_features"])


def test_run_walk_forward_models_records_selected_candidate_and_test_metrics(
    monkeypatch,
):
    captured = {}
    prepared_frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2020-01-01"])})
    fold = {
        "fold_index": 0,
        "train_years": [2015, 2016, 2017, 2018, 2019],
        "validation_years": [2020],
        "test_years": [2021],
        "train_date_range": {"start": "2015-01-01", "end": "2019-12-31"},
        "validation_date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "test_date_range": {"start": "2021-01-01", "end": "2021-12-31"},
    }

    class FakeRegressor:
        def predict(self, x):
            captured["predict_x"] = x.copy()
            return np.array([0.20, 0.10])

    def fake_select(
        x_train,
        y_train,
        x_val,
        y_val,
        validation_split_metadata,
        candidate_configs=None,
    ):
        captured["candidate_configs"] = candidate_configs
        pd.testing.assert_frame_equal(
            validation_split_metadata,
            pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
        )
        return FakeRegressor(), {
            "selected_candidate_id": 1,
            "selected_candidate_name": "candidate_1",
            "selected_validation_top_n_mean_excess_return": 0.03,
        }

    def fake_top_n_reports(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["top_n_predictions"] = np.asarray(ranked_predictions)
        captured["top_n_prediction_days"] = prediction_days
        captured["top_n_random_trials"] = random_trials
        captured["top_n_random_trial_workers"] = random_trial_workers
        return {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.04},
                    "momentum_baseline": {"average_basket_excess_return": 0.01},
                    "universe": {"average_basket_excess_return": 0.02},
                }
            }
        }

    monkeypatch.setattr(
        trainer,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (prepared_frame, ["feature_a"]),
    )
    monkeypatch.setattr(
        trainer,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        trainer,
        "build_walk_forward_split",
        lambda *args, **kwargs: {
            "x_train": np.array([[1.0], [2.0]]),
            "x_val": np.array([[3.0], [4.0]]),
            "x_test": np.array([[5.0], [6.0]]),
            "y_train": np.array([0.01, 0.02]),
            "y_val": np.array([0.03, 0.04]),
            "split_metadata": {
                "val": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
                "test": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
            },
            "split_date_ranges": {
                "train": {"start": "2015-01-01", "end": "2019-12-15"},
                "validation": {"start": "2020-01-01", "end": "2020-12-15"},
                "test": {"start": "2021-01-01", "end": "2021-12-31"},
            },
        },
    )
    monkeypatch.setattr(
        trainer,
        "build_xgboost_regressor_candidate_configs",
        lambda params: [
            {"candidate_id": 0, "candidate_name": "candidate_0", "params": {}}
        ],
    )
    monkeypatch.setattr(
        trainer,
        "select_xgboost_regressor_by_validation_top_n",
        fake_select,
    )
    monkeypatch.setattr(trainer, "build_top_n_selection_reports", fake_top_n_reports)

    report = trainer.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        prediction_days=10,
        random_trials=7,
        random_trial_workers=2,
    )

    assert report["prediction_days"] == 10
    assert report["folds"][0]["selected_candidate_name"] == "candidate_1"
    assert report["folds"][0]["validation_selection_score"] == 0.03
    assert report["folds"][0]["train_date_range"] == {
        "start": "2015-01-01",
        "end": "2019-12-15",
    }
    assert report["folds"][0]["validation_date_range"] == {
        "start": "2020-01-01",
        "end": "2020-12-15",
    }
    assert report["folds"][0]["test_date_range"] == {
        "start": "2021-01-01",
        "end": "2021-12-31",
    }
    assert np.isclose(report["folds"][0]["top_n"]["top_5"]["model_excess"], 0.04)
    assert np.isclose(
        report["folds"][0]["top_n"]["top_5"]["model_minus_momentum"],
        0.03,
    )
    np.testing.assert_array_equal(captured["top_n_predictions"], [0.20, 0.10])
    assert captured["top_n_prediction_days"] == 10
    assert captured["top_n_random_trials"] == 7
    assert captured["top_n_random_trial_workers"] == 2
    assert report["aggregate"]["selected_candidate_counts"] == {"candidate_1": 1}


def test_parse_args_defaults_to_ten_prediction_days():
    args = trainer.parse_args([])

    assert args.prediction_days == 10
    assert args.horizons == [10]
    assert args.all_horizons is False
    assert args.period == "5y"
    assert args.universe == DEFAULT_TRAINING_UNIVERSE
    assert args.no_cache is False
    assert args.random_trials == 100
    assert args.random_trial_workers == 8
    assert args.walk_forward is False


def test_parse_args_accepts_period():
    args = trainer.parse_args(["--period", "10y"])

    assert args.period == "10y"


def test_parse_args_accepts_training_universe():
    args = trainer.parse_args(["--universe", "broad_sector_etfs"])

    assert args.universe == "broad_sector_etfs"


def test_parse_args_accepts_random_trials():
    args = trainer.parse_args(["--random-trials", "20"])

    assert args.random_trials == 20


def test_parse_args_accepts_random_trial_workers():
    args = trainer.parse_args(["--random-trial-workers", "2"])

    assert args.random_trial_workers == 2


def test_parse_args_accepts_walk_forward_options():
    args = trainer.parse_args(
        [
            "--walk-forward",
            "--prediction-days",
            "20",
            "--walk-forward-min-train-years",
            "4",
            "--walk-forward-validation-years",
            "2",
            "--walk-forward-test-years",
            "1",
        ]
    )

    assert args.walk_forward is True
    assert args.horizons == [20]
    assert args.walk_forward_min_train_years == 4
    assert args.walk_forward_validation_years == 2
    assert args.walk_forward_test_years == 1


def test_parse_args_rejects_invalid_random_trials_values():
    for argv in [
        ["--random-trials", "0"],
        ["--random-trials", "-1"],
    ]:
        with pytest.raises(SystemExit):
            trainer.parse_args(argv)


def test_parse_args_rejects_removed_parallel_random_trials_flag():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--parallel-random-trials"])


def test_parse_args_rejects_invalid_random_trial_workers_values():
    for argv in [
        ["--random-trial-workers", "0"],
        ["--random-trial-workers", "-1"],
    ]:
        with pytest.raises(SystemExit):
            trainer.parse_args(argv)


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


def test_parse_args_rejects_walk_forward_with_all_horizons():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--walk-forward", "--all-horizons"])


def test_parse_args_rejects_unknown_training_universe():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--universe", "bad_name"])


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
    trained_random_trials = []
    trained_random_trial_workers = []
    prepare_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
    ):
        trained_horizons.append(prediction_days)
        trained_random_trials.append(random_trials)
        trained_random_trial_workers.append(random_trial_workers)
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
        trainer.main(
            [
                "--all-horizons",
                "--period",
                "10y",
                "--random-trials",
                "20",
                "--random-trial-workers",
                "2",
            ]
        )

    assert prepare_calls == [
        (get_training_tickers(DEFAULT_TRAINING_UNIVERSE), "10y", True)
    ]
    assert trained_horizons == [5, 10, 20, 50]
    assert trained_random_trials == [20, 20, 20, 20]
    assert trained_random_trial_workers == [2, 2, 2, 2]
    assert "Selected YFinance period: 10y" in caplog.text
    assert "Raw YFinance OHLCV cache enabled: True" in caplog.text
    assert "Horizon Comparison Summary:" in caplog.text
    assert "50d:" in caplog.text
    output = capsys.readouterr().out
    assert "Horizon Comparison Summary:" in output
    assert "50d:" in output
    assert "XGBoost Beat-Benchmark Classification Report" not in output


def test_main_passes_selected_broad_sector_etf_universe_to_prepare_data(
    monkeypatch,
):
    prepare_calls = []
    trained_horizons = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
    ):
        trained_horizons.append(prediction_days)
        return {
            "prediction_days": prediction_days,
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.01},
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": False,
                        "momentum_score_column": "momentum_10d",
                    },
                    "relative_momentum_baseline": {
                        "available": False,
                        "relative_momentum_score_column": "relative_momentum_10d",
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)
    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    trainer.main(
        [
            "--prediction-days",
            "10",
            "--period",
            "10y",
            "--universe",
            "broad_sector_etfs",
        ]
    )

    assert prepare_calls == [(get_training_tickers("broad_sector_etfs"), "10y", True)]
    assert trained_horizons == [10]


def test_main_single_horizon_prints_top_n_basket_summary(monkeypatch, capsys):
    trained_horizons = []
    trained_random_trials = []
    trained_random_trial_workers = []
    prepare_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
    ):
        trained_horizons.append(prediction_days)
        trained_random_trials.append(random_trials)
        trained_random_trial_workers.append(random_trial_workers)
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

    trainer.main(
        [
            "--prediction-days",
            "10",
            "--no-cache",
            "--random-trials",
            "20",
            "--random-trial-workers",
            "2",
        ]
    )

    assert prepare_calls == [
        (get_training_tickers(DEFAULT_TRAINING_UNIVERSE), "5y", False)
    ]
    assert trained_horizons == [10]
    assert trained_random_trials == [20]
    assert trained_random_trial_workers == [2]
    output = capsys.readouterr().out
    assert "XGBoost Top-N Basket Backtest Summary:" in output
    assert "top_5:" in output
    assert "Horizon Comparison Summary:" not in output
    assert "XGBoost Beat-Benchmark Classification Report" not in output


def test_main_walk_forward_runs_walk_forward_path(monkeypatch, capsys):
    prepare_calls = []
    walk_forward_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    def fake_run_walk_forward_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        min_train_years=5,
        validation_years=1,
        test_years=1,
    ):
        walk_forward_calls.append(
            {
                "prediction_days": prediction_days,
                "random_trials": random_trials,
                "random_trial_workers": random_trial_workers,
                "min_train_years": min_train_years,
                "validation_years": validation_years,
                "test_years": test_years,
            }
        )
        return {
            "prediction_days": prediction_days,
            "folds": [
                {
                    "fold_index": 0,
                    "train_date_range": {
                        "start": "2015-01-01",
                        "end": "2019-12-31",
                    },
                    "validation_date_range": {
                        "start": "2020-01-01",
                        "end": "2020-12-31",
                    },
                    "test_date_range": {
                        "start": "2021-01-01",
                        "end": "2021-12-31",
                    },
                    "selected_candidate_name": "candidate_0",
                    "validation_selection_score": 0.01,
                    "top_n": {
                        "top_5": {
                            "model_excess": 0.02,
                            "model_minus_momentum": 0.01,
                            "model_minus_universe": 0.03,
                        }
                    },
                }
            ],
            "aggregate": {
                "fold_count": 1,
                "selected_candidate_counts": {"candidate_0": 1},
                "top_n": {
                    "top_5": {
                        "average_model_excess": 0.02,
                        "average_model_minus_momentum": 0.01,
                        "average_model_minus_universe": 0.03,
                        "fold_win_rate_vs_momentum": 1.0,
                        "fold_win_rate_vs_universe": 1.0,
                    }
                },
            },
        }

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)
    monkeypatch.setattr(
        trainer,
        "run_walk_forward_models",
        fake_run_walk_forward_models,
    )
    monkeypatch.setattr(
        trainer,
        "train_models",
        lambda *args, **kwargs: pytest.fail("normal training path should not run"),
    )

    report = trainer.main(
        [
            "--walk-forward",
            "--prediction-days",
            "10",
            "--period",
            "10y",
            "--random-trials",
            "20",
            "--random-trial-workers",
            "2",
            "--walk-forward-min-train-years",
            "4",
        ]
    )

    assert prepare_calls == [
        (get_training_tickers(DEFAULT_TRAINING_UNIVERSE), "10y", True)
    ]
    assert walk_forward_calls == [
        {
            "prediction_days": 10,
            "random_trials": 20,
            "random_trial_workers": 2,
            "min_train_years": 4,
            "validation_years": 1,
            "test_years": 1,
        }
    ]
    assert report["aggregate"]["fold_count"] == 1
    output = capsys.readouterr().out
    assert "Walk-Forward Top-N Summary:" in output
    assert "top_5:" in output
    assert "XGBoost Top-N Basket Backtest Summary:" not in output


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
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(predicted_returns)
        captured["prediction_days"] = prediction_days
        captured["random_trials"] = random_trials
        captured["random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 1}},
            "basket_backtest": {"top_5": {"model": {}}},
            "basket_backtest_by_year": {"top_5": {"2024": {"model": {}}}},
        }

    def fake_build_probability_ranked_top_n_selection_reports(
        split_metadata,
        classifier_probabilities,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["classifier_split_metadata"] = split_metadata
        captured["classifier_probabilities"] = np.asarray(classifier_probabilities)
        captured["classifier_prediction_days"] = prediction_days
        captured["classifier_random_trials"] = random_trials
        captured["classifier_random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 1}},
            "basket_backtest": {"top_5": {"model": {}}},
            "basket_backtest_by_year": {"top_5": {"2024": {"classifier_model": {}}}},
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
    monkeypatch.setattr(
        trainer,
        "build_probability_ranked_top_n_selection_reports",
        fake_build_probability_ranked_top_n_selection_reports,
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

    report = trainer.log_xgboost_test_report(
        y_train=np.array([0.01, -0.02, 0.03]),
        y_val=np.array([0.02, -0.01, 0.04]),
        y_test=y_test,
        direction_y_test=direction_y_test,
        test_split_metadata=test_split_metadata,
        classifier_predictions=classifier_predictions,
        classifier_validation_probability_up=np.array([0.55, 0.45, 0.65]),
        classifier_probability_up=np.array([0.60, 0.40, 0.70]),
        regressor_predictions=regressor_predictions,
        random_trials=20,
        random_trial_workers=2,
        regressor_validation_selection_report={
            "candidates": [],
            "selected_candidate_name": "candidate_0_baseline",
            "selected_validation_top_n_mean_excess_return": 0.01,
        },
    )

    np.testing.assert_array_equal(captured["y_true"], direction_y_test)
    np.testing.assert_array_equal(captured["y_pred"], classifier_predictions)
    pd.testing.assert_frame_equal(captured["split_metadata"], test_split_metadata)
    pd.testing.assert_frame_equal(
        captured["classifier_split_metadata"], test_split_metadata
    )
    np.testing.assert_array_equal(captured["ranked_predictions"], regressor_predictions)
    np.testing.assert_array_equal(
        captured["classifier_probabilities"],
        np.array([0.60, 0.40, 0.70]),
    )
    assert captured["prediction_days"] == 10
    assert captured["classifier_prediction_days"] == 10
    assert captured["random_trials"] == 20
    assert captured["classifier_random_trials"] == 20
    assert captured["random_trial_workers"] == 2
    assert captured["classifier_random_trial_workers"] == 2
    assert set(report) == {
        "regressor_validation_selection",
        "ranked_selection",
        "basket_backtest",
        "basket_backtest_by_year",
        "classifier_probability_ranked_selection",
        "classifier_probability_basket_backtest",
        "classifier_probability_basket_backtest_by_year",
    }
    assert report["regressor_validation_selection"]["selected_candidate_name"] == (
        "candidate_0_baseline"
    )
    assert report["basket_backtest_by_year"] == {"top_5": {"2024": {"model": {}}}}
    assert report["classifier_probability_basket_backtest_by_year"] == {
        "top_5": {"2024": {"classifier_model": {}}}
    }
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
    basket_by_year_report = {
        "top_5": {
            "2023": {
                "model": {
                    "average_basket_excess_return": 0.01,
                    "beat_benchmark_rate": 0.5288,
                    "evaluated_dates": 3,
                },
                "random_baseline": {
                    "average_basket_excess_return": 0.004,
                },
                "momentum_baseline": {
                    "available": True,
                    "average_basket_excess_return": 0.008,
                },
                "relative_momentum_baseline": {
                    "available": True,
                    "average_basket_excess_return": 0.009,
                },
                "universe": {
                    "average_basket_excess_return": 0.0015,
                },
            }
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
            "basket_backtest_by_year": basket_by_year_report,
        },
    )
    monkeypatch.setattr(
        trainer,
        "build_probability_ranked_top_n_selection_reports",
        lambda *args, **kwargs: {
            "ranked_selection": top_n_report,
            "basket_backtest": basket_report,
            "basket_backtest_by_year": basket_by_year_report,
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
    assert any(
        message.startswith("XGBoost Top-N Basket Backtest By-Year Summary:")
        for message in info_messages
    )
    assert not any(
        message.startswith(
            "XGBoost Classifier-Probability Top-N Basket Backtest By-Year Summary:"
        )
        for message in info_messages
    )
    assert any(
        message.startswith(
            "XGBoost Classifier-Probability Top-N Ranked Selection Summary:"
        )
        for message in info_messages
    )
    assert any(
        message.startswith(
            "XGBoost Classifier-Probability Top-N Basket Backtest Summary:"
        )
        for message in info_messages
    )
    classifier_ranked_summary = next(
        message
        for message in info_messages
        if message.startswith(
            "XGBoost Classifier-Probability Top-N Ranked Selection Summary:"
        )
    )
    classifier_basket_summary = next(
        message
        for message in info_messages
        if message.startswith(
            "XGBoost Classifier-Probability Top-N Basket Backtest Summary:"
        )
    )
    assert "XGBoost Top-N Ranked Selection Summary:" not in classifier_ranked_summary
    assert "XGBoost Top-N Basket Backtest Summary:" not in classifier_basket_summary
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
    assert "XGBoost Top-N Basket Backtest By-Year Report:" in caplog.text
    assert (
        "XGBoost Classifier-Probability Top-N Ranked Selection Report:" in caplog.text
    )
    assert "XGBoost Classifier-Probability Top-N Basket Backtest Report:" in caplog.text
    assert (
        "XGBoost Classifier-Probability Top-N Basket Backtest By-Year Report:"
        in caplog.text
    )


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
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(ranked_predictions)
        captured["prediction_days"] = prediction_days
        captured["random_trials"] = random_trials
        captured["random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"model": {}}},
            "basket_backtest": basket_report,
        }

    monkeypatch.setattr(
        trainer,
        "build_top_n_selection_reports",
        fake_build_top_n_selection_reports,
    )
    monkeypatch.setattr(
        trainer,
        "build_probability_ranked_top_n_selection_reports",
        lambda *args, **kwargs: {
            "ranked_selection": {"top_5": {"model": {}}},
            "basket_backtest": basket_report,
        },
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
            random_trials=20,
            random_trial_workers=2,
        )

    pd.testing.assert_frame_equal(captured["split_metadata"], test_split_metadata)
    np.testing.assert_array_equal(captured["ranked_predictions"], regressor_predictions)
    assert captured["prediction_days"] == 10
    assert captured["random_trials"] == 20
    assert captured["random_trial_workers"] == 2
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
