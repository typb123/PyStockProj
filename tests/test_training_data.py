"""Focused ownership tests for training-data acquisition and assembly."""

import logging

import pandas as pd
import pytest

import src.data.training_data as training_data


def test_default_cache_path_preserves_project_relative_location_and_filename():
    cache_path = training_data.get_yfinance_cache_path("brk/b", "1 y")

    assert cache_path == (
        training_data.PROJECT_ROOT / "data/cache/yfinance/BRK_B__1_y.csv"
    )
    assert training_data.YFINANCE_CACHE_DIR == (
        training_data.PROJECT_ROOT / "data/cache/yfinance"
    )
    assert training_data.YFINANCE_CACHE_FORMAT == "csv"


def test_prepare_data_parallel_preserves_input_order_and_filters_skips(monkeypatch):
    calls = []

    def fake_fetch_tickers_data(ticker, period="5y", use_cache=True):
        calls.append((ticker, period, use_cache))
        if ticker == "BAD":
            return None
        return pd.DataFrame(
            {"Ticker": [ticker], "value": [{"MSFT": 1, "AAPL": 3}[ticker]]}
        )

    monkeypatch.setattr(
        training_data,
        "fetch_tickers_data",
        fake_fetch_tickers_data,
    )

    result = training_data.prepare_data_parallel(
        ["MSFT", "BAD", "AAPL"],
        period="1y",
        use_cache=False,
    )

    assert sorted(calls) == sorted(
        [
            ("MSFT", "1y", False),
            ("BAD", "1y", False),
            ("AAPL", "1y", False),
        ]
    )
    assert result.to_dict("records") == [
        {"Ticker": "MSFT", "value": 1},
        {"Ticker": "AAPL", "value": 3},
    ]


def test_prepare_data_parallel_raises_when_every_ticker_is_skipped(monkeypatch):
    monkeypatch.setattr(
        training_data,
        "fetch_tickers_data",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="No data was fetched for any ticker"):
        training_data.prepare_data_parallel(["BAD", "WORSE"])


def test_explicit_cache_directory_remains_supported(tmp_path):
    cache_path = training_data.get_yfinance_cache_path(
        "AAPL",
        "10y",
        cache_dir=tmp_path,
    )

    assert cache_path == tmp_path / "AAPL__10y.csv"


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

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(training_data, "calculate_data", fake_calculate_data)

    result = training_data.prepare_data_parallel(
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

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(training_data, "calculate_data", fake_calculate_data)

    with caplog.at_level(logging.INFO):
        result = training_data.prepare_data_parallel(
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

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(training_data, "calculate_data", lambda df: df)

    result = training_data.prepare_data_parallel(
        ["SPY", "AAPL"],
        period="1y",
        use_cache=False,
    )

    assert calls == [("SPY", "1y"), ("AAPL", "1y")]
    assert set(result["Ticker"]) == {"SPY", "AAPL"}


def test_empty_fetch_returns_without_normalizing_or_writing_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
    empty_data = pd.DataFrame()
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return empty_data

    def fake_normalize_raw_ohlcv_index(data):
        raise AssertionError("empty data should not be normalized")

    def fake_write_yfinance_cache(data, ticker, period, cache_dir=None):
        raise AssertionError("empty data should not be cached")

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(
        training_data,
        "normalize_raw_ohlcv_index",
        fake_normalize_raw_ohlcv_index,
    )
    monkeypatch.setattr(
        training_data, "write_yfinance_cache", fake_write_yfinance_cache
    )

    result = training_data.fetch_raw_ticker_data("BAD", period="1y", use_cache=True)

    assert calls == [("BAD", "1y")]
    assert result.empty


def test_cache_hit_loads_cached_data_without_fetching_yfinance(tmp_path, monkeypatch):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = training_data.get_yfinance_cache_path("AAPL", "1y")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_data = pd.DataFrame(
        {"Close": [100.0], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    cached_data.index.name = "Date"
    cached_data.to_csv(cache_path)

    def fake_fetch_stock_data(ticker, period="5y"):
        raise AssertionError("yfinance fetch should not be called on cache hit")

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)

    result = training_data.fetch_raw_ticker_data(
        "AAPL", period="1y", use_cache=True
    )

    pd.testing.assert_frame_equal(result, cached_data)


def test_cache_hit_normalizes_timezone_aware_index(tmp_path, monkeypatch):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = training_data.get_yfinance_cache_path("AAPL", "1y")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "Date,Close\n2024-03-11 00:00:00-04:00,100.0\n2024-03-08 00:00:00-05:00,99.0\n",
        encoding="utf-8",
    )

    def fake_fetch_stock_data(ticker, period="5y"):
        raise AssertionError("yfinance fetch should not be called on cache hit")

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)

    result = training_data.fetch_raw_ticker_data(
        "AAPL", period="1y", use_cache=True
    )

    expected = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-03-08", "2024-03-11"]),
    )
    expected.index.name = "Date"
    pd.testing.assert_frame_equal(result, expected)
    assert result.index.tz is None


def test_cache_miss_fetches_data_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
    fetched_data = pd.DataFrame(
        {"Close": [100.0], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    fetched_data.index.name = "Date"
    calls = []

    def fake_fetch_stock_data(ticker, period="5y"):
        calls.append((ticker, period))
        return fetched_data

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)

    result = training_data.fetch_raw_ticker_data(
        "AAPL", period="1y", use_cache=True
    )

    assert calls == [("AAPL", "1y")]
    pd.testing.assert_frame_equal(result, fetched_data)
    cache_path = training_data.get_yfinance_cache_path("AAPL", "1y")
    assert cache_path.exists()
    cached_result = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(cached_result, fetched_data)


def test_cache_miss_normalizes_fetched_timezone_aware_index_before_caching(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
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

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)

    result = training_data.fetch_raw_ticker_data(
        "AAPL", period="1y", use_cache=True
    )

    expected = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-03-08", "2024-03-11"]),
    )
    expected.index.name = "Date"
    pd.testing.assert_frame_equal(result, expected)
    assert result.index.tz is None

    cache_path = training_data.get_yfinance_cache_path("AAPL", "1y")
    cached_result = pd.read_csv(cache_path, index_col=0)
    cached_result = training_data.normalize_raw_ohlcv_index(cached_result)
    pd.testing.assert_frame_equal(cached_result, expected)


def test_normalize_raw_ohlcv_index_handles_mixed_dst_offsets():
    data = pd.DataFrame(
        {"Close": [100.0, 99.0]},
        index=[
            "2024-03-11 00:00:00-04:00",
            "2024-03-08 00:00:00-05:00",
        ],
    )

    result = training_data.normalize_raw_ohlcv_index(data)

    expected = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-03-08", "2024-03-11"]),
    )
    expected.index.name = "Date"
    pd.testing.assert_frame_equal(result, expected)
    assert result.index.tz is None


def test_no_cache_bypasses_cache_read_and_write(tmp_path, monkeypatch):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = training_data.get_yfinance_cache_path("AAPL", "1y")
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

    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)

    result = training_data.fetch_raw_ticker_data(
        "AAPL", period="1y", use_cache=False
    )

    assert calls == [("AAPL", "1y")]
    pd.testing.assert_frame_equal(result, fetched_data)
    unchanged_cache = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(unchanged_cache, cached_data)


def test_cache_read_failure_falls_back_to_fetch(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(training_data, "YFINANCE_CACHE_DIR", tmp_path)
    cache_path = training_data.get_yfinance_cache_path("AAPL", "1y")
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
    monkeypatch.setattr(training_data, "fetch_stock_data", fake_fetch_stock_data)

    with caplog.at_level(logging.WARNING):
        result = training_data.fetch_raw_ticker_data(
            "AAPL", period="1y", use_cache=True
        )

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

    result = training_data.validate_input_data(data)

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
        training_data.validate_input_data(data)

    assert "Data validation complete." in caplog.text
    assert "NaN values before preparation" not in caplog.text
    assert "Column Close:" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        training_data.validate_input_data(data)

    assert "NaN values before preparation: 1" in caplog.text
    assert "Column Close: 1 NaN values" in caplog.text
