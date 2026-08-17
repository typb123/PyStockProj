"""Cached market-data acquisition and preparation for model training."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.data.data_fetch import fetch_stock_data
from src.data.technical_indicators import calculate_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YFINANCE_CACHE_DIR = PROJECT_ROOT / "data/cache/yfinance"
YFINANCE_CACHE_FORMAT = "csv"


def validate_input_data(data):
    """
    Validate the input data structure before feature/target preparation.

    Missing values are logged but not filled here; DataPreparator handles required
    feature/target row dropping without future-looking imputation.
    """
    if data.empty:
        raise ValueError("Input data is empty.")

    nan_count = data.isnull().sum().sum()
    logging.debug(f"NaN values before preparation: {nan_count}")

    if nan_count > 0:
        nan_by_column = data.isnull().sum()
        nan_columns = [col for col in data.columns if nan_by_column[col] > 0]
        for col in nan_columns:
            logging.debug(
                f"Column {col}: {nan_by_column[col]} NaN values ({nan_by_column[col] / len(data) * 100:.2f}%)"
            )

    logging.info("Data validation complete.")
    return data


def _sanitize_cache_key(value):
    """Normalize ticker and period strings for filesystem cache paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def get_yfinance_cache_path(ticker, period, cache_dir=None):
    """Return the raw OHLCV cache path for a ticker/period pair."""
    cache_dir = YFINANCE_CACHE_DIR if cache_dir is None else cache_dir
    ticker_key = _sanitize_cache_key(ticker.upper())
    period_key = _sanitize_cache_key(period)
    return Path(cache_dir) / f"{ticker_key}__{period_key}.{YFINANCE_CACHE_FORMAT}"


def normalize_raw_ohlcv_index(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw OHLCV indexes to sorted, timezone-naive date timestamps."""
    normalized_data = data.copy()
    normalized_index = pd.DatetimeIndex(pd.to_datetime(normalized_data.index, utc=True))
    normalized_data.index = normalized_index.tz_convert(None).normalize()
    normalized_data.index.name = "Date"
    return normalized_data.sort_index()


def load_cached_yfinance_data(ticker, period, cache_dir=None):
    """Load raw yfinance OHLCV data from cache, or return None on miss/failure."""
    cache_path = get_yfinance_cache_path(ticker, period, cache_dir=cache_dir)
    if not cache_path.exists():
        logging.info(f"YFinance cache miss for {ticker} period={period}.")
        return None

    try:
        data = pd.read_csv(cache_path, index_col=0)
        data = normalize_raw_ohlcv_index(data)
        logging.info(f"YFinance cache hit for {ticker} period={period}: {cache_path}")
        return data
    except Exception as exc:
        logging.warning(
            f"Failed to read YFinance cache for {ticker} period={period} "
            f"from {cache_path}; refetching. Error: {exc}"
        )
        return None


def write_yfinance_cache(data, ticker, period, cache_dir=None):
    """Write raw yfinance OHLCV data to cache; warn but do not fail on errors."""
    cache_path = get_yfinance_cache_path(ticker, period, cache_dir=cache_dir)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path)
        logging.info(f"Wrote YFinance cache for {ticker} period={period}: {cache_path}")
    except Exception as exc:
        logging.warning(
            f"Failed to write YFinance cache for {ticker} period={period} "
            f"to {cache_path}. Error: {exc}"
        )


def fetch_raw_ticker_data(ticker, period="5y", use_cache=True):
    """Fetch raw OHLCV data, optionally using the ticker/period cache."""
    if use_cache:
        cached_data = load_cached_yfinance_data(ticker, period)
        if cached_data is not None:
            return cached_data

    stock_data = fetch_stock_data(ticker, period=period)
    if stock_data.empty:
        return stock_data

    stock_data = normalize_raw_ohlcv_index(stock_data)
    if use_cache:
        write_yfinance_cache(stock_data, ticker, period)
    return stock_data


def fetch_tickers_data(ticker, period="5y", use_cache=True):
    """
    Fetch one ticker and generate indicators before ticker-level concatenation.

    calculate_data is order-dependent and not grouped internally, so the current
    fetch path calls it while each DataFrame still contains one ticker only.
    """
    try:
        stock_data = fetch_raw_ticker_data(ticker, period=period, use_cache=use_cache)
        if stock_data.empty:
            logging.debug(f"No data returned for {ticker}. Skipping...")
            return None

        logging.debug(f"Data size before processing for {ticker}: {stock_data.shape}")

        stock_data = calculate_data(stock_data)
        stock_data["prediction_date"] = stock_data.index
        stock_data["Ticker"] = ticker

        logging.debug(f"Data size after processing for {ticker}: {stock_data.shape}")

        time.sleep(0.2)  # Small delay to avoid rate limits
        return stock_data
    except Exception as e:
        logging.debug(f"Failed to fetch data for {ticker}: {e}", exc_info=True)
        return None


def prepare_data_parallel(tickers, period="5y", use_cache=True) -> pd.DataFrame:
    """Fetch and prepare each ticker independently, then concatenate valid results."""
    logging.info(
        f"Fetching data for {len(tickers)} tickers with period={period}; "
        f"cache_enabled={use_cache}."
    )

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(
                lambda ticker: fetch_tickers_data(
                    ticker,
                    period=period,
                    use_cache=use_cache,
                ),
                tickers,
            )
        )

    all_data = [data for data in results if data is not None and not data.empty]
    skipped_tickers = [
        ticker for ticker, data in zip(tickers, results) if data is None or data.empty
    ]

    logging.info(
        f"Fetched valid data for {len(all_data)} of {len(tickers)} requested tickers."
    )
    if skipped_tickers:
        logging.warning(
            f"Skipped {len(skipped_tickers)} tickers with no usable data: {skipped_tickers}"
        )

    if not all_data:
        logging.error("No data fetched for training. Exiting...")
        raise ValueError("No data was fetched for any ticker.")

    return pd.concat(all_data, ignore_index=True)
