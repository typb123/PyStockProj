"""Target-free daily feature preparation for ranked-watchlist inference."""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.data_fetch import fetch_stock_data
from src.data.technical_indicators import calculate_data
from src.data.training_data import normalize_raw_ohlcv_index


MARKET_TIMEZONE = ZoneInfo("America/New_York")
CURRENT_SESSION_CUTOFF = time(16, 30)
COMPLETED_SESSION_POLICY = (
    "Use daily bars before the current America/New_York calendar date, and allow "
    "the current-date bar at or after 16:30 America/New_York."
)


class InferenceDataError(ValueError):
    """Raised when one ticker cannot provide target-free inference features."""


def _current_market_timestamp(now=None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz=MARKET_TIMEZONE) if now is None else pd.Timestamp(now)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(MARKET_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(MARKET_TIMEZONE)
    return timestamp


def completed_daily_history(data: pd.DataFrame, *, now=None) -> pd.DataFrame:
    """Normalize daily OHLCV and retain only conservatively completed sessions."""
    if data.empty:
        return data.copy()
    normalized = normalize_raw_ohlcv_index(data)
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    market_now = _current_market_timestamp(now)
    market_date = market_now.normalize().tz_localize(None)
    latest_eligible_date = (
        market_date
        if market_now.time() >= CURRENT_SESSION_CUTOFF
        else market_date - pd.Timedelta(days=1)
    )
    return normalized.loc[normalized.index <= latest_eligible_date].copy()


def prepare_completed_ticker_features(
    ticker: str,
    *,
    period: str = "5y",
    now=None,
) -> pd.DataFrame:
    """Fetch one ticker and calculate features without constructing future targets."""
    raw_data = fetch_stock_data(ticker, period=period)
    if raw_data.empty:
        raise InferenceDataError("no market data returned")

    completed_history = completed_daily_history(raw_data, now=now)
    if completed_history.empty:
        raise InferenceDataError("no completed daily bars available")

    try:
        features = calculate_data(completed_history)
    except (KeyError, TypeError, ValueError) as error:
        raise InferenceDataError(f"feature calculation failed: {error}") from error

    features["prediction_date"] = pd.to_datetime(features.index).normalize()
    features["Ticker"] = ticker
    return features
