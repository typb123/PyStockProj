"""Yahoo Finance data access helpers.

The main training path uses daily OHLCV history for technical indicators. EPS
helpers are optional fundamental-data utilities and are not currently part of
the required model feature path.
"""

import pandas as pd
import logging
import yfinance as yf
from src.config import LOG_FILE


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s: - %(levelname)s -%(message)s'
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


YFINANCE_PROVIDER = "Yahoo Finance"
YFINANCE_LIBRARY = "yfinance"
YFINANCE_AUTO_ADJUST = True
YFINANCE_PRICE_CONVENTION = "yfinance_auto_adjusted_ohlcv"


def get_yfinance_data_provenance(training_period: str | None = None) -> dict:
    """Return the market-data semantics recorded with trained artifacts."""
    return {
        "provider": YFINANCE_PROVIDER,
        "library": YFINANCE_LIBRARY,
        "library_version": str(getattr(yf, "__version__", "unknown")),
        "auto_adjust": YFINANCE_AUTO_ADJUST,
        "price_convention": YFINANCE_PRICE_CONVENTION,
        "training_period": training_period,
    }


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch adjusted historical OHLCV with explicit yfinance adjustment semantics.

    Provider-specific extra columns such as Dividends, Stock Splits, or Capital
    Gains may be present and are cleaned or ignored downstream.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The period to fetch data for. Defaults to "5y".

    Returns:
        pd.DataFrame: The historical stock data.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, auto_adjust=YFINANCE_AUTO_ADJUST)

        if data.empty:
            logging.debug(f"No data found for {ticker}. DataFrame is empty.")

        return data
    except Exception as e:
        logging.debug(f"Error fetching data for {ticker}: {e}", exc_info=True)
        # Keep callers on a single DataFrame path; training skips empty ticker results.
        return pd.DataFrame()


def fetch_eps_data(ticker: str) -> pd.DataFrame:
    """
    Fetch optional reported EPS data from Yahoo Finance.

    EPS is a fundamental-data helper and is not currently required by the core
    technical-indicator training features.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        pd.DataFrame: Reported EPS values by date, or empty DataFrame if unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates

        if earnings_dates is None or 'Reported EPS' not in earnings_dates.columns:
            logging.warning(f"No earnings EPS data available for {ticker}.")
            return pd.DataFrame()
        eps_data = earnings_dates[['Reported EPS']].reset_index()
        eps_data.columns = ['Date', 'EPS']
        eps_data['Date'] = pd.to_datetime(eps_data['Date'])
        return eps_data.sort_values(by='Date')
    except Exception as e:
        logging.error(f"Error fetching EPS data for {ticker}: {e}")
        return pd.DataFrame()


def add_eps_to_daily_data(daily_data: pd.DataFrame, eps_data: pd.DataFrame) -> pd.DataFrame:
    """
    Attach optional EPS values to daily price data using the latest prior report.

    This helper is kept separate from the core technical-indicator path.

    Args:
        daily_data (pd.DataFrame): The daily stock data.
        eps_data (pd.DataFrame): Reported EPS data by date.

    Returns:
        pd.DataFrame: The daily stock data with EPS data.
    """
    try:
        if eps_data.empty:
            daily_data['EPS'] = pd.NA
            return daily_data

        daily_data_reset = daily_data.reset_index()

        merged_data = pd.merge_asof(
            daily_data_reset.sort_values('Date'),
            eps_data.sort_values('Date'),
            on='Date',
            direction='backward'
        )
        return merged_data.set_index('Date')

    except Exception as e:
        logging.error(f"Error adding EPS data to daily data: {e}")
        return daily_data
