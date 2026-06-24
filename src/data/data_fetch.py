import pandas as pd
import logging
import yfinance as yf
from src.config import LOG_FILE


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s: - %(levelname)s -%(message)s'
)

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The period to fetch data for. Defaults to "5y".

    Returns:
        pd.DataFrame: The historical stock data.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            logging.warning(f"No data found for {ticker}. DataFrame is empty.")

        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame instead of None

def fetch_eps_data(ticker: str) -> pd.DataFrame:
    """
    Fetch earnings date from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        pd.DataFrame: The earnings date.
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
    Add EPS data to daily stock data.

    Args:
        daily_data (pd.DataFrame): The daily stock data.
        eps_data (pd.DataFrame): The earnings date.

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
