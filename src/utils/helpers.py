import pandas as pd
import logging
from pandas.tseries.offsets import BDay


def get_prediction_date(stockData: pd.DataFrame, tradingDaysAhead: int) -> str:
    """
    Returns the projected date `tradingDaysAhead` from the most recent date in the stock data,
    skipping non-trading days (weekends and holidays).

    Parameters:
        stockData (pd.DataFrame): A DataFrame with a datetime index (yfinance format).
        tradingDaysAhead (int): The number of trading days ahead to predict.

    Returns:
        str: The projected future date (formatted MM/DD/YYYY).
    """
    try:
        last_trading_date = stockData.index.max()  # Get the most recent trading date
        future_date = last_trading_date + BDay(tradingDaysAhead)  # Add business days
        return future_date.strftime("%m/%d/%Y")  # Format the date as MM/DD/YYYY
    except Exception as e:
        logging.error(f"Error calculating future prediction date: {e}")
        raise