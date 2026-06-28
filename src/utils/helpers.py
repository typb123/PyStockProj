"""Date-formatting helpers for prediction-horizon display."""

import pandas as pd
import logging
from pandas.tseries.offsets import BDay


def get_prediction_date(stock_data: pd.DataFrame, trading_days_ahead: int) -> str:
    """Return the business-day prediction date displayed for a ticker."""
    try:
        last_trading_date = stock_data.index.max()  # Get the most recent trading date
        future_date = last_trading_date + BDay(trading_days_ahead)  # Add business days
        return future_date.strftime("%m/%d/%Y")  # Format the date as MM/DD/YYYY
    except Exception as e:
        logging.error(f"Error calculating future prediction date: {e}")
        raise
