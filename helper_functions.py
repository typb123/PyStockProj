import joblib
import logging
import copy
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from config import (
    MODEL_PATHS,
    REQUIRED_COLUMNS,
    XG_PARAMS,
    LOG_FILE
)
from technical_indicators import CalculateData
import yfinance as yf
from pandas.tseries.offsets import BDay

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s: - %(levelname)s -%(message)s'
)

def fetchStockData(ticker: str, period: str = "5y") -> pd.DataFrame:
    import yfinance as yf
    try:
        #logging.info(f"Fetching data for {ticker} with period={period}...")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            logging.warning(f"No data found for {ticker}. DataFrame is empty.")
        else:
            pass
            #logging.info(f"Data fetched for {ticker}: {data.shape[0]} rows.")
        
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame instead of None

def getFuturePredictionDate(stockData: pd.DataFrame, tradingDaysAhead: int) -> str:
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


def predictPrice(ticker: str) -> dict:
    """
    Predict price movement using trained models
    
    Parameters:
        ticker (str)
        
    Returns:
        dict: Contains the predicted direction, return, and expected price
    """
    try:
        logging.info(f"Loading model for {ticker}...")
        
        # Load linear model
        linear_model = joblib.load(MODEL_PATHS['linear'])
        
        # Load XGBoost models - create with same parameters used during training
        params = copy.deepcopy(XG_PARAMS)
        
        # Create fresh instances with the same parameters
        xgb_classifier = XGBClassifier(**params)
        xgb_regressor = XGBRegressor(**params)
        
        # Load model data
        xgb_classifier.load_model(MODEL_PATHS['classifier'].replace('.pkl', '.json'))
        xgb_regressor.load_model(MODEL_PATHS['regressor'].replace('.pkl', '.json'))
        
        # Load preparator
        data_preparator = joblib.load(MODEL_PATHS['preparator'])
        feature_columns = data_preparator.featureColumns 
        
        # Fetch recent stock data
        data = fetchStockData(ticker, period="5y")   
        processed_data = CalculateData(data)
        
        missing_cols = set(REQUIRED_COLUMNS) - set(processed_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}") 
            
        # Check if processed_data is empty
        if processed_data.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        
        # Extract latest features
        latest_features = processed_data.iloc[-1][feature_columns].values.reshape(1, -1)
        
        # Handle NaNs
        if np.isnan(latest_features).any():
            nan_count = np.isnan(latest_features).sum()
            if nan_count > 0:
                logging.warning(f"NaN values detected in latest features for {ticker}: {nan_count} NaNs found.")
            latest_features = np.nan_to_num(latest_features)
        
        # Add Linear Regression Prediction as a feature
        lr_prediction = linear_model.predict(latest_features).reshape(-1, 1)
        latest_features = np.column_stack((latest_features, lr_prediction))
        
        # Make predictions
        predicted_direction = xgb_classifier.predict(latest_features)[0]
        predicted_return = xgb_regressor.predict(latest_features)[0]
        
        # Interpret results
        direction = "Up" if predicted_direction == 1 else "Down"
        last_close_price = processed_data['Close'].iloc[-1]
        expected_price = last_close_price * (1 + predicted_return)
        
        result = {
            'direction': direction,
            'predicted_return': predicted_return,
            'expected_price': expected_price
        }
        
        logging.info(f"Prediction for {ticker}: {result}")
        return result
    except Exception as e:
        logging.error(f"Error predicting price for {ticker}: {e}")
        return {"error": str(e)}
