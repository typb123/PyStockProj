import joblib
import logging
import copy
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from src.config import (
    MODEL_PATHS,
    REQUIRED_COLUMNS,
    LOG_FILE,
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
)
from src.data.technical_indicators import CalculateData
from src.data.data_fetch import fetch_stock_data
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s: - %(levelname)s -%(message)s'
)
def predict_price(ticker: str) -> dict:
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
        params = copy.deepcopy(XG_PARAMS_CLASSIFIER)  # Use the same parameters for both models
        
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
        data = fetch_stock_data(ticker, period="5y")   
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
