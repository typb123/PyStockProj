import joblib
import logging
import copy
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from src.config import (
    MODEL_PATHS,
    LOG_FILE,
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
)
from src.data.technical_indicators import calculate_data
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
        scaler_lr = joblib.load(MODEL_PATHS['linear_scaler'])
        model_metadata = joblib.load(MODEL_PATHS['model_metadata'])

        # Load XGBoost models - create with same parameters used during training
        classifier_params = copy.deepcopy(XG_PARAMS_CLASSIFIER)
        regressor_params = copy.deepcopy(XG_PARAMS_REGRESSOR)

        # Create fresh instances with the same parameters
        xgb_classifier = XGBClassifier(**classifier_params)
        xgb_regressor = XGBRegressor(**regressor_params)

        # Load model data
        xgb_classifier.load_model(MODEL_PATHS['classifier'].replace('.pkl', '.json'))
        xgb_regressor.load_model(MODEL_PATHS['regressor'].replace('.pkl', '.json'))

        # Load preparator
        data_preparator = joblib.load(MODEL_PATHS['preparator'])
        feature_columns = data_preparator.feature_columns

        # Fetch recent stock data
        data = fetch_stock_data(ticker, period="5y")
        processed_data = calculate_data(data)

        missing_cols = set(feature_columns) - set(processed_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check if processed_data is empty
        if processed_data.empty:
            raise ValueError(f"No data available for ticker {ticker}")

        # Extract latest features
        latest_feature_row = processed_data.iloc[-1][feature_columns]
        numeric_latest_feature_row = pd.to_numeric(latest_feature_row, errors="coerce")
        invalid_features = numeric_latest_feature_row.index[
            latest_feature_row.isna()
            | numeric_latest_feature_row.isna()
            | ~np.isfinite(numeric_latest_feature_row)
        ].tolist()
        if invalid_features:
            raise ValueError(f"Invalid latest feature values for {ticker}: {invalid_features}")

        latest_features = numeric_latest_feature_row.to_numpy(dtype=float).reshape(1, -1)
        latest_features = data_preparator.scalar.transform(latest_features)
        latest_features_df = pd.DataFrame(latest_features, columns=feature_columns)

        # Add Linear Regression Prediction as a feature
        linear_features = model_metadata["linear_features"]
        lr_features = scaler_lr.transform(latest_features_df[linear_features])
        lr_prediction = linear_model.predict(lr_features)[0]

        classifier_features = model_metadata["classifier_features"]
        regressor_features = model_metadata["regressor_features"]
        classifier_input = latest_features_df[classifier_features]
        regressor_input = latest_features_df[regressor_features]

        # Make predictions
        predicted_direction = xgb_classifier.predict(classifier_input)[0]
        predicted_return = xgb_regressor.predict(regressor_input)[0]

        # Interpret results
        direction = "Up" if predicted_direction == 1 else "Down"
        last_close_price = processed_data['Close'].iloc[-1]
        expected_price = last_close_price * (1 + predicted_return)

        result = {
            'direction': direction,
            'linear_predicted_return': float(lr_prediction),
            'predicted_return': float(predicted_return),
            'expected_price': float(expected_price)
        }

        logging.info(f"Prediction for {ticker}: {result}")
        return result
    except ValueError:
        logging.error(f"Error predicting price for {ticker}: invalid input data")
        raise
    except Exception as e:
        logging.error(f"Error predicting price for {ticker}: {e}")
        return {"error": str(e)}
