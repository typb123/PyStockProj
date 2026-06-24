import joblib
import pandas as pd
import numpy as np
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    classification_report,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from src.data.data_prep import DataPreparator
from src.data.technical_indicators import calculate_data
from src.data.data_fetch import fetch_stock_data
from xgboost import XGBRegressor, XGBClassifier
from typing import Optional
from src.config import (
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR, 
    MODEL_PATHS,
    TRAINING_TICKERS,
    EARLY_STOPPING_ROUNDS,
    PREDICTION_DAYS,
    TEST_SIZE,
)


logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s: - %(levelname)s -%(message)s'
)
def validate_input_data(data):
    """
    Validate the input data to ensure there are no NaN values.
    """
    # Check initial NaN values
    nan_count_before = data.isnull().sum().sum()
    logging.info(f"NaN values before filling: {nan_count_before}")
    
    if nan_count_before > 0:
        # Log NaN distribution by column
        nan_by_column = data.isnull().sum()
        nan_columns = [col for col in data.columns if nan_by_column[col] > 0]
        for col in nan_columns:
            logging.info(f"Column {col}: {nan_by_column[col]} NaN values ({nan_by_column[col]/len(data)*100:.2f}%)")
        
        # Standard filling - avoid chained assignment
        data = data.ffill().bfill()
        
        # Debug - check NaNs after standard filling
        remaining_nan_count = data.isnull().sum().sum()
        logging.info(f"NaN values after ffill/bfill: {remaining_nan_count}")
        
        # If still have NaNs, handle each column properly
        if remaining_nan_count > 0:
            for col in data.columns:
                if data[col].isnull().any():
                    if col == 'ATR':
                        # Special handling for ATR column
                        logging.info(f"Handling ATR column (has {data['ATR'].isnull().sum()} NaNs)")
                        atr_mean = data.loc[data['ATR'].notnull(), 'ATR'].mean()
                        data.loc[data['ATR'].isnull(), 'ATR'] = atr_mean
                    elif data[col].dtype.kind in 'ifc':
                        # For numeric columns, use mean
                        col_mean = data[col].mean()
                        if pd.isna(col_mean):  # If mean itself is NaN
                            data.loc[data[col].isnull(), col] = 0
                        else:
                            data.loc[data[col].isnull(), col] = col_mean
                    else:
                        # For non-numeric columns
                        data.loc[data[col].isnull(), col] = 0
            
            # Final check for any remaining NaNs
            final_nan_count = data.isnull().sum().sum()
            if final_nan_count > 0:
                # One last attempt - replace any remaining NaNs with 0
                logging.warning(f"Still have {final_nan_count} NaNs after targeted filling, using zeros")
                data = data.fillna(0)
                
                # If we still have NaNs, we need to know which columns
                if data.isnull().sum().sum() > 0:
                    problem_cols = [col for col in data.columns if data[col].isnull().any()]
                    logging.error(f"Columns still containing NaNs: {problem_cols}")
                    # Drop problematic columns as last resort
                    data = data.drop(columns=problem_cols)
                    logging.warning(f"Dropped problematic columns: {problem_cols}")
    
    # Final validation
    if data.isnull().values.any():
        raise ValueError("Data still contains NaN values after all filling attempts.")
    
    logging.info("Data validation complete - no NaN values remain.")
    return data
            
def fetch_tickers_data(ticker, period="5y"):
    try:
        stock_data = fetch_stock_data(ticker, period=period)
        if stock_data.empty:
            logging.warning(f"No data returned for {ticker}. Skipping...")
            return None
        
        # Log data size before processing
        logging.info(f"Data size before processing for {ticker}: {stock_data.shape}")
        
        stock_data = calculate_data(stock_data)
        stock_data["Ticker"] = ticker
        
        # Log data size after processing
        logging.info(f"Data size after processing for {ticker}: {stock_data.shape}")
        
        time.sleep(0.2)  # Small delay to avoid rate limits
        return stock_data
    except Exception as e:
        logging.warning(f"Failed to fetch data for {ticker}: {e}")
        return None
    

    
def prepare_data_parallel(tickers, period="5y") -> pd.DataFrame:
    logging.info(f"Fetching data for {len(tickers)} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda ticker: fetch_tickers_data(ticker, period=period), tickers)
    
    all_data = [data for data in results if data is not None and not data.empty]
    
    logging.info(f"Total tickers fetched with valid data: {len(all_data)}")

    if not all_data:
        logging.error("No data fetched for training. Exiting...")
        raise ValueError("No data was fetched for any ticker.")
    
    return pd.concat(all_data, ignore_index=True)

def cross_validate_model(x, y, model=None, cv=5):
    """
    Perform cross-validation on the given model and data.
    
    Parameters:
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Target variable.
        model: Scikit-learn model (default: LinearRegression).
        cv (int): Number of cross-validation folds.
    
    Returns:
        float: Average cross-validated MSE.
    """
    if model is None:
        model = LinearRegression()
    
    # Define custom scoring for negative MSE
    scores = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=cv)
    if scores is None:
        raise ValueError("Cross-validation scoring failed. Please check your model and data.")
    avg_mse = -np.mean(scores)
    logging.info(f"Cross-Validated MSE (cv={cv}): {avg_mse:.4f}")
    return avg_mse
    

def evaluate_model(model, x_test, y_test, model_type="regression"):
    y_pred = model.predict(x_test)

    if model_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        #print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    elif model_type == "classification":
        accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
        report = classification_report(y_test.astype(int), y_pred.astype(int))
        logging.info(f"Accuracy: {accuracy:.4%}")
        logging.info(f"Classification Report:\n{report}")
        #print(f" Accuracy: {accuracy:.4%}")
        #print(f" Classification Report:\n{report}")

    
    
def train_models(data: pd.DataFrame) -> None:
    logging.info("Training models with explicitly defined feature arrays.")

    # Validate and prepare data
    data = validate_input_data(data)
    data_preparator = DataPreparator()
    prepared_data = data_preparator.prepare_for_train(
        data, prediction_days=PREDICTION_DAYS, test_size=TEST_SIZE
    )

    # Extract train/test datasets with feature names
    all_features = prepared_data["feature_names"]
    x_train_full = pd.DataFrame(prepared_data["x_train"], columns=all_features)
    x_val_full = pd.DataFrame(prepared_data["x_val"], columns=all_features)
    x_test_full = pd.DataFrame(prepared_data["x_test"], columns=all_features)
    y_train = prepared_data["y_train"]
    y_val = prepared_data["y_val"]
    y_test = prepared_data["y_test"]

    # Define feature sets for each model
    linear_features = [
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 
        'Open', 'Close', 'rsi', 'signalLine', 'ATR', 
        '20_day_avg', 'macd', 'BB_Std', 'obv', 'dailyReturn', 
        'macdHistogram',  'vma_20',  'High', 'Low', 'BB_Middle', 
        'BB_Upper', 'BB_Lower','stoch_k', 'stoch_d', 'Volume', 
        '10_day_avg', 'volatility', 'vma_10', '5_day_avg',
    ] 
    # REMOVED FROM LINEAR REGRESSION:   
    #  
    classifier_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '5_day_avg', '10_day_avg', '20_day_avg',
        'dailyReturn', 'volatility', 'rsi',
        'macd', 'signalLine', 'macdHistogram',
        'obv', 'vma_10', 'vma_20',
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Std',
        'ATR', 'stoch_k', 'stoch_d'
    ]
    regressor_features  = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '5_day_avg', '10_day_avg', '20_day_avg',
        'dailyReturn', 'volatility', 'rsi',
        'macd', 'signalLine', 'macdHistogram',
        'obv', 'vma_10', 'vma_20',
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Std',
        'ATR', 'stoch_k', 'stoch_d'
    ] 

    # Standardize features for Linear Regression
    scaler_lr = StandardScaler()
    x_train_lr_scaled = scaler_lr.fit_transform(x_train_full[linear_features])
    x_val_lr_scaled = scaler_lr.transform(x_val_full[linear_features])
    x_test_lr_scaled = scaler_lr.transform(x_test_full[linear_features])

    # Train Linear Regression on scaled features
    logging.info("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(x_train_lr_scaled, y_train)
    evaluate_model(linear_model, x_test_lr_scaled, y_test, model_type="regression")

    # Log feature importances for Linear Regression
    importances_lr = np.abs(linear_model.coef_)
    feature_importance_lr = pd.DataFrame({
        'feature': linear_features,
        'importance': importances_lr
    }).sort_values(by='importance', ascending=False)
    logging.info("Feature Importances for Linear Regression:")
    for _, row in feature_importance_lr.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

    # Generate Linear Regression predictions as additional feature
    train_lr_preds = linear_model.predict(x_train_lr_scaled).reshape(-1, 1)
    val_lr_preds = linear_model.predict(x_val_lr_scaled).reshape(-1, 1)
    test_lr_preds = linear_model.predict(x_test_lr_scaled).reshape(-1, 1)

    # Add LR predictions to Classifier and Regressor feature sets
    classifier_feature_names = classifier_features + ["LinearRegression_Prediction"]
    regressor_feature_names  = regressor_features + ["LinearRegression_Prediction"]

    x_train_classifier = np.column_stack((x_train_full[classifier_features], train_lr_preds))
    x_val_classifier = np.column_stack((x_val_full[classifier_features], val_lr_preds))
    x_test_classifier = np.column_stack((x_test_full[classifier_features], test_lr_preds))
    x_train_regressor = np.column_stack((x_train_full[regressor_features], train_lr_preds))
    x_val_regressor = np.column_stack((x_val_full[regressor_features], val_lr_preds))
    x_test_regressor = np.column_stack((x_test_full[regressor_features], test_lr_preds))

    # Binary labels for classifier
    direction_y_train = (y_train > 0).astype(int)
    direction_y_val = (y_val > 0).astype(int)
    direction_y_test = (y_test > 0).astype(int)

    # Train XGBoost Classifier
    logging.info("Training XGBoost Classifier...")
    classifier = XGBClassifier(**XG_PARAMS_CLASSIFIER)
    classifier.fit(
        x_train_classifier, direction_y_train,
        eval_set=[(x_val_classifier, direction_y_val)],
    )
    evaluate_model(classifier, x_test_classifier, direction_y_test, model_type="classification")

    # Log feature importances for XGBoost Classifier
    importances_clf = classifier.feature_importances_
    feature_importance_clf = pd.DataFrame({
        'feature': classifier_feature_names,
        'importance': importances_clf
    }).sort_values(by='importance', ascending=False)
    logging.info("Feature Importances for XGBoost Classifier:")
    for _, row in feature_importance_clf.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

    # Train XGBoost Regressor
    logging.info("Training XGBoost Regressor...")
    regressor = XGBRegressor(**XG_PARAMS_REGRESSOR)
    regressor.fit(
        x_train_regressor, y_train,
        eval_set=[(x_val_regressor, y_val)],
    )
    evaluate_model(regressor, x_test_regressor, y_test, model_type="regression")

    # Log feature importances for XGBoost Regressor
    importances_reg = regressor.feature_importances_
    feature_importance_reg = pd.DataFrame({
        'feature': regressor_feature_names,
        'importance': importances_reg
    }).sort_values(by='importance', ascending=False)
    logging.info("Feature Importances for XGBoost Regressor:")
    for _, row in feature_importance_reg.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

    # Save models
    joblib.dump(linear_model, MODEL_PATHS["linear"])
    classifier.save_model(MODEL_PATHS["classifier"].replace(".pkl", ".json"))
    regressor.save_model(MODEL_PATHS["regressor"].replace(".pkl", ".json"))
    logging.info("Training completed. Models saved successfully.")

def main():
    data = prepare_data_parallel(TRAINING_TICKERS, period="5y")
    if data.empty:
        logging.error("No data fetched for training. Exiting...")
        return
    
    logging.info("Starting model training...")
    train_models(data)
    logging.info("Model training completed.")
    print("Model training completed. Check training.log for details.")
if __name__ == "__main__":
    main()
