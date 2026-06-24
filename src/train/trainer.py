import joblib
import pandas as pd
import numpy as np
import cupy as cp
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
from src.data.technical_indicators import CalculateData
from app import fetchStockData
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
            
def fetch_tickers_data(ticker):
    try:
        stockData = fetchStockData(ticker, period="5y")
        if stockData.empty:
            logging.warning(f"No data returned for {ticker}. Skipping...")
            return None
        
        # Log data size before processing
        logging.info(f"Data size before processing for {ticker}: {stockData.shape}")
        
        stockData = CalculateData(stockData)
        stockData["Ticker"] = ticker
        
        # Log data size after processing
        logging.info(f"Data size after processing for {ticker}: {stockData.shape}")
        
        time.sleep(0.2)  # Small delay to avoid rate limits
        return stockData
    except Exception as e:
        logging.warning(f"Failed to fetch data for {ticker}: {e}")
        return None
    

    
def prepare_data_parallel(tickers, period="5y") -> pd.DataFrame:
    logging.info(f"Fetching data for {len(tickers)} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_tickers_data, tickers)
    
    allData = [data for data in results if data is not None and not data.empty]
    
    logging.info(f"Total tickers fetched with valid data: {len(allData)}")

    if not allData:
        logging.error("No data fetched for training. Exiting...")
        raise ValueError("No data was fetched for any ticker.")
    
    return pd.concat(allData, ignore_index=True)

def cross_validate_model(X, Y, model=None, cv=5):
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
    scores = cross_val_score(model, cp.asnumpy(X), cp.asnumpy(Y), scoring='neg_mean_squared_error', cv=cv)
    if scores is None:
        raise ValueError("Cross-validation scoring failed. Please check your model and data.")
    avg_mse = -np.mean(scores)
    logging.info(f"Cross-Validated MSE (cv={cv}): {avg_mse:.4f}")
    return avg_mse
    

def evaluate_model(model, XTest, YTest, model_type="regression"):
    # Only call .get() if YTest is a CuPy array
    if isinstance(YTest, cp.ndarray):
        YTest = YTest.get()
        
    YPred = model.predict(XTest)
    if isinstance(YPred, cp.ndarray):
        YPred = YPred.get()

    if model_type == "regression":
        mse = mean_squared_error(YTest, YPred)
        mae = mean_absolute_error(YTest, YPred)
        r2 = r2_score(YTest, YPred)
        logging.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        #print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    elif model_type == "classification":
        accuracy = accuracy_score(YTest.astype(int), YPred.astype(int))
        report = classification_report(YTest.astype(int), YPred.astype(int))
        logging.info(f"Accuracy: {accuracy:.4%}")
        logging.info(f"Classification Report:\n{report}")
        #print(f" Accuracy: {accuracy:.4%}")
        #print(f" Classification Report:\n{report}")

    
    
def train_models(data: pd.DataFrame) -> None:
    logging.info("Training models with explicitly defined feature arrays.")

    # Validate and prepare data
    data = validate_input_data(data)
    dataPreparator = DataPreparator()
    preparedData = dataPreparator.prepareForTrain(
        data, predictionDays=PREDICTION_DAYS, testSize=TEST_SIZE
    )

    # Extract train/test datasets with feature names
    all_features = preparedData["featureNames"]
    XTrain_full = pd.DataFrame(preparedData["XTrain"], columns=all_features)
    XTest_full  = pd.DataFrame(preparedData["XTest"],  columns=all_features)
    YTrain      = preparedData["YTrain"]
    YTest       = preparedData["YTest"]

    # Define feature sets for each model
    linear_features = [
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 
        'Open', 'Close', 'rsi', 'signalLine', 'ATR', 
        '20_day_avg', 'macd', 'BB_Std', 'obv', 'dailyReturn', 
        'macdHistogram',  'vma_20',  'High', 'Low', 'BB_Middle', 
        'BB_Upper', 'BB_Lower','stoch_k', 'stoch_d', 'Volume', 
        '10_day_avg', 'volatility', 'vma_10', HERE:   '5_day_avg',
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
        'ATR', 'stoch_k', 'stoch_d', 'EPS'
    ]
    regressor_features  = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '5_day_avg', '10_day_avg', '20_day_avg',
        'dailyReturn', 'volatility', 'rsi',
        'macd', 'signalLine', 'macdHistogram',
        'obv', 'vma_10', 'vma_20',
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Std',
        'ATR', 'stoch_k', 'stoch_d', 'EPS'
    ] 

    # Standardize features for Linear Regression
    scaler_lr = StandardScaler()
    XTrain_lr_scaled = scaler_lr.fit_transform(XTrain_full[linear_features])
    XTest_lr_scaled = scaler_lr.transform(XTest_full[linear_features])

    # Train Linear Regression on scaled features
    logging.info("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(XTrain_lr_scaled, YTrain)
    evaluate_model(linear_model, XTest_lr_scaled, YTest, model_type="regression")

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
    train_lr_preds = linear_model.predict(XTrain_lr_scaled).reshape(-1, 1)
    test_lr_preds  = linear_model.predict(XTest_lr_scaled).reshape(-1, 1)

    # Add LR predictions to Classifier and Regressor feature sets
    classifier_feature_names = classifier_features + ["LinearRegression_Prediction"]
    regressor_feature_names  = regressor_features + ["LinearRegression_Prediction"]

    XTrain_classifier = np.column_stack((XTrain_full[classifier_features], train_lr_preds))
    XTest_classifier  = np.column_stack((XTest_full[classifier_features], test_lr_preds))
    XTrain_regressor  = np.column_stack((XTrain_full[regressor_features], train_lr_preds))
    XTest_regressor   = np.column_stack((XTest_full[regressor_features], test_lr_preds))

    # Convert to CuPy arrays for GPU acceleration
    XTrain_classifier_cp = cp.array(XTrain_classifier)
    XTest_classifier_cp  = cp.array(XTest_classifier)
    XTrain_regressor_cp  = cp.array(XTrain_regressor)
    XTest_regressor_cp   = cp.array(XTest_regressor)

    # Binary labels for classifier
    direction_YTrain = (YTrain > 0).astype(int)
    direction_YTest  = (YTest > 0).astype(int)

    # Train XGBoost Classifier
    logging.info("Training XGBoost Classifier...")
    classifier = XGBClassifier(**XG_PARAMS_CLASSIFIER)
    classifier.fit(
        XTrain_classifier_cp, direction_YTrain,
        eval_set=[(XTest_classifier_cp, direction_YTest)],
    )
    evaluate_model(classifier, XTest_classifier_cp, direction_YTest, model_type="classification")

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
        XTrain_regressor_cp, YTrain,
        eval_set=[(XTest_regressor_cp, YTest)],
    )
    evaluate_model(regressor, XTest_regressor_cp, YTest, model_type="regression")

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