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
from data_prep import DataPreparator
from technical_indicators import CalculateData
from main import fetchStockData
from xgboost import XGBRegressor, XGBClassifier
from typing import Optional
from config import (
    XG_PARAMS, 
    MODEL_PATHS,
    TRAINING_TICKERS,
    REQUIRED_COLUMNS,
    EARLY_STOPPING_ROUNDS,
    PREDICTION_DAYS,
    TEST_SIZE,
    CV_FOLDS
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
            
def fetchTickersData(ticker):
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
    

    
def prepareDataParallel(tickers, period="5y") -> pd.DataFrame:
    logging.info(f"Fetching data for {len(tickers)} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetchTickersData, tickers)
    
    allData = [data for data in results if data is not None and not data.empty]
    
    logging.info(f"Total tickers fetched with valid data: {len(allData)}")

    if not allData:
        logging.error("No data fetched for training. Exiting...")
        raise ValueError("No data was fetched for any ticker.")
    
    return pd.concat(allData, ignore_index=True)

def crossValidateModel(X, Y, model=None, cv=5):
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
        print(f" MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    elif model_type == "classification":
        accuracy = accuracy_score(YTest.astype(int), YPred.astype(int))
        report = classification_report(YTest.astype(int), YPred.astype(int))
        logging.info(f"Accuracy: {accuracy:.4%}")
        logging.info(f"Classification Report:\n{report}")
        print(f" Accuracy: {accuracy:.4%}")
        print(f" Classification Report:\n{report}")

    
    
def trainAndEvaluate(data: pd.DataFrame) -> None:
    """
    Trains three models: 
        - Linear Regression for baseline prediction
        - XGBoost Classifier for direction prediction
        - XGBoost Regressor for price prediction

    Parameters:
        data (pd.DataFrame): Cleaned stock data with technical indicators
        predictionDays (int): Number of days ahead for prediction
        testSize (float): Proportion of data reserved for testing
        cv (int): Number of folds for cross-validation

    Returns:
        None
    """
    logging.info("Preparing data for training...")
    validate_input_data(data)
    
    dataPreparator = DataPreparator()
    preparedData = dataPreparator.prepareForTrain(data, PREDICTION_DAYS, TEST_SIZE)

    XTrain = preparedData["XTrain"]
    XTest = preparedData["XTest"]
    YTrain = preparedData["YTrain"]
    YTest = preparedData["YTest"]
    #featureNames = preparedData["featureNames"]    ***ADD THIS BACK IN LATER***
    featureNames = [
    'Open', 'Close', 'Volume',
    '5_day_avg', '20_day_avg',
    'volatility', 'rsi', 'obv', 'vma_10',
    'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
    'chikou_span', 'BB_Std', 'BB_Middle', 'BB_Upper', 'BB_Lower', 
    'macd', 'signalLine', 'macdHistogram', 'vma_20', 'High', 
    'Low', 'stoch_k', 'stoch_d', '10_day_avg', 'dailyReturn','ATR']
    #REMOVED: 
    
    logging.info("\nTraining Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(XTrain, YTrain)
    
    #Add linear regression model as a feature
    train_lr_predictions = linear_model.predict(XTrain).reshape(-1, 1)
    test_lr_predictions = linear_model.predict(XTest).reshape(-1, 1)
    XTrain = np.column_stack((XTrain, train_lr_predictions))
    XTest = np.column_stack((XTest, test_lr_predictions))
    featureNames.append("LinearRegression_Prediction")
    
    # Convert to cupy arrays for GPU processing
    XTrain = cp.array(XTrain)
    XTest = cp.array(XTest)
    YTrain = cp.array(YTrain)
    YTest = cp.array(YTest)
    #Train XGBoost Model for Direction 
    logging.info("\nTraining XGBoost Classifier for direction prediction...")
    params = copy.deepcopy(XG_PARAMS)

    directionTrain = cp.asnumpy((YTrain > 0).astype(float))
    directionTest = cp.asnumpy((YTest > 0).astype(float))


    classifier = XGBClassifier(**params)
    classifier.fit(XTrain, directionTrain, eval_set=[(XTest, directionTest)])


    # XGBoost Regressor
    logging.info("\nTraining XGBoost Regressor for price prediction...")
    regressor = XGBRegressor(**params)
    regressor.fit(XTrain, YTrain, eval_set=[(XTest, YTest)])

    # Evaluate models
    evaluate_model(classifier, XTest, directionTest, model_type="classification")
    evaluate_model(regressor, XTest, YTest, model_type="regression")

    # Print model types before saving (for debugging)
    print(f"Linear model type: {type(linear_model)}")
    print(f"Classifier type: {type(classifier)}")
    print(f"Regressor type: {type(regressor)}")

    # Save models - linear_model with joblib, XGBoost models with their native format
    joblib.dump(linear_model, MODEL_PATHS['linear'])
    classifier.save_model(MODEL_PATHS['classifier'].replace('.pkl', '.json'))
    regressor.save_model(MODEL_PATHS['regressor'].replace('.pkl', '.json'))
    joblib.dump(dataPreparator, MODEL_PATHS['preparator'])
    joblib.dump(featureNames, MODEL_PATHS['features'])


def main():
    data = prepareDataParallel(TRAINING_TICKERS, period="5y")
    if data.empty:
        logging.error("No data fetched for training. Exiting...")
        return
    
    logging.info("Starting model training...")
    trainAndEvaluate(data)
if __name__ == "__main__":
    main()