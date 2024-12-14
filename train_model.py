import joblib
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from data_prep import DataPreparator
from technical_indicators import CalculateData
from main import fetchStockData

#These are the 150 tickers I am currently training on
TRAINING_TICKERS = [
    "SPY", "XLU", "XLE", "XLV", "VOOG",
    "VOOV", "VB", "TSLA", "LAD", "JPM",
    "AAPL", "SCHW", "NVDA", "META", "GE",
    "INTC", "BX", "KR", "FDX", "MSFT",
    "CVX", "PSX", "PG", "PFE", "AMGN",
    "BRK-A", "BRK-B", "RCL", "AMZN", "HD",
    "DUK", "NEE", "GS", "CAT", "MCD",
    "QQQ", "VTI", "IWM", "XLF", "XLK",
    "V", "MA", "GOOGL", "COST", "WMT",
    "UNH", "JNJ", "WFC", "CRM", "ADBE",
    "AMD", "PYPL", "BA", "MMM", "UPS",
    "VZ", "T", "MRK", "ABBV", "ABT",
    "NOW", "CL", "IBM", "CSCO", "HON",
    "MU", "QCOM", "UBER", "MRNA", "SQ",
    "NFLX", "CMG", "GM", "F", "BAC",
    "C", "MS", "AMAT", "SBUX", "DELL",
    "SNOW", "AVGO", "CCI", "DIA", "XBI",
    "XTL", "XRT", "KO", "PEP", "MDT",
    "BABA", "LMT", "ORCL", "TMUS", "ROKU",
    "COIN", "Z", "SHOP", "LYFT", "WDC",
    "PANW", "ZS", "CRM", "DDOG", "NET",
    "PLTR", "SYK", "DHR", "LLY", "ISRG",
    "TXN", "CRWD", "ZM", "PINS", "OKTA",
    "NOC", "GD", "RTX", "BAESY", "TXT", 
    "NVMI", "NVST", "QQQM", "ADI", "FTNT", 
    "OXY", "XOM", "EOG", "HAL", "SLB", 
    "PM", "CLX", "STZ", "HSY", "MKC", 
    "USB", "TROW", "PRU", "MET", "CME", 
    "BDX", "ZBH", "BMRN", "BIIB", "ILMN", 
    "O", "SPG", "AMT", "PLD", "EQIX"
    ]

def fetchTickersData(ticker):
    """
    Fetch and prepare data for a single ticker
    """
    try:
        # Print message to indicate fetching data
        #print(f"Fetching data for {ticker}...")
        
        # Fetch data for the current ticker
        stockData = fetchStockData(ticker, period="5y")
        
        # Calculate technical indicators for the fetched data
        stockData = CalculateData(stockData)
        
        # Add ticker column for reference
        stockData["Ticker"] = ticker
        
        return stockData
    except Exception as e:
        # Print error message if fetching data fails
        print(f"Error fetching data for {ticker}: {e}")
        return None
    
def prepareDataParallel(tickers, period="5y") -> pd.DataFrame:
    """
    Fetch and prepare data for multiple tickers in parallel.
    Parameters:
        tickers (list): List of ticker symbols.
        period (str): Data period to fetch (default: "5y").
    Returns:
        pd.DataFrame: Combined DataFrame containing data for all tickers.
    """
    print(f"Fetching data for {len(tickers)} tickers...")
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetchTickersData, tickers)
    allData = [data for data in results if data is not None]
    if not allData:
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
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Perform cross-validation
    scores = cross_val_score(model, X, Y, scoring=scorer, cv=cv)
    avg_mse = -np.mean(scores)
    
    print(f"Cross-Validated MSE (cv={cv}): {avg_mse:.4f}")
    return avg_mse
    
    
def trainAndEvaluate(data, predictionDays=5, testSize=0.2, cv=5):
    """
    Train and evaluate a linear regression model on the given data.
    parameters: 
        data: pd.DataFrame containing stock data.
        predictionDays: int number of days ahead to predict.
        testSize: float proportion of data reserved for testing.
        cv: int number of cross-validation folds.
    Returns:
        model: Trained model.
        scalar: Scaler object used for feature scaling.
        featureNames: List of feature names.
    """
    print("Preparing data for training...")
    dataPreparator = DataPreparator()
    preparedData = dataPreparator.prepareForTrain(data, predictionDays, testSize)

    XTrain = preparedData["XTrain"]
    XTest = preparedData["XTest"]
    YTrain = preparedData["YTrain"]
    YTest = preparedData["YTest"]
    featureNames = preparedData["featureNames"]

    model = LinearRegression().fit(XTrain, YTrain)
    YPred = model.predict(XTest)
    
    print("Training and evaluation complete. Model metrics: ")
    print("\nFeature Importance (Linear Regression):")
    importance = sorted(
        zip(featureNames, model.coef_),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feature, coef in importance:
        print(f"{feature}: {coef:.4f}")
    print(f"\nMSE: {mean_squared_error(YTest, YPred):.4f}")
    print(f"MAE: {mean_absolute_error(YTest, YPred):.4f}")
    print(f"R^2: {r2_score(YTest, YPred):.4f}")
    crossValidateModel(XTrain, YTrain, model=model, cv=cv)

    return model, preparedData["scalar"], featureNames

def main():
    data = prepareDataParallel(TRAINING_TICKERS, period="5y")
    model, scalar, featureNames = trainAndEvaluate(data, predictionDays=5, testSize=0.2, cv=5)
    joblib.dump(model, "models/linear_regression_model.pkl")
    joblib.dump(scalar, "models/scalar.pkl")
    joblib.dump(featureNames, "models/feature_names.pkl")
    print("Model and assets saved successfully.")

if __name__ == "__main__":
    main()