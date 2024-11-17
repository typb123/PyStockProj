import pandas as pd
from data_prep import DataPreparator
from technical_indicators import CalculateData
from main import fetchStockData


def test_pipeline_with_yfinance():
    # Initialize components
    data_preparator = DataPreparator()

    # Step 1: Fetch stock data
    ticker = "AAPL"  # Replace with the desired stock ticker
    period = "1y"  # Test with a smaller period for faster results
    print("Fetching stock data...")
    df = fetchStockData(ticker, period)
    print(f"Fetched data shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")

    # Step 2: Add technical indicators
    print("\nAdding technical indicators...")
    df_with_indicators = CalculateData(df)
    print(f"Data shape after adding indicators: {df_with_indicators.shape}")
    print(f"Sample data with indicators:\n{df_with_indicators.head()}")

    # Step 3: Prepare data for training
    print("\nPreparing data for training...")
    prepared_data = data_preparator.prepareForTrain(df_with_indicators, predictionDays=14, testSize=0.2)

    XTrain = prepared_data['XTrain']
    XTest = prepared_data['XTest']
    YTrain = prepared_data['YTrain']
    YTest = prepared_data['YTest']
    featureNames = prepared_data['featureNames']

    print(f"Feature Names: {featureNames}")
    print(f"XTrain shape: {XTrain.shape}, XTest shape: {XTest.shape}")
    print(f"YTrain shape: {YTrain.shape}, YTest shape: {YTest.shape}")
    print(f"Sample XTrain:\n{XTrain[:5]}")
    print(f"Sample YTrain:\n{YTrain[:5]}")

    # Step 4: Check scaling
    print("\nVerifying scaling...")
    print(f"XTrain mean: {XTrain.mean(axis=0)}")
    print(f"XTrain std deviation: {XTrain.std(axis=0)}")

    # Additional checks
    print("\nChecking for NaN values...")
    print(f"XTrain NaNs: {pd.isnull(XTrain).sum()}")
    print(f"XTest NaNs: {pd.isnull(XTest).sum()}")

if __name__ == "__main__":
    test_pipeline_with_yfinance()
