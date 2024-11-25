import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_prep import DataPreparator
from technical_indicators import CalculateData
from main import fetchStockData

#These are the 12 tickers I am currently training on
TRAINING_TICKERS = ["SPY", "XLU", "XLE", "XLV", "VOOG", 
                    "VOOV", "VB", "TSLA", "LAD", "JPM", 
                    "AAPL", "SCHW", "NVDA", "META", "GE",
                    "INTC", "BX", "KR", "FDX", "MSFT", 
                    "CVX", "PSX", "PG", "PFE", "AMGN", 
                    "BRK-A", "BRK-B", "RCL", "AMZN", "HD", 
                    "DUK", "NEE", "GS", "CAT", "MCD"]

def fetchAndPrepareData(tickers, period="5y") -> pd.DataFrame:
    """Here we fetch and combine historical data for my specified ticker symbols"""
    allData = []
    for ticker in tickers:
        try:
            # Fetch data for the current ticker
            print(f"Fetching data for {ticker}...")
            stockData = fetchStockData(ticker, period=period)
            
            # Calculate technical indicators for the fetched data
            stockData = CalculateData(stockData)
            
            # Add ticker column for reference
            stockData["Ticker"] = ticker
            
            # Append the processed data to the list
            allData.append(stockData)
        except Exception as e:
            # Print error message if fetching data fails
            print(f"Error fetching data for {ticker}: {e}")
    
    # Combine all fetched data into a single DataFrame
    combineData = pd.concat(allData, ignore_index=True)
    
    # Print the shape of the combined dataset
    print(f"Combined dataset shape: {combineData.shape}")
    
    return combineData

def trainLinearRegression(XTrain, YTrain) -> LinearRegression:
    """Train a linear regression model on the given training data"""
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(XTrain, YTrain)
    print("Model training complete.")
    return model

def evaluateModel(model, XTest, YTest):
    """Evaluate the trained model on the test data"""
    YPred = model.predict(XTest)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(YTest, YPred)
    mae = mean_absolute_error(YTest, YPred)
    r2 = r2_score(YTest, YPred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

def analyzeLinearRegressionFeatureImportance(model, featureNames):
    """Analyze feature importance for linear regression."""
    print("\nFeature Importance (Linear Regression):")
    for feature, coef in zip(featureNames, model.coef_):
        print(f"{feature}: {coef:.4f}")

def main():
    """Main function to train and save my linear regression model."""
    #Fetch and process data
    data = fetchAndPrepareData(TRAINING_TICKERS, period="5y")

    #Prepare data for training
    dataPreparator = DataPreparator()
    preparedData = dataPreparator.prepareForTrain(data, predictionDays=14, testSize = 0.2)

    XTrain = preparedData['XTrain']
    XTest = preparedData['XTest']
    YTrain = preparedData['YTrain']
    YTest = preparedData['YTest']

    #Train Linear Regression model
    model = trainLinearRegression(XTrain, YTrain)

     # Analyze feature importance
    analyzeLinearRegressionFeatureImportance(model, preparedData['featureNames'])

    #Evaluate the model
    evaluateModel(model, XTest, YTest)

    #Save the model and scalar 
    joblib.dump(model, "models/linear_regression_model.pkl")
    joblib.dump(dataPreparator.scalar, "models/scalar.pkl")
    print("Model and scalar saved successfully")

if __name__ == "__main__":
    main()
