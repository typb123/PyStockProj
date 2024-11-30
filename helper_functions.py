import yfinance as yf
from pandas.tseries.offsets import BDay
import pandas as pd
import joblib
from technical_indicators import CalculateData

def fetchStockData(ticker, period="5y"):
    """
    Fetch historical stock data
    
    Parameters: ticker (str) and period(str)
    
    Return: pd.DataFrame containing stock data.
    """
    stockData = yf.Ticker(ticker)
    histData = stockData.history(period=period)

    #Error
    if histData.empty:
        raise ValueError(f"No data found for ticker symbol '{ticker}'. Please check the symbol and try again.")
    
    return histData

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
        # Ensure the index is a DatetimeIndex
        if not isinstance(stockData.index, pd.DatetimeIndex):
            raise ValueError("The stockData index must be a DatetimeIndex.")
        
        # Get the most recent trading date
        lastTradingDate = stockData.index.max()

        # Project the future date using business days
        futureDate = lastTradingDate + BDay(tradingDaysAhead)

        # Convert to MM/DD/YYYY format
        return futureDate.strftime('%m/%d/%Y')
    
    except Exception as e:
        print(f"Error in getFuturePredictionDate: {e}")
        raise


def predictPrice(ticker):
    """
    Predict price movement using my linear regression model
    
    Parameters:
        ticker (str)
        
    Returns:
        dict: Contains the predicted direction, return, and expected price
    """
    try:
        #Load trained model and scalar
        model = joblib.load("models/linear_regression_model.pkl")
        scalar = joblib.load("models/scalar.pkl")
        featureColumns = joblib.load("models/feature_names.pkl")

        #Fetch the most recent data(1 month)
        recentData = fetchStockData(ticker, period="6mo")

        #Calculate technical indicators
        processedData = CalculateData(recentData)

        #Extract the latest row for predictions
        latestFeatures = processedData.iloc[-1]#Last row
        X = latestFeatures[featureColumns].values.reshape(1,-1) #Reshape for model input


        #Scale the features
        XScaled = scalar.transform(X)

        #Prediction
        predictedReturn = model.predict(XScaled)[0]

        #Calculate expected price
        lastClosePrice = recentData['Close'].iloc[-1]
        expectedPrice = lastClosePrice * (1 + predictedReturn)

        #Direction 
        direction = "Up" if predictedReturn > 0 else "Down"

        return {
        "direction": direction,
        "predictedReturn": predictedReturn,
        "expectedPrice": expectedPrice
        }
    except Exception as e:
        print(f"An error occured in predictPrice: {e}")
        return "Prediction unavailable"
    
