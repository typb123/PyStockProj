import pandas as pd
import numpy as np
from typing import List


def CalculateData(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add some technical indicators to my stock data for model input
    
    -Moving average (5, 10, 20 day)
    -Daily Return 
    -Rolling volatility 
    -Relative Strength Index (RSI)
    -Moving Average Convergence Divergence (MACD)

    Parameters: data(pd.DataFrame): Stock DataFrame with "Close" price column. Raises error otherwise

    Return: pd.DataFrame with added technical indicators. 
    """

    #Validate input
    if 'Close' not in data.columns:
        raise KeyError("Input DataFrame must contain 'Close' price column")
    
    #Create a copy
    result = data.copy()

    def calculateMovingAverages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 5, 10, 20 day moving averages"""
        windows = [5, 10, 20]
        for window in windows:
            df[f'{window}_day_avg'] = df['Close'].rolling(window=window).mean()
        return df
    
    def calculateReturnsAndVolatility(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns and rolling volatility"""
        df['dailyReturn'] = df['Close'].pct_change()
        df['volatility'] = df['dailyReturn'].rolling(window = 10).std()
        return df
    
    def calculateRSI(df: pd.DataFrame, period = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        deltaPrice = df['Close'].diff(1)
        gainPrice = (deltaPrice.where(deltaPrice > 0, 0)).rolling(window=period).mean()
        lossPrice = (-deltaPrice.where(deltaPrice < 0, 0)).rolling(window=period).mean()

        #Division by zero
        relativeStrength = np.where(lossPrice == 0, 0, gainPrice / lossPrice)
        df['rsi'] = 100 - (100 / (1 + relativeStrength))
        return df
    
    def calculateMACD(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD, signal line, and histogram"""
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()

        df['macd'] = ema12 - ema26
        df['signalLine'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdHistogram'] = df['macd'] - df['signalLine']
        return df
    
    def calculateOBV(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return df
    
    def calculateVMA(df: pd.DataFrame, windows=[10, 20]) -> pd.DataFrame:
        """Calculate Volume Moving Average for the given windows"""
        for window in windows:
            df[f'vma_{window}'] = df['Volume'].rolling(window=window).mean()
        return df
    
    def calculateIchimokuCloud(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components and add them to the DataFrame."""
        # Validate input
        required_columns = ['High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Input DataFrame must contain '{col}' column.")
    
        # Calculate Tenkan-sen (Conversion Line)
        df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2

        # Calculate Kijun-sen (Base Line)
        df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

        # Calculate Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

        # Calculate Senkou Span B (Leading Span B)
        df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)

        # Calculate Chikou Span (Lagging Span)
        df['chikou_span'] = df['Close'].shift(-26)

        return df

    

    #Calculate all indicators 
    result = calculateMovingAverages(result)
    result = calculateReturnsAndVolatility(result)
    result = calculateRSI(result)
    result = calculateMACD(result)
    result = calculateVMA(result)
    result = calculateOBV(result) 
    result = calculateIchimokuCloud(result)

    result = result.dropna()

    return result


