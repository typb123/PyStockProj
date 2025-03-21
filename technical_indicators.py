import pandas as pd
import numpy as np
from typing import List, Optional


def CalculateData(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add technical indicators to stock data for model input.

    Indicators:
        - Moving Averages (5, 10, 20 day SMA)
        - Daily Return
        - Rolling Volatility
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD)
        - On-Balance Volume (OBV)
        - Volume Moving Average (VMA)
        - Ichimoku Cloud components
        - Bollinger Bands
        - Average True Range (ATR)
        - Stochastic Oscillator

    Parameters:
        data (pd.DataFrame): Stock DataFrame with 'Close', 'High', 'Low', 'Volume' columns.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.

    Raises:
        KeyError: If required columns are missing.
    """
    # Validate input
    required_columns = {'Close', 'High', 'Low', 'Volume'}
    missing_cols = required_columns - set(data.columns)
    if missing_cols:
        raise KeyError(f"Input DataFrame must contain {missing_cols} columns")

    # Create a working copy
    df = data.copy()

    # --- Indicator Functions ---
    def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Calculate Simple Moving Averages for specified windows."""
        for window in windows:
            df[f'{window}_day_avg'] = df['Close'].rolling(window=window, min_periods=1).mean()
        return df

    def calculate_returns_and_volatility(df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
        """Calculate daily returns and rolling volatility."""
        df['dailyReturn'] = df['Close'].pct_change()
        df['volatility'] = df['dailyReturn'].rolling(window=vol_window, min_periods=1).std()
        return df

    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index with division-by-zero protection."""
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=5).mean()
        rs = gain / (loss + 1e-10)  # Small epsilon to avoid infinity
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD, signal line, and histogram."""
        ema_fast = df['Close'].ewm(span=fast, min_periods=5, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, min_periods=5, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['signalLine'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macdHistogram'] = df['macd'] - df['signalLine']
        return df

    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return df

    def calculate_vma(df: pd.DataFrame, windows: List[int] = [10, 20]) -> pd.DataFrame:
        """Calculate Volume Moving Average for specified windows."""
        for window in windows:
            df[f'vma_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
        return df

    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df['BB_Middle'] = df['Close'].rolling(window=period, min_periods=5).mean()
        df['BB_Std'] = df['Close'].rolling(window=period, min_periods=5).std()
        df['BB_Upper'] = df['BB_Middle'] + std_dev * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - std_dev * df['BB_Std']
        return df

    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR)."""
        # Calculate True Range
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        # Replace NaNs in high_close_prev and low_close_prev with 0
        high_close_prev = high_close_prev.fillna(0)
        low_close_prev = low_close_prev.fillna(0)
        
        # Find the greatest of the three
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR with better handling of NaN values
        df['ATR'] = true_range.rolling(window=period, min_periods=1).mean().fillna(0)
        
        return df

    def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator (%K and %D)."""
        low_min = df['Low'].rolling(window=period, min_periods=1).min()
        high_max = df['High'].rolling(window=period, min_periods=1).max()
        df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth_k, min_periods=1).mean()
        return df

    def calculate_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components."""
        df['tenkan_sen'] = (df['High'].rolling(window=9, min_periods=5).max() + 
                            df['Low'].rolling(window=9, min_periods=5).min()) / 2
        df['kijun_sen'] = (df['High'].rolling(window=26, min_periods=5).max() + 
                           df['Low'].rolling(window=26, min_periods=5).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['High'].rolling(window=52, min_periods=5).max() + 
                               df['Low'].rolling(window=52, min_periods=5).min()) / 2).shift(26)
        df['chikou_span'] = df['Close'].shift(-26)
        return df

    # --- Compute All Indicators ---
    df = calculate_moving_averages(df)
    df = calculate_returns_and_volatility(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_vma(df)
    df = calculate_obv(df)
    df = calculate_ichimoku_cloud(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)
    df = calculate_stochastic(df)


    return df
