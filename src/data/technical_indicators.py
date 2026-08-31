"""Technical indicator feature generation for one ticker's chronological history.

This module does not group by ticker internally. Call calculate_data() on one
ticker at a time so rolling windows, shifts, and cumulative indicators do not
bleed across ticker boundaries.
"""

import pandas as pd
import numpy as np
from typing import List


def calculate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for one ticker's ordered OHLCV history.

    Rolling and shifted indicators use the input row order directly. Some early
    rows intentionally contain NaNs until enough lookback history exists;
    DataPreparator later drops rows missing required features or targets.

    Indicators:
        - Moving Averages (5, 10, 20 day SMA)
        - Daily Return
        - Trailing Momentum Returns
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
        data (pd.DataFrame): Stock DataFrame with OHLCV columns.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.

    Raises:
        KeyError: If required columns are missing.
    """
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    missing_cols = required_columns - set(data.columns)
    if missing_cols:
        raise KeyError(f"Input DataFrame must contain {missing_cols} columns")

    df = data.copy()

    # --- Indicator Functions ---
    def calculate_price_relative_features(df: pd.DataFrame) -> pd.DataFrame:
        """Express same-session OHLC information relative to the close."""
        df['open_to_close'] = df['Open'] / df['Close'] - 1
        df['high_to_close'] = df['High'] / df['Close'] - 1
        df['low_to_close'] = df['Low'] / df['Close'] - 1
        return df

    def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Calculate scale-invariant Simple Moving Average distances."""
        for window in windows:
            average = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'sma_{window}_to_close'] = average / df['Close'] - 1
        return df

    def calculate_returns_and_volatility(df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
        """Calculate daily returns and rolling volatility."""
        # pct_change creates an expected first-row NaN for the missing prior close.
        df['dailyReturn'] = df['Close'].pct_change()
        df['volatility'] = df['dailyReturn'].rolling(window=vol_window, min_periods=1).std()
        return df

    def calculate_trailing_momentum(
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50],
    ) -> pd.DataFrame:
        """Calculate past-looking trailing returns for momentum baselines."""
        for window in windows:
            df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
        return df

    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index with division-by-zero protection."""
        # min_periods delays RSI until there is enough history for a meaningful signal.
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=5).mean()
        rs = gain / (loss + 1e-10)  # Small epsilon to avoid infinity
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD, signal line, and histogram relative to the close."""
        # Early MACD values remain NaN until the EMA lookback has enough rows.
        ema_fast = df['Close'].ewm(span=fast, min_periods=5, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, min_periods=5, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - signal_line
        df['macd_to_close'] = macd / df['Close']
        df['signal_line_to_close'] = signal_line / df['Close']
        df['macd_histogram_to_close'] = macd_histogram / df['Close']
        return df

    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        # OBV is cumulative over the provided single-ticker order.
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return df

    def calculate_vma(df: pd.DataFrame, windows: List[int] = [10, 20]) -> pd.DataFrame:
        """Calculate Volume Moving Average for specified windows."""
        for window in windows:
            df[f'vma_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
        return df

    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate scale-invariant Bollinger band distances."""
        # Bollinger bands intentionally start after a minimum lookback window.
        middle = df['Close'].rolling(window=period, min_periods=5).mean()
        std = df['Close'].rolling(window=period, min_periods=5).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        df['bb_middle_to_close'] = middle / df['Close'] - 1
        df['bb_upper_to_close'] = upper / df['Close'] - 1
        df['bb_lower_to_close'] = lower / df['Close'] - 1
        df['bb_std_to_close'] = std / df['Close']
        return df

    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range relative to the close."""
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
        atr = true_range.rolling(window=period, min_periods=1).mean().fillna(0)
        df['atr_to_close'] = atr / df['Close']

        return df

    def calculate_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator (%K and %D)."""
        low_min = df['Low'].rolling(window=period, min_periods=1).min()
        high_max = df['High'].rolling(window=period, min_periods=1).max()
        df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth_k, min_periods=1).mean()
        return df

    def calculate_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate past-looking Ichimoku-style features relative to the close."""
        tenkan_sen = (df['High'].rolling(window=9, min_periods=5).max() +
                      df['Low'].rolling(window=9, min_periods=5).min()) / 2
        kijun_sen = (df['High'].rolling(window=26, min_periods=5).max() +
                     df['Low'].rolling(window=26, min_periods=5).min()) / 2
        # Positive shifts move historical cloud values forward; they do not expose future rows.
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((df['High'].rolling(window=52, min_periods=5).max() +
                         df['Low'].rolling(window=52, min_periods=5).min()) / 2).shift(26)
        # Chikou-inspired ML features use a past close, not the future-shifted charting span.
        chikou_lag_close = df['Close'].shift(26)
        df['tenkan_sen_to_close'] = tenkan_sen / df['Close'] - 1
        df['kijun_sen_to_close'] = kijun_sen / df['Close'] - 1
        df['senkou_span_a_to_close'] = senkou_span_a / df['Close'] - 1
        df['senkou_span_b_to_close'] = senkou_span_b / df['Close'] - 1
        df['chikou_lag_close_26_to_close'] = chikou_lag_close / df['Close'] - 1
        df['chikou_return_26'] = (df['Close'] - df['Close'].shift(26)) / df['Close'].shift(26)
        df['chikou_above_lag_26'] = (df['Close'] > df['Close'].shift(26)).astype(int)
        return df

    # --- Compute All Indicators ---
    df = calculate_price_relative_features(df)
    df = calculate_moving_averages(df)
    df = calculate_returns_and_volatility(df)
    df = calculate_trailing_momentum(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_vma(df)
    df = calculate_obv(df)
    df = calculate_ichimoku_cloud(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)
    df = calculate_stochastic(df)


    return df
