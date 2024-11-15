import yfinance as yf


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







    


