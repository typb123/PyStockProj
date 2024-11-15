from technical_indicators import CalculateData
from main import fetchStockData


def testCalculateData(ticker="AAPL", period="1y"):
    """
    Test the CalculateData function by fetching stock data for a given ticker,
    applying the CalculateData function, and displaying the last 10 rows
    of the processed DataFrame.

    Parameters:
        ticker (str): The stock ticker symbol to fetch and test (default is "AAPL").
        period (str): The time period for historical data (default is "1y" for 1 year).
    
    Returns:
        pd.DataFrame: The processed DataFrame with added technical indicators.
    """
    try:
        # Step 1: Fetch stock data
        data = fetchStockData(ticker, period=period)
        print(f"Fetched data for {ticker} ({period}) - {len(data)} rows")

        # Step 2: Apply CalculateData to add indicators
        processed_data = CalculateData(data)
        print(f"Calculated data with technical indicators - {len(processed_data)} rows")

        # Step 3: Display the last 10 rows to verify indicators
        print("\nLast 10 rows with technical indicators:")
        print(processed_data.tail(10))
        
        return processed_data  # Return the DataFrame for further testing if needed

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the test if this script is executed
if __name__ == "__main__":
    testCalculateData()