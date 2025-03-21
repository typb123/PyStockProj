from helper_functions import fetchStockData, getFuturePredictionDate, predictPrice
import yfinance as yf

#Constants 
WELCOME_MESSAGE = "\n\nWelcome to my stock prediction program.\n"
INVALID = "\nInvalid option! Please try again.\n"
GOODBYE = "Exiting the program. Thank you and goodbye!"

def getStockInfo(ticker):
    """Fetch and display my stock information"""
    try:
        one_month_stock_data = fetchStockData(ticker, period="1mo")
        one_year_stock_data = fetchStockData(ticker, period="1y")

        # Calculate 1-month high and low
        one_month_high = one_month_stock_data['High'].max()
        one_month_low = one_month_stock_data['Low'].min()

        # Calculate 52-week high and low
        wk52_high = one_year_stock_data['High'].max()
        wk52_low = one_year_stock_data['Low'].min()

        # Fetch current price using yfinance
        ticker_info = yf.Ticker(ticker).info
        current_price = ticker_info.get('currentPrice', None)

        # Get future prediction date (5 trading days ahead)
        prediction_date = getFuturePredictionDate(one_year_stock_data, tradingDaysAhead=5)

        # Fetch prediction
        prediction = predictPrice(ticker)

        # ---- Output Results ----
        print(f"\nStock Information for {ticker}:")
        print(f"Current Price: ${current_price:.2f}" if current_price else "Current Price: Unavailable")
        print(f"1-Month High: ${one_month_high:.2f}")
        print(f"1-Month Low: ${one_month_low:.2f}")
        print(f"52-Week High: ${wk52_high:.2f}")
        print(f"52-Week Low: ${wk52_low:.2f}")

        if isinstance(prediction, dict) and 'error' not in prediction:
            print(f"\nPrediction for {ticker} on {prediction_date}:")
            print(f"  Direction: {prediction['direction']}")
            print(f"  Predicted Return: {prediction['predicted_return']:.4%}")
            print(f"  Expected Price: ${prediction['expected_price']:.2f}")
            print("\n")
        else:
            print(f"Prediction Error: {prediction.get('error', 'Unavailable')}")

    except Exception as e:
        print(f"An error occurred: {e}")
        
        

def mainMenu():
    """Main Menu for my console application"""
    print(WELCOME_MESSAGE)

    while True:
        print("1. Enter Ticker Symbol")
        print("2. Exit")

        try:
            #Take user input and validate it
            userChoice = int(input("Please enter an option: ").strip())
        except ValueError:
            print(INVALID)
            continue

        if userChoice == 1:
            ticker = input("Enter a ticker symbol (e.g AAPL): ").strip().upper()
            if ticker:
                getStockInfo(ticker)
            else:
                print(INVALID)
                continue
        elif userChoice == 2:
            print(GOODBYE)
            break
        else:
            print(INVALID) 



if __name__ == "__main__":
    mainMenu()











    


