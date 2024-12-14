from helper_functions import fetchStockData, getFuturePredictionDate, predictPrice, yf

#Constants 
WELCOME_MESSAGE = "\n\nWelcome to my stock prediction program.\n"
INVALID = "\nInvalid option! Please try again.\n"
GOODBYE = "Exiting the program. Thank you and goodbye!"

def getStockInfo(ticker):
    """Fetch and display my stock information"""
    try:
        oneMonthStockData = fetchStockData(ticker, period="1mo")
        oneYearStockData = fetchStockData(ticker, period="1y")

        # Calculate 1-month high and low
        oneMonthHigh = oneMonthStockData['High'].max()
        oneMonthLow = oneMonthStockData['Low'].min()
        # Calculate 52-week high and low
        wk52High = oneYearStockData['High'].max()
        wk52Low = oneYearStockData['Low'].min()
        # Fetch Current Price
        tickerInfo = yf.Ticker(ticker).info
        currentPrice = tickerInfo.get('currentPrice', None)
        #Get date of 5 trading days
        predictionDate = getFuturePredictionDate(oneYearStockData, tradingDaysAhead=5)

        # Fetch prediction
        prediction = predictPrice(ticker)
        
        # Output results
        print(f"\nStock Information for {ticker}:")
        if currentPrice is not None:
            print(f"Current Price: ${currentPrice}")
        else:
            print(f"Current Price: Unavailable")
        print(f"1-Month High: ${oneMonthHigh:.2f}")
        print(f"1-Month Low: ${oneMonthLow:.2f}")
        print(f"52-Week High: ${wk52High:.2f}")
        print(f"52-Week Low: ${wk52Low:.2f}")
        if isinstance(prediction, dict):
            print(f"Prediction for {ticker} on {predictionDate}:")
            print(f"  Direction: {prediction['direction']}")
            print(f"  Predicted Return: {prediction['predictedReturn']:.4%}")
            print(f"  Expected Price: ${prediction['expectedPrice']:.2f}")
            print("\n")
        else:
            print(prediction)

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











    


