import yfinance as yf


#Contants 
WELCOME_MESSAGE = "\n\nWelcome to my stock prediction program.\n"
INVALID = "\nInvalid option! Please try again.\n"
GOODBYE = "Exiting the program. Thank you and goodbye!"


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

def predictPrice(ticker):
    """This is a placeholder"""

    return "Unavailable at this time"



def mainMenu():
    """Main Menu for my console application"""
    print(WELCOME_MESSAGE)

    while True:
        print("1. Enter Ticker Symbol")
        print("2. Exit")

        try:
            #Take user input and validate it
            userChoice = int(input("Please enter and option: ").strip())
        except ValueError:
            print(INVALID)
            continue

        if userChoice == 1:
            ticker = input("Enter a ticker symbol (e.g AAPL): ").strip().upper()

            try:
                oneMonthStockData = fetchStockData(ticker, period="1mo")
                oneYearStockData = fetchStockData(ticker, period="1y")

                #Calculate 1-month high and low
                oneMonthHigh = oneMonthStockData['High'].max()
                oneMonthLow = oneMonthStockData['Low'].min()
                #Calculate 52wk high and low
                wk52High = oneYearStockData['High'].max()
                wk52Low = oneYearStockData['Low'].min()

                #Fetch prediction
                prediction = predictPrice(ticker)

                #Output results
                print(f"\nStock Information for {ticker}:")
                print(f"1-Month High: ${oneMonthHigh:.2f}")
                print(f"1-Month Low: ${oneMonthLow:.2f}")
                print(f"52-Week High: ${wk52High:.2f}")
                print(f"52-Week Low: ${wk52Low:.2f}")
                print(f"14-Day Price Prediction: {prediction}")
                print("\n")

            except Exception as e:
                print(f"An error occurred: {e}")

        elif userChoice == 2:
            print(GOODBYE)
            break
        else:
            print(INVALID) 



#if __name__ == "__main__":
 #   mainMenu()











    


