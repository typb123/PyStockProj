from src.config import PREDICTION_DAYS
from src.data.data_fetch import fetch_stock_data
from src.inference.predictor import predict_price
from src.utils.helpers import get_prediction_date
import yfinance as yf

# Constants
WELCOME_MESSAGE = "\n\nWelcome to my stock prediction program.\n"
INVALID = "\nInvalid option! Please try again.\n"
GOODBYE = "Exiting the program. Thank you and goodbye!"


def get_stock_info(ticker):
    """Fetch and display my stock information"""
    try:
        one_month_stock_data = fetch_stock_data(ticker, period="1mo")
        one_year_stock_data = fetch_stock_data(ticker, period="1y")

        if one_month_stock_data.empty or one_year_stock_data.empty:
            print(
                f"No price data found for {ticker}. Check the ticker symbol and try again."
            )
            return

        # Calculate 1-month high and low
        one_month_high = one_month_stock_data["High"].max()
        one_month_low = one_month_stock_data["Low"].min()

        # Calculate 52-week high and low
        wk52_high = one_year_stock_data["High"].max()
        wk52_low = one_year_stock_data["Low"].min()

        # Fetch current price using yfinance
        ticker_info = yf.Ticker(ticker).info
        current_price = ticker_info.get("currentPrice", None)

        prediction_date = get_prediction_date(
            one_year_stock_data,
            trading_days_ahead=PREDICTION_DAYS,
        )

        # Fetch prediction
        prediction = predict_price(ticker)

        # ---- Output Results ----
        print(f"\nStock Information for {ticker}:")
        print(
            f"Current Price: ${current_price:.2f}"
            if current_price
            else "Current Price: Unavailable"
        )
        print(f"1-Month High: ${one_month_high:.2f}")
        print(f"1-Month Low: ${one_month_low:.2f}")
        print(f"52-Week High: ${wk52_high:.2f}")
        print(f"52-Week Low: ${wk52_low:.2f}")

        if isinstance(prediction, dict) and "error" not in prediction:
            print(
                f"\nPrediction for {ticker} on {prediction_date} "
                f"({PREDICTION_DAYS} trading days ahead):"
            )
            print(
                "  Predicted Excess Return vs SPY: "
                f"{prediction['predicted_excess_return']:.4%}"
            )
            print(f"  Signal: {prediction['signal']}")
            print("\n")
        else:
            print(f"Prediction Error: {prediction.get('error', 'Unavailable')}")

    except Exception as e:
        print(f"An error occurred: {e}")


def main_menu():
    """Main Menu for my console application"""
    print(WELCOME_MESSAGE)

    while True:
        user_input = input("Enter a ticker symbol, or q to quit: ").strip()

        if user_input.lower() in {"q", "quit", "exit", "2"}:
            print(GOODBYE)
            break

        if not user_input:
            print(INVALID)
            continue

        get_stock_info(user_input.upper())


if __name__ == "__main__":
    main_menu()
