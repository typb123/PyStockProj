"""Run a medium-size research smoke check for training and prediction.

This script is a repeatable research helper, not a unit test. It trains on a
fixed 10-ticker universe, then prints one AAPL prediction using the saved models.
"""

from src.inference.predictor import predict_price
from src.train.trainer import prepare_data_parallel, train_models


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "UNH", "XOM", "SPY"]
PERIOD = "5y"


def main() -> None:
    print("=== Medium signal check: training 10 tickers over 5y ===")
    data = prepare_data_parallel(TICKERS, period=PERIOD)
    train_models(data)

    print("=== AAPL prediction after training ===")
    print(predict_price("AAPL"))


if __name__ == "__main__":
    main()
