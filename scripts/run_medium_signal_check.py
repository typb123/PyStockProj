"""Small research smoke check for training and prediction.

This script trains on a fixed 10-ticker universe and prints one AAPL prediction
using the saved models. It is useful for a quick manual sanity check, but it is
not the main experiment runner.

For the current SPY-relative research workflow, use:

    python -m src.train.trainer --all-horizons
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
