"""Run controlled experiments for the current raw-return training approach.

This script is for research only. It intentionally reuses the current training
pipeline while allowing small parameter and prediction-horizon overrides.
"""

from copy import deepcopy

from src.config import XG_PARAMS_CLASSIFIER, XG_PARAMS_REGRESSOR
from src.inference.predictor import predict_price
from src.train.trainer import prepare_data_parallel, train_models


TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "UNH",
    "XOM",
    "SPY",
]
PERIOD = "5y"
SELECTED_EXPERIMENT = "baseline_5d"


def build_experiments() -> dict:
    baseline_classifier = deepcopy(XG_PARAMS_CLASSIFIER)
    baseline_regressor = deepcopy(XG_PARAMS_REGRESSOR)

    regressor_looser = deepcopy(XG_PARAMS_REGRESSOR)
    regressor_looser.update(
        {
            "max_depth": 4,
            "min_child_weight": 2,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        }
    )

    classifier_looser = deepcopy(XG_PARAMS_CLASSIFIER)
    classifier_looser.update(
        {
            "scale_pos_weight": 1.0,
            "min_child_weight": 2,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        }
    )

    return {
        "baseline_5d": {
            "prediction_days": 5,
            "classifier_params": baseline_classifier,
            "regressor_params": baseline_regressor,
        },
        "regressor_looser_5d": {
            "prediction_days": 5,
            "classifier_params": baseline_classifier,
            "regressor_params": regressor_looser,
        },
        "classifier_looser_5d": {
            "prediction_days": 5,
            "classifier_params": classifier_looser,
            "regressor_params": baseline_regressor,
        },
        "baseline_10d": {
            "prediction_days": 10,
            "classifier_params": baseline_classifier,
            "regressor_params": baseline_regressor,
        },
    }


def main() -> None:
    experiments = build_experiments()

    if SELECTED_EXPERIMENT not in experiments:
        raise ValueError(
            f"Unknown experiment {SELECTED_EXPERIMENT!r}. "
            f"Available experiments: {sorted(experiments)}"
        )

    experiment = experiments[SELECTED_EXPERIMENT]

    print(f"=== Running experiment: {SELECTED_EXPERIMENT} ===")
    print(f"Prediction days: {experiment['prediction_days']}")
    print(f"Classifier params: {experiment['classifier_params']}")
    print(f"Regressor params: {experiment['regressor_params']}")

    data = prepare_data_parallel(TICKERS, period=PERIOD)
    train_models(data, **experiment)

    print("=== Sample predictions after training ===")
    for ticker in ["AAPL", "MSFT", "ARM", "XSD", "ROK", "VCIT", "SPY"]:
        print(ticker, predict_price(ticker))


if __name__ == "__main__":
    main()
