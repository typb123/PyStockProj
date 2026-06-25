"""Run controlled experiments for the current raw-return training approach.

This script is for research only. It intentionally reuses the current training
pipeline while allowing small parameter and prediction-horizon overrides.
"""

import sys
import ast
import math
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
SELECTED_EXPERIMENT = "baseline_10d"
TRAINING_LOG = PROJECT_ROOT / "training.log"
VALIDATION_THRESHOLD_LABEL = "XGBoost Classifier Validation-Selected Threshold Report"
REGRESSOR_QUANTILE_LABEL = "XGBoost Regressor Predicted Return Quantile Report"
COMBINED_SIGNAL_LABEL = "XGBoost Combined Signal Report"


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
        "regressor_looser_10d": {
            "prediction_days": 10,
            "classifier_params": baseline_classifier,
            "regressor_params": regressor_looser,
        },
    }


def _extract_report_from_log(log_path: Path, label: str):
    if not log_path.exists():
        return None

    marker = f"{label}: "
    matching_line = None
    for line in log_path.read_text().splitlines():
        if marker in line:
            matching_line = line

    if matching_line is None:
        return None

    report_text = matching_line.split(marker, 1)[1]
    try:
        return ast.literal_eval(report_text.replace("nan", "None"))
    except (SyntaxError, ValueError):
        return None


def _format_percent(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unavailable"
    return f"{value:.2%}"


def _print_report_section(title, stats, count_key="selected_count"):
    print(f"\n{title}:")
    if not stats:
        print("- unavailable")
        return

    print(f"- selected count: {stats.get(count_key, 'unavailable')}")
    print(f"- avg actual return: {_format_percent(stats.get('avg_actual_return'))}")
    print(f"- precision: {_format_percent(stats.get('precision'))}")


def _print_experiment_summary(experiment_name, experiment, log_path=TRAINING_LOG):
    validation_threshold_report = _extract_report_from_log(
        log_path,
        VALIDATION_THRESHOLD_LABEL,
    )
    regressor_quantile_report = _extract_report_from_log(
        log_path,
        REGRESSOR_QUANTILE_LABEL,
    )
    combined_signal_report = _extract_report_from_log(log_path, COMBINED_SIGNAL_LABEL)

    validation_stats = (
        validation_threshold_report.get("selected_test_stats")
        if validation_threshold_report
        else None
    )
    regressor_top_20_stats = (
        regressor_quantile_report.get("top_20_pct")
        if regressor_quantile_report
        else None
    )

    print("\n=== Experiment Summary ===")
    print(f"Experiment: {experiment_name}")
    print(f"Prediction horizon: {experiment['prediction_days']} trading days")
    _print_report_section("Validation-selected threshold only", validation_stats)
    _print_report_section("Regressor top 20% only", regressor_top_20_stats, "count")
    _print_report_section("Combined signal", combined_signal_report)

    candidates = [
        ("validation threshold", validation_stats),
        ("regressor top 20%", regressor_top_20_stats),
        ("combined signal", combined_signal_report),
    ]
    available_candidates = [
        (name, stats["avg_actual_return"])
        for name, stats in candidates
        if stats and stats.get("avg_actual_return") is not None
    ]

    print("\nQuick read:")
    if not available_candidates:
        print("- Not enough report data to compare signals.")
        return

    best_name, best_return = max(available_candidates, key=lambda item: item[1])
    print(f"- Highest avg actual return: {best_name} ({_format_percent(best_return)})")


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
    _print_experiment_summary(SELECTED_EXPERIMENT, experiment)

    print("=== Sample predictions after training ===")
    for ticker in ["AAPL", "MSFT", "ARM", "XSD", "ROK", "VCIT", "SPY"]:
        print(ticker, predict_price(ticker))


if __name__ == "__main__":
    main()
