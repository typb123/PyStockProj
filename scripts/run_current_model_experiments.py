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

TICKER_UNIVERSES = {
    "small_10": [
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
    ],
    "core_25": [
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
        "V",
        "MA",
        "HD",
        "COST",
        "WMT",
        "PG",
        "JNJ",
        "LLY",
        "CVX",
        "CAT",
        "KO",
        "PEP",
        "AVGO",
        "QQQ",
        "IWM",
    ],
    "core_50": [
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
        "V",
        "MA",
        "HD",
        "COST",
        "WMT",
        "PG",
        "JNJ",
        "LLY",
        "CVX",
        "CAT",
        "KO",
        "PEP",
        "AVGO",
        "QQQ",
        "IWM",
        "BRK-B",
        "BAC",
        "WFC",
        "MS",
        "GS",
        "ABBV",
        "MRK",
        "PFE",
        "TMO",
        "ABT",
        "ORCL",
        "CRM",
        "ADBE",
        "AMD",
        "QCOM",
        "LIN",
        "HON",
        "GE",
        "DE",
        "UPS",
        "NEE",
        "DUK",
        "AMT",
        "PLD",
        "TLT",
    ],
    "broad_200": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "AVGO",
        "ORCL",
        "ADBE",
        "CRM",
        "AMD",
        "QCOM",
        "TXN",
        "INTC",
        "AMAT",
        "MU",
        "LRCX",
        "KLAC",
        "ADI",
        "PANW",
        "CRWD",
        "NOW",
        "SNOW",
        "DDOG",
        "NET",
        "FTNT",
        "ZS",
        "IBM",
        "CSCO",
        "SHOP",
        "UBER",
        "PYPL",
        "SQ",
        "NFLX",
        "ROKU",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "SCHW",
        "BLK",
        "AXP",
        "V",
        "MA",
        "COF",
        "USB",
        "PNC",
        "TFC",
        "CME",
        "ICE",
        "SPGI",
        "MCO",
        "BRK-B",
        "UNH",
        "JNJ",
        "LLY",
        "MRK",
        "ABBV",
        "PFE",
        "TMO",
        "ABT",
        "DHR",
        "ISRG",
        "SYK",
        "MDT",
        "BSX",
        "BMY",
        "AMGN",
        "GILD",
        "REGN",
        "VRTX",
        "BIIB",
        "BMRN",
        "CVS",
        "CI",
        "HUM",
        "ELV",
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "HAL",
        "OXY",
        "PSX",
        "VLO",
        "MPC",
        "KMI",
        "WMB",
        "NEE",
        "DUK",
        "SO",
        "D",
        "AEP",
        "EXC",
        "SRE",
        "PEG",
        "ED",
        "XEL",
        "HD",
        "LOW",
        "COST",
        "WMT",
        "TGT",
        "SBUX",
        "MCD",
        "NKE",
        "LULU",
        "TJX",
        "BKNG",
        "MAR",
        "HLT",
        "CMG",
        "ORLY",
        "AZO",
        "F",
        "GM",
        "RCL",
        "CCL",
        "PG",
        "KO",
        "PEP",
        "MDLZ",
        "CL",
        "KMB",
        "KHC",
        "GIS",
        "HSY",
        "MKC",
        "STZ",
        "PM",
        "MO",
        "TAP",
        "CAT",
        "DE",
        "GE",
        "HON",
        "RTX",
        "LMT",
        "NOC",
        "GD",
        "BA",
        "UPS",
        "FDX",
        "UNP",
        "CSX",
        "NSC",
        "ETN",
        "EMR",
        "MMM",
        "ITW",
        "PH",
        "ROK",
        "LIN",
        "SHW",
        "APD",
        "ECL",
        "FCX",
        "NEM",
        "DOW",
        "DD",
        "ALB",
        "MOS",
        "AMT",
        "PLD",
        "EQIX",
        "SPG",
        "O",
        "CCI",
        "PSA",
        "DLR",
        "WELL",
        "VICI",
        "TLT",
        "IEF",
        "LQD",
        "HYG",
        "GLD",
        "SLV",
    ],
}
PERIOD = "5y"
SELECTED_EXPERIMENT = "baseline_10d"
SELECTED_UNIVERSE = "core_25"
TRAINING_LOG = PROJECT_ROOT / "training.log"
ACTUAL_BASELINE_LABEL = "XGBoost Actual Return Baseline Report"
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


def _print_baseline_section(stats):
    print("\nFull test-set baseline:")
    if not stats:
        print("- unavailable")
        return

    print(f"- count: {stats.get('count', 'unavailable')}")
    print(f"- avg actual return: {_format_percent(stats.get('avg_actual_return'))}")
    print(
        f"- positive return rate: {_format_percent(stats.get('positive_return_rate'))}"
    )


def _print_experiment_summary(
    experiment_name,
    experiment,
    universe_name,
    ticker_count,
    log_path=TRAINING_LOG,
):
    actual_baseline_report = _extract_report_from_log(log_path, ACTUAL_BASELINE_LABEL)
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
    print(f"Universe: {universe_name}")
    print(f"Ticker count: {ticker_count}")
    print(f"Prediction horizon: {experiment['prediction_days']} trading days")
    _print_baseline_section(actual_baseline_report)
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
    baseline_return = (
        actual_baseline_report.get("avg_actual_return")
        if actual_baseline_report
        else None
    )
    if baseline_return is None:
        print("- Best selected bucket vs full-test average: unavailable")
        return

    difference_points = (best_return - baseline_return) * 100
    print(
        "- Best selected bucket beat full-test average by: "
        f"{difference_points:.2f} percentage points"
    )


def main() -> None:
    experiments = build_experiments()

    if SELECTED_EXPERIMENT not in experiments:
        raise ValueError(
            f"Unknown experiment {SELECTED_EXPERIMENT!r}. "
            f"Available experiments: {sorted(experiments)}"
        )
    if SELECTED_UNIVERSE not in TICKER_UNIVERSES:
        raise ValueError(
            f"Unknown ticker universe {SELECTED_UNIVERSE!r}. "
            f"Available universes: {sorted(TICKER_UNIVERSES)}"
        )

    experiment = experiments[SELECTED_EXPERIMENT]
    tickers = TICKER_UNIVERSES[SELECTED_UNIVERSE]

    print(f"=== Running experiment: {SELECTED_EXPERIMENT} ===")
    print(f"Ticker universe: {SELECTED_UNIVERSE}")
    print(f"Ticker count: {len(tickers)}")
    print(f"Prediction days: {experiment['prediction_days']}")
    print(f"Classifier params: {experiment['classifier_params']}")
    print(f"Regressor params: {experiment['regressor_params']}")

    data = prepare_data_parallel(tickers, period=PERIOD)
    train_models(data, **experiment)
    _print_experiment_summary(
        SELECTED_EXPERIMENT,
        experiment,
        SELECTED_UNIVERSE,
        len(tickers),
    )

    print("=== Sample predictions after training ===")
    for ticker in ["AAPL", "MSFT", "ARM", "XSD", "ROK", "JPM", "SPY"]:
        print(ticker, predict_price(ticker))


if __name__ == "__main__":
    main()
