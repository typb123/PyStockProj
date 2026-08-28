"""Training entry point for SPY-relative stock prediction models.

The pipeline keeps Linear Regression as a separate baseline artifact, while
XGBoost trains a beat-benchmark classifier and an excess-return regressor.
"""

import argparse
import logging
from src.data.training_data import (
    PROJECT_ROOT,
    YFINANCE_CACHE_DIR,
    YFINANCE_CACHE_FORMAT,
    fetch_raw_ticker_data,
    fetch_tickers_data,
    get_yfinance_cache_path,
    load_cached_yfinance_data,
    normalize_raw_ohlcv_index,
    prepare_data_parallel,
    validate_input_data,
    write_yfinance_cache,
)
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    TARGET_MODE_EXCESS_RETURN,
    TARGET_MODES,
    WALK_FORWARD_TARGET_MODES,
    resolve_model_feature_columns,
    validate_target_mode,
    validate_walk_forward_target_mode,
)
from src.train.reporting import (
    evaluate_model,
    format_horizon_comparison_summary,
    format_same_date_ranking_diagnostics_summary,
    format_top_n_basket_backtest_by_year_summary,
    format_top_n_basket_backtest_summary,
    format_top_n_ranked_selection_summary,
    format_walk_forward_fold_summary,
    format_walk_forward_summary,
    format_xgboost_regressor_validation_selection_report,
    log_feature_importances,
    log_rank_ndcg_test_report,
    log_ranking_classifier_test_report,
    log_xgboost_test_report,
)
from src.train.model_training import (
    VALIDATION_TOP_N_SELECTION_VALUES,
    XG_PARAMS_RANKER,
    build_xgboost_regressor_candidate_configs,
    cross_validate_model,
    select_xgboost_regressor_by_validation_top_n,
    train_cross_sectional_rank_ndcg_model,
    train_cross_sectional_ranking_model,
    train_models,
)
from src.train.walk_forward_runner import run_walk_forward_models
from src.config import (
    DEFAULT_TRAINING_UNIVERSE,
    TRAINING_UNIVERSES,
    get_training_tickers,
    PREDICTION_DAYS,
)
from src.train.evaluation.walk_forward import (
    DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
    DEFAULT_WALK_FORWARD_TEST_YEARS,
    DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
)


logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s: - %(levelname)s -%(message)s",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ALL_HORIZONS = [5, 10, 20, 50]


def _positive_int(value, argument_name="value"):
    """Parse a command-line value as a positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must be a positive integer"
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{argument_name} must be a positive integer")
    return parsed


def _positive_prediction_days(value):
    """Parse the prediction horizon CLI argument."""
    return _positive_int(value, "prediction_days")


def _positive_random_trials(value):
    """Parse the random baseline trial-count CLI argument."""
    return _positive_int(value, "random_trials")


def _positive_random_trial_workers(value):
    """Parse the random baseline worker-count CLI argument."""
    return _positive_int(value, "random_trial_workers")


def _positive_year_count(value, argument_name="years"):
    """Parse a positive walk-forward year-count CLI argument."""
    return _positive_int(value, argument_name)


def parse_args(argv=None):
    """Parse trainer CLI options and resolve the requested prediction horizons."""
    parser = argparse.ArgumentParser(
        description="Train SPY-relative stock prediction models."
    )
    horizon_group = parser.add_mutually_exclusive_group()
    horizon_group.add_argument(
        "-d",
        "--prediction-days",
        type=_positive_prediction_days,
        default=None,
        help=f"Prediction horizon in trading days. Defaults to {PREDICTION_DAYS}.",
    )
    horizon_group.add_argument(
        "--all-horizons",
        action="store_true",
        help=f"Train research horizons {ALL_HORIZONS}.",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help='YFinance history period for raw OHLCV fetches. Defaults to "5y".',
    )
    parser.add_argument(
        "--universe",
        choices=sorted(TRAINING_UNIVERSES),
        default=DEFAULT_TRAINING_UNIVERSE,
        help=f"Training universe to fetch. Defaults to {DEFAULT_TRAINING_UNIVERSE}.",
    )
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default=TARGET_MODE_EXCESS_RETURN,
        help=(
            "Training target mode. Cross-sectional modes are accepted for "
            "experimental ranking research."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Bypass the raw YFinance OHLCV cache entirely: always fetch from "
            "YFinance and do not read or write cache files."
        ),
    )
    parser.add_argument(
        "--random-trials",
        type=_positive_random_trials,
        default=100,
        help=(
            "Controls random baseline trials for Top-N ranked-selection and "
            "basket-backtest reports."
        ),
    )
    parser.add_argument(
        "--random-trial-workers",
        type=_positive_random_trial_workers,
        default=8,
        help=(
            "Worker count for Top-N random baseline trials. Use 1 for sequential "
            "execution."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Run expanding-window yearly walk-forward evaluation for one horizon "
            "instead of normal training/artifact saving."
        ),
    )
    parser.add_argument(
        "--walk-forward-min-train-years",
        type=lambda value: _positive_year_count(value, "walk_forward_min_train_years"),
        default=DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
        help=("Minimum training-history window in years for walk-forward evaluation."),
    )
    parser.add_argument(
        "--walk-forward-validation-years",
        type=lambda value: _positive_year_count(value, "walk_forward_validation_years"),
        default=DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
        help="Validation window in years for walk-forward evaluation.",
    )
    parser.add_argument(
        "--walk-forward-test-years",
        type=lambda value: _positive_year_count(value, "walk_forward_test_years"),
        default=DEFAULT_WALK_FORWARD_TEST_YEARS,
        help="Test window in years for walk-forward evaluation.",
    )

    args = parser.parse_args(argv)
    if args.walk_forward and args.all_horizons:
        parser.error("--walk-forward supports one prediction horizon at a time.")
    if args.walk_forward and args.target_mode not in WALK_FORWARD_TARGET_MODES:
        parser.error(
            "--walk-forward supports target modes: "
            f"{', '.join(WALK_FORWARD_TARGET_MODES)}."
        )

    if args.all_horizons:
        args.horizons = list(ALL_HORIZONS)
    else:
        args.prediction_days = args.prediction_days or PREDICTION_DAYS
        args.horizons = [args.prediction_days]

    return args


def main(argv=None):
    """Run the end-to-end training workflow from CLI arguments."""
    args = parse_args(argv)
    validate_target_mode(args.target_mode)
    use_cache = not args.no_cache
    training_tickers = get_training_tickers(args.universe)
    logging.info(f"Selected YFinance period: {args.period}")
    logging.info(f"Selected target mode: {args.target_mode}")
    logging.info(f"Raw YFinance OHLCV cache enabled: {use_cache}")
    logging.info(
        f"Selected training universe: {args.universe} "
        f"({len(training_tickers)} tickers): {training_tickers}"
    )
    data = prepare_data_parallel(
        training_tickers,
        period=args.period,
        use_cache=use_cache,
    )
    if data.empty:
        logging.error("No data fetched for training. Exiting...")
        return

    if args.walk_forward:
        prediction_days = args.horizons[0]
        walk_forward_report = run_walk_forward_models(
            data.copy(),
            prediction_days=prediction_days,
            random_trials=args.random_trials,
            random_trial_workers=args.random_trial_workers,
            min_train_years=args.walk_forward_min_train_years,
            validation_years=args.walk_forward_validation_years,
            test_years=args.walk_forward_test_years,
            target_mode=args.target_mode,
        )
        summary = format_walk_forward_summary(walk_forward_report)
        logging.info(summary)
        print(summary)
        return walk_forward_report

    horizon_reports = {}
    for prediction_days in args.horizons:
        logging.info(f"Starting model training for prediction_days={prediction_days}.")
        horizon_reports[prediction_days] = train_models(
            data.copy(),
            prediction_days=prediction_days,
            target_mode=args.target_mode,
            random_trials=args.random_trials,
            random_trial_workers=args.random_trial_workers,
        )
        logging.info(f"Model training completed for prediction_days={prediction_days}.")

    if args.all_horizons:
        summary = format_horizon_comparison_summary(horizon_reports)
        logging.info(summary)
        print(summary)
    else:
        prediction_days = args.horizons[0]
        summary = format_top_n_basket_backtest_summary(
            horizon_reports[prediction_days]["basket_backtest"]
        )
        print(summary)

    print("Model training completed. Check training.log for details.")


if __name__ == "__main__":
    main()
