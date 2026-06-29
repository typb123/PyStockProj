"""Training entry point for SPY-relative stock prediction models.

The pipeline keeps Linear Regression as a separate baseline artifact, while
XGBoost trains a beat-benchmark classifier and an excess-return regressor.
"""

import argparse
import re
import joblib
import pandas as pd
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.data.data_prep import DataPreparator
from src.data.technical_indicators import calculate_data
from src.data.data_fetch import fetch_stock_data
from src.train.evaluation import (
    build_actual_return_baseline_report,
    build_classification_report,
    build_combined_signal_report,
    build_model_only_top_n_basket_backtest_report,
    build_probability_summary,
    build_probability_tail_report,
    build_probability_threshold_report,
    build_probability_ranked_top_n_selection_reports,
    build_predicted_return_quantile_report,
    build_regression_report,
    build_return_correlation_report,
    build_top_n_selection_reports,
    build_trading_relevance_report,
    build_validation_selected_threshold_report,
)
from xgboost import XGBRegressor, XGBClassifier
from src.config import (
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
    MODEL_PATHS,
    TRAINING_TICKERS,
    PREDICTION_DAYS,
    TEST_SIZE,
)
from src.features.feature_contract import MODEL_FEATURE_COLUMNS


logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s: - %(levelname)s -%(message)s",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ALL_HORIZONS = [5, 10, 20, 50]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YFINANCE_CACHE_DIR = PROJECT_ROOT / "data/cache/yfinance"
YFINANCE_CACHE_FORMAT = "csv"
VALIDATION_TOP_N_SELECTION_VALUES = (5, 10, 20)


def validate_input_data(data):
    """
    Validate the input data structure before feature/target preparation.

    Missing values are logged but not filled here; DataPreparator handles required
    feature/target row dropping without future-looking imputation.
    """
    if data.empty:
        raise ValueError("Input data is empty.")

    nan_count = data.isnull().sum().sum()
    logging.debug(f"NaN values before preparation: {nan_count}")

    if nan_count > 0:
        nan_by_column = data.isnull().sum()
        nan_columns = [col for col in data.columns if nan_by_column[col] > 0]
        for col in nan_columns:
            logging.debug(
                f"Column {col}: {nan_by_column[col]} NaN values ({nan_by_column[col] / len(data) * 100:.2f}%)"
            )

    logging.info("Data validation complete.")
    return data


def log_feature_importances(model_name, feature_importances, top_n=5):
    """Log a compact INFO summary and full DEBUG feature-importance details."""
    top_importances = feature_importances.head(top_n)
    top_importance_summary = {
        row["feature"]: round(float(row["importance"]), 4)
        for _, row in top_importances.iterrows()
    }
    logging.info(
        f"Top {top_n} Feature Importances for {model_name}: {top_importance_summary}"
    )
    logging.debug(f"Feature Importances for {model_name}:")
    for _, row in feature_importances.iterrows():
        logging.debug(f"{row['feature']}: {row['importance']:.4f}")


def format_top_n_ranked_selection_summary(
    top_n_report,
    title="XGBoost Top-N Ranked Selection Summary:",
):
    """Format nested Top-N ranked-selection diagnostics for readable INFO logs."""
    lines = [title]

    for bucket_name, bucket_report in top_n_report.items():
        model = bucket_report.get("model", {})
        random_baseline = bucket_report.get("random_baseline", {})
        momentum_baseline = bucket_report.get("momentum_baseline", {})
        relative_momentum_baseline = bucket_report.get("relative_momentum_baseline", {})
        universe = bucket_report.get("universe", {})
        momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
        relative_momentum_label = relative_momentum_baseline.get(
            "relative_momentum_score_column",
            "relative_momentum",
        )

        model_excess = model.get("average_selected_excess_return_vs_benchmark")
        random_excess = random_baseline.get(
            "average_selected_excess_return_vs_benchmark"
        )
        momentum_excess = momentum_baseline.get(
            "average_selected_excess_return_vs_benchmark"
        )
        relative_momentum_excess = relative_momentum_baseline.get(
            "average_selected_excess_return_vs_benchmark"
        )
        universe_excess = universe.get("average_excess_return_vs_benchmark")
        beat_rate = model.get("beat_benchmark_rate")

        lines.append(f"{bucket_name}:")
        if momentum_baseline.get("available", True):
            momentum_text = _format_percent(momentum_excess)
            minus_momentum_text = _format_percent_delta(
                model_excess,
                momentum_excess,
            )
        else:
            momentum_text = "unavailable"
            minus_momentum_text = "unavailable"
        if relative_momentum_baseline.get("available", True):
            relative_momentum_text = _format_percent(relative_momentum_excess)
            minus_relative_momentum_text = _format_percent_delta(
                model_excess,
                relative_momentum_excess,
            )
        else:
            relative_momentum_text = "unavailable"
            minus_relative_momentum_text = "unavailable"

        lines.append(
            "  "
            f"model excess={_format_percent(model_excess)}, "
            f"random excess={_format_percent(random_excess)}, "
            f"{momentum_label} excess={momentum_text}, "
            f"{relative_momentum_label} excess={relative_momentum_text}, "
            f"universe excess={_format_percent(universe_excess)}"
        )
        lines.append(
            "  "
            f"model minus random={_format_percent_delta(model_excess, random_excess)}, "
            f"model minus momentum={minus_momentum_text}, "
            f"model minus relative momentum={minus_relative_momentum_text}, "
            f"model beat rate={_format_percent(beat_rate)}"
        )

    return "\n".join(lines)


def format_top_n_basket_backtest_summary(
    top_n_report,
    title="XGBoost Top-N Basket Backtest Summary:",
):
    """Format nested Top-N basket backtest diagnostics for readable INFO logs."""
    lines = [title]

    for bucket_name, bucket_report in top_n_report.items():
        model = bucket_report.get("model", {})
        random_baseline = bucket_report.get("random_baseline", {})
        momentum_baseline = bucket_report.get("momentum_baseline", {})
        relative_momentum_baseline = bucket_report.get("relative_momentum_baseline", {})
        universe = bucket_report.get("universe", {})
        benchmark = bucket_report.get("benchmark", {})
        momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
        relative_momentum_label = relative_momentum_baseline.get(
            "relative_momentum_score_column",
            "relative_momentum",
        )

        model_raw = model.get("average_basket_raw_return")
        model_excess = model.get("average_basket_excess_return")
        random_excess = random_baseline.get("average_basket_excess_return")
        momentum_excess = momentum_baseline.get("average_basket_excess_return")
        relative_momentum_excess = relative_momentum_baseline.get(
            "average_basket_excess_return"
        )
        universe_excess = universe.get("average_basket_excess_return")
        benchmark_raw = benchmark.get("average_basket_raw_return")
        beat_rate = model.get("beat_benchmark_rate")

        lines.append(f"{bucket_name}:")
        if momentum_baseline.get("available", True):
            momentum_text = _format_percent(momentum_excess)
            minus_momentum_text = _format_percent_delta(
                model_excess,
                momentum_excess,
            )
        else:
            momentum_text = "unavailable"
            minus_momentum_text = "unavailable"
        if relative_momentum_baseline.get("available", True):
            relative_momentum_text = _format_percent(relative_momentum_excess)
            minus_relative_momentum_text = _format_percent_delta(
                model_excess,
                relative_momentum_excess,
            )
        else:
            relative_momentum_text = "unavailable"
            minus_relative_momentum_text = "unavailable"

        lines.append(
            "  "
            f"model raw={_format_percent(model_raw)}, "
            f"model excess={_format_percent(model_excess)}, "
            f"random excess={_format_percent(random_excess)}, "
            f"{momentum_label} excess={momentum_text}, "
            f"{relative_momentum_label} excess={relative_momentum_text}, "
            f"universe excess={_format_percent(universe_excess)}, "
            f"benchmark raw={_format_percent(benchmark_raw)}"
        )
        lines.append(
            "  "
            f"model minus random={_format_percent_delta(model_excess, random_excess)}, "
            f"model minus momentum={minus_momentum_text}, "
            f"model minus relative momentum={minus_relative_momentum_text}, "
            f"beat benchmark rate={_format_percent(beat_rate)}"
        )

    return "\n".join(lines)


def format_top_n_basket_backtest_by_year_summary(
    top_n_by_year_report,
    title="XGBoost Top-N Basket Backtest By-Year Summary:",
):
    """Format by-year Top-N basket backtest diagnostics for readable INFO logs."""
    lines = [title]

    for bucket_name, bucket_report in top_n_by_year_report.items():
        for year, year_report in bucket_report.items():
            model = year_report.get("model", {})
            random_baseline = year_report.get("random_baseline", {})
            momentum_baseline = year_report.get("momentum_baseline", {})
            relative_momentum_baseline = year_report.get(
                "relative_momentum_baseline", {}
            )
            universe = year_report.get("universe", {})
            momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
            relative_momentum_label = relative_momentum_baseline.get(
                "relative_momentum_score_column",
                "relative_momentum",
            )

            model_excess = model.get("average_basket_excess_return")
            random_excess = random_baseline.get("average_basket_excess_return")
            momentum_excess = momentum_baseline.get("average_basket_excess_return")
            relative_momentum_excess = relative_momentum_baseline.get(
                "average_basket_excess_return"
            )
            universe_excess = universe.get("average_basket_excess_return")
            beat_rate = model.get("beat_benchmark_rate")
            evaluated_dates = model.get("evaluated_dates")

            if momentum_baseline.get("available", True):
                momentum_text = _format_percent(momentum_excess)
                minus_momentum_text = _format_percent_delta(
                    model_excess,
                    momentum_excess,
                )
            else:
                momentum_text = "unavailable"
                minus_momentum_text = "unavailable"
            if relative_momentum_baseline.get("available", True):
                relative_momentum_text = _format_percent(relative_momentum_excess)
                minus_relative_momentum_text = _format_percent_delta(
                    model_excess,
                    relative_momentum_excess,
                )
            else:
                relative_momentum_text = "unavailable"
                minus_relative_momentum_text = "unavailable"

            lines.append(
                f"{year} {bucket_name}: "
                f"model excess={_format_percent(model_excess)}, "
                f"random excess={_format_percent(random_excess)}, "
                f"{momentum_label} excess={momentum_text}, "
                f"{relative_momentum_label} excess={relative_momentum_text}, "
                f"universe excess={_format_percent(universe_excess)}, "
                f"model minus random={_format_percent_delta(model_excess, random_excess)}, "
                f"model minus momentum={minus_momentum_text}, "
                f"model minus relative momentum={minus_relative_momentum_text}, "
                f"beat benchmark rate={_format_percent(beat_rate)}, "
                f"evaluated dates={_format_count(evaluated_dates)}"
            )

    return "\n".join(lines)


def format_horizon_comparison_summary(horizon_reports):
    """Format basket backtest metrics across prediction horizons."""
    lines = ["Horizon Comparison Summary:"]

    for prediction_days in sorted(horizon_reports):
        lines.append(f"{prediction_days}d:")
        basket_report = horizon_reports[prediction_days].get("basket_backtest", {})

        for bucket_name, bucket_report in basket_report.items():
            model = bucket_report.get("model", {})
            random_baseline = bucket_report.get("random_baseline", {})
            momentum_baseline = bucket_report.get("momentum_baseline", {})
            relative_momentum_baseline = bucket_report.get(
                "relative_momentum_baseline", {}
            )
            universe = bucket_report.get("universe", {})
            benchmark = bucket_report.get("benchmark", {})
            momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
            relative_momentum_label = relative_momentum_baseline.get(
                "relative_momentum_score_column",
                "relative_momentum",
            )

            if momentum_baseline.get("available", True):
                momentum_text = _format_percent(
                    momentum_baseline.get("average_basket_excess_return")
                )
            else:
                momentum_text = "unavailable"
            if relative_momentum_baseline.get("available", True):
                relative_momentum_text = _format_percent(
                    relative_momentum_baseline.get("average_basket_excess_return")
                )
            else:
                relative_momentum_text = "unavailable"

            lines.append(
                "  "
                f"{bucket_name} "
                f"model excess={_format_percent(model.get('average_basket_excess_return'))}, "
                f"random={_format_percent(random_baseline.get('average_basket_excess_return'))}, "
                f"{momentum_label}={momentum_text}, "
                f"{relative_momentum_label}={relative_momentum_text}, "
                f"universe={_format_percent(universe.get('average_basket_excess_return'))}, "
                f"benchmark raw={_format_percent(benchmark.get('average_basket_raw_return'))}"
            )

    return "\n".join(lines)


def _format_percent(value):
    """Format optional numeric report values as percentages for logs."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_percent_delta(left, right):
    """Format the percentage-point difference between two report values."""
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return "n/a"
    return f"{float(left) - float(right):.2%}"


def _format_count(value):
    """Format optional count-like report values for logs."""
    if value is None or pd.isna(value):
        return "n/a"
    return str(int(value))


def _finite_report_float(value):
    """Return a finite float for scalar report values, otherwise None."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _candidate_params_with_overrides(base_params, overrides):
    """Return XGBoost params with conservative candidate overrides applied."""
    candidate_params = dict(base_params)
    candidate_params.update(overrides)
    return candidate_params


def build_xgboost_regressor_candidate_configs(base_params=None):
    """Build the small validation-selection candidate set for the regressor."""
    base_params = dict(XG_PARAMS_REGRESSOR if base_params is None else base_params)
    base_max_depth = int(base_params.get("max_depth", 5))
    base_min_child_weight = float(base_params.get("min_child_weight", 5))
    base_gamma = float(base_params.get("gamma", 0.05))
    base_reg_lambda = float(base_params.get("reg_lambda", 5.0))

    candidate_specs = [
        ("candidate_0_baseline", {}),
        (
            "candidate_1_shallower_more_regularized",
            {
                "max_depth": max(2, base_max_depth - 1),
                "min_child_weight": base_min_child_weight + 2,
                "gamma": base_gamma * 1.5,
                "reg_lambda": base_reg_lambda * 1.5,
            },
        ),
        (
            "candidate_2_slightly_deeper_less_regularized",
            {
                "max_depth": base_max_depth + 1,
                "min_child_weight": max(1, base_min_child_weight - 2),
                "gamma": base_gamma * 0.7,
                "reg_lambda": base_reg_lambda * 0.7,
            },
        ),
    ]

    candidates = []
    seen_param_sets = set()
    for candidate_id, (candidate_name, overrides) in enumerate(candidate_specs):
        params = _candidate_params_with_overrides(base_params, overrides)
        param_key = tuple(sorted(params.items()))
        if param_key in seen_param_sets:
            continue
        seen_param_sets.add(param_key)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "params": params,
            }
        )

    return candidates


def _validation_top_n_model_excess_returns(
    validation_basket_report,
    top_n_values=VALIDATION_TOP_N_SELECTION_VALUES,
):
    """Extract bucket-level validation model basket excess returns."""
    bucket_excess_returns = {}
    usable_values = []

    for top_n in top_n_values:
        bucket_name = f"top_{top_n}"
        excess_return = _finite_report_float(
            validation_basket_report.get(bucket_name, {})
            .get("model", {})
            .get("average_basket_excess_return")
        )
        bucket_excess_returns[bucket_name] = excess_return
        if excess_return is not None:
            usable_values.append(excess_return)

    return bucket_excess_returns, usable_values


def _validation_top_n_mean_excess_return(
    validation_basket_report,
    top_n_values=VALIDATION_TOP_N_SELECTION_VALUES,
):
    """Mean available validation model basket excess across Top-N buckets."""
    bucket_excess_returns, usable_values = _validation_top_n_model_excess_returns(
        validation_basket_report,
        top_n_values=top_n_values,
    )
    if not usable_values:
        return np.nan, bucket_excess_returns, 0

    return float(np.mean(usable_values)), bucket_excess_returns, len(usable_values)


def _is_better_validation_candidate(candidate_score, best_score):
    """Return whether candidate_score should replace best_score."""
    if candidate_score is None or pd.isna(candidate_score):
        return False
    if best_score is None or pd.isna(best_score):
        return True
    return float(candidate_score) > float(best_score)


def select_xgboost_regressor_by_validation_top_n(
    x_train,
    y_train,
    x_val,
    y_val,
    validation_split_metadata,
    candidate_configs=None,
    top_n_values=VALIDATION_TOP_N_SELECTION_VALUES,
):
    """Train candidate regressors and select by validation Top-N basket excess."""
    candidate_configs = (
        build_xgboost_regressor_candidate_configs()
        if candidate_configs is None
        else list(candidate_configs)
    )
    if not candidate_configs:
        raise ValueError("At least one XGBoost regressor candidate is required.")

    candidate_reports = []
    selected_model = None
    selected_candidate_report = None
    best_score = None

    for candidate_order, candidate_config in enumerate(candidate_configs):
        params = dict(candidate_config["params"])
        candidate_id = candidate_config.get("candidate_id", candidate_order)
        candidate_name = candidate_config.get(
            "candidate_name",
            f"candidate_{candidate_id}",
        )

        candidate_model = XGBRegressor(**params)
        candidate_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        validation_predictions = candidate_model.predict(x_val)
        validation_basket_report = build_model_only_top_n_basket_backtest_report(
            validation_split_metadata,
            validation_predictions,
            top_n_values=top_n_values,
        )
        (
            validation_score,
            bucket_excess_returns,
            available_bucket_count,
        ) = _validation_top_n_mean_excess_return(
            validation_basket_report,
            top_n_values=top_n_values,
        )
        candidate_report = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "params": params,
            "validation_top_n_mean_excess_return": validation_score,
            "validation_top_n_basket_excess_returns": bucket_excess_returns,
            "available_bucket_count": available_bucket_count,
        }
        candidate_reports.append(candidate_report)

        if selected_model is None or _is_better_validation_candidate(
            validation_score,
            best_score,
        ):
            selected_model = candidate_model
            selected_candidate_report = candidate_report
            best_score = validation_score

    if selected_candidate_report is None:
        selected_candidate_report = candidate_reports[0]

    for candidate_report in candidate_reports:
        candidate_report["selected"] = candidate_report is selected_candidate_report

    selection_report = {
        "selection_metric": "validation_top_n_mean_excess_return",
        "selection_bucket_policy": (
            "Mean of available top_5/top_10/top_20 model "
            "average_basket_excess_return values; unavailable or NaN buckets "
            "are ignored."
        ),
        "top_n_values": list(top_n_values),
        "candidates": candidate_reports,
        "selected_candidate_id": selected_candidate_report["candidate_id"],
        "selected_candidate_name": selected_candidate_report["candidate_name"],
        "selected_params": selected_candidate_report["params"],
        "selected_validation_top_n_mean_excess_return": (
            selected_candidate_report["validation_top_n_mean_excess_return"]
        ),
    }

    return selected_model, selection_report


def format_xgboost_regressor_validation_selection_report(selection_report):
    """Format validation-based XGBoost regressor selection for INFO logs."""
    lines = ["XGBoost Regressor Validation Top-N Selection Report:"]
    candidates = selection_report.get("candidates", [])

    for candidate_report in candidates:
        selected_suffix = " selected" if candidate_report.get("selected") else ""
        bucket_excess_returns = candidate_report.get(
            "validation_top_n_basket_excess_returns",
            {},
        )
        bucket_names = sorted(
            bucket_excess_returns,
            key=lambda bucket_name: int(bucket_name.split("_", maxsplit=1)[1]),
        )
        bucket_text = ", ".join(
            f"{bucket_name}={_format_percent(bucket_excess_returns.get(bucket_name))}"
            for bucket_name in bucket_names
        )
        lines.append(
            f"{candidate_report.get('candidate_name')}{selected_suffix}: "
            f"score={_format_percent(candidate_report.get('validation_top_n_mean_excess_return'))}, "
            f"{bucket_text}, "
            f"params={candidate_report.get('params')}"
        )

    lines.append(
        "selected="
        f"{selection_report.get('selected_candidate_name')} "
        f"score={_format_percent(selection_report.get('selected_validation_top_n_mean_excess_return'))}"
    )
    return "\n".join(lines)


def _sanitize_cache_key(value):
    """Normalize ticker and period strings for filesystem cache paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def get_yfinance_cache_path(ticker, period, cache_dir=None):
    """Return the raw OHLCV cache path for a ticker/period pair."""
    cache_dir = YFINANCE_CACHE_DIR if cache_dir is None else cache_dir
    ticker_key = _sanitize_cache_key(ticker.upper())
    period_key = _sanitize_cache_key(period)
    return Path(cache_dir) / f"{ticker_key}__{period_key}.{YFINANCE_CACHE_FORMAT}"


def normalize_raw_ohlcv_index(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw OHLCV indexes to sorted, timezone-naive date timestamps."""
    normalized_data = data.copy()
    normalized_index = pd.DatetimeIndex(pd.to_datetime(normalized_data.index, utc=True))
    normalized_data.index = normalized_index.tz_convert(None).normalize()
    normalized_data.index.name = "Date"
    return normalized_data.sort_index()


def load_cached_yfinance_data(ticker, period, cache_dir=None):
    """Load raw yfinance OHLCV data from cache, or return None on miss/failure."""
    cache_path = get_yfinance_cache_path(ticker, period, cache_dir=cache_dir)
    if not cache_path.exists():
        logging.info(f"YFinance cache miss for {ticker} period={period}.")
        return None

    try:
        data = pd.read_csv(cache_path, index_col=0)
        data = normalize_raw_ohlcv_index(data)
        logging.info(f"YFinance cache hit for {ticker} period={period}: {cache_path}")
        return data
    except Exception as exc:
        logging.warning(
            f"Failed to read YFinance cache for {ticker} period={period} "
            f"from {cache_path}; refetching. Error: {exc}"
        )
        return None


def write_yfinance_cache(data, ticker, period, cache_dir=None):
    """Write raw yfinance OHLCV data to cache; warn but do not fail on errors."""
    cache_path = get_yfinance_cache_path(ticker, period, cache_dir=cache_dir)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path)
        logging.info(f"Wrote YFinance cache for {ticker} period={period}: {cache_path}")
    except Exception as exc:
        logging.warning(
            f"Failed to write YFinance cache for {ticker} period={period} "
            f"to {cache_path}. Error: {exc}"
        )


def fetch_raw_ticker_data(ticker, period="5y", use_cache=True):
    """Fetch raw OHLCV data, optionally using the ticker/period cache."""
    if use_cache:
        cached_data = load_cached_yfinance_data(ticker, period)
        if cached_data is not None:
            return cached_data

    stock_data = fetch_stock_data(ticker, period=period)
    if stock_data.empty:
        return stock_data

    stock_data = normalize_raw_ohlcv_index(stock_data)
    if use_cache:
        write_yfinance_cache(stock_data, ticker, period)
    return stock_data


def fetch_tickers_data(ticker, period="5y", use_cache=True):
    """
    Fetch one ticker and generate indicators before ticker-level concatenation.

    calculate_data is order-dependent and not grouped internally, so the current
    fetch path calls it while each DataFrame still contains one ticker only.
    """
    try:
        stock_data = fetch_raw_ticker_data(ticker, period=period, use_cache=use_cache)
        if stock_data.empty:
            logging.debug(f"No data returned for {ticker}. Skipping...")
            return None

        logging.debug(f"Data size before processing for {ticker}: {stock_data.shape}")

        stock_data = calculate_data(stock_data)
        stock_data["prediction_date"] = stock_data.index
        stock_data["Ticker"] = ticker

        logging.debug(f"Data size after processing for {ticker}: {stock_data.shape}")

        time.sleep(0.2)  # Small delay to avoid rate limits
        return stock_data
    except Exception as e:
        logging.debug(f"Failed to fetch data for {ticker}: {e}", exc_info=True)
        return None


def prepare_data_parallel(tickers, period="5y", use_cache=True) -> pd.DataFrame:
    """Fetch and prepare each ticker independently, then concatenate valid results."""
    logging.info(
        f"Fetching data for {len(tickers)} tickers with period={period}; "
        f"cache_enabled={use_cache}."
    )

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(
                lambda ticker: fetch_tickers_data(
                    ticker,
                    period=period,
                    use_cache=use_cache,
                ),
                tickers,
            )
        )

    all_data = [data for data in results if data is not None and not data.empty]
    skipped_tickers = [
        ticker for ticker, data in zip(tickers, results) if data is None or data.empty
    ]

    logging.info(
        f"Fetched valid data for {len(all_data)} of {len(tickers)} requested tickers."
    )
    if skipped_tickers:
        logging.warning(
            f"Skipped {len(skipped_tickers)} tickers with no usable data: {skipped_tickers}"
        )

    if not all_data:
        logging.error("No data fetched for training. Exiting...")
        raise ValueError("No data was fetched for any ticker.")

    return pd.concat(all_data, ignore_index=True)


def cross_validate_model(x, y, model=None, cv=5):
    """
    Perform cross-validation on the given model and data.

    Parameters:
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Target variable.
        model: Scikit-learn model (default: LinearRegression).
        cv (int): Number of cross-validation folds.

    Returns:
        float: Average cross-validated MSE.
    """
    if model is None:
        model = LinearRegression()

    # Define custom scoring for negative MSE
    scores = cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=cv)
    if scores is None:
        raise ValueError(
            "Cross-validation scoring failed. Please check your model and data."
        )
    avg_mse = -np.mean(scores)
    logging.info(f"Cross-Validated MSE (cv={cv}): {avg_mse:.4f}")
    return avg_mse


def evaluate_model(model, x_test, y_test, model_type="regression"):
    """Log a compact regression or classification report for a held-out split."""
    y_pred = model.predict(x_test)

    if model_type == "regression":
        report = build_regression_report(y_test, y_pred)
        logging.info(
            f"MSE: {report['mse']:.4f}, MAE: {report['mae']:.4f}, R²: {report['r2']:.4f}"
        )

    elif model_type == "classification":
        report = build_classification_report(y_test, y_pred)
        logging.info(f"Accuracy: {report['accuracy']:.4%}")
        logging.info(f"Classification Report: {report}")


def log_xgboost_test_report(
    y_train,
    y_val,
    y_test,
    direction_y_test,
    test_split_metadata,
    classifier_predictions,
    classifier_validation_probability_up,
    classifier_probability_up,
    regressor_predictions,
    prediction_days=PREDICTION_DAYS,
    random_trials=100,
    random_trial_workers=4,
    regressor_validation_selection_report=None,
):
    """
    Log final SPY-relative XGBoost diagnostics without tuning on test data.

    Validation probabilities may select a beat-benchmark threshold; test
    probabilities only evaluate the already-selected rule. y values are excess
    returns relative to SPY.
    """
    classification_report_data = build_classification_report(
        direction_y_test, classifier_predictions
    )
    regression_report_data = build_regression_report(
        y_test, regressor_predictions, y_train=y_train
    )
    actual_return_baseline_report = build_actual_return_baseline_report(y_test)
    trading_report_data = build_trading_relevance_report(
        y_test, regressor_predictions, classifier_predictions
    )
    probability_summary = build_probability_summary(classifier_probability_up)
    probability_tail_report = build_probability_tail_report(
        classifier_probability_up, y_test
    )
    probability_threshold_report = build_probability_threshold_report(
        classifier_probability_up, y_test
    )
    validation_selected_threshold_report = build_validation_selected_threshold_report(
        classifier_validation_probability_up,
        y_val,
        classifier_probability_up,
        y_test,
    )
    predicted_return_quantile_report = build_predicted_return_quantile_report(
        y_test, regressor_predictions
    )
    return_correlation_report = build_return_correlation_report(
        y_test, regressor_predictions
    )
    combined_signal_report = build_combined_signal_report(
        classifier_probability_up,
        y_test,
        regressor_predictions,
    )
    top_n_selection_reports = build_top_n_selection_reports(
        test_split_metadata,
        regressor_predictions,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    top_n_ranked_selection_report = top_n_selection_reports["ranked_selection"]
    top_n_basket_backtest_report = top_n_selection_reports["basket_backtest"]
    top_n_basket_backtest_by_year_report = top_n_selection_reports.get(
        "basket_backtest_by_year",
        {},
    )
    classifier_top_n_selection_reports = build_probability_ranked_top_n_selection_reports(
        test_split_metadata,
        classifier_probability_up,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    classifier_top_n_ranked_selection_report = classifier_top_n_selection_reports[
        "ranked_selection"
    ]
    classifier_top_n_basket_backtest_report = classifier_top_n_selection_reports[
        "basket_backtest"
    ]
    classifier_top_n_basket_backtest_by_year_report = (
        classifier_top_n_selection_reports.get("basket_backtest_by_year", {})
    )

    logging.info(
        f"XGBoost Beat-Benchmark Classification Report: {classification_report_data}"
    )
    logging.info(f"XGBoost Excess Return Regression Report: {regression_report_data}")
    logging.info(
        f"XGBoost Excess Return Baseline Report: {actual_return_baseline_report}"
    )
    logging.info(f"XGBoost Beat-Benchmark Relevance Report: {trading_report_data}")
    logging.info(f"XGBoost Beat-Benchmark Probability Summary: {probability_summary}")
    logging.info(
        f"XGBoost Beat-Benchmark Probability Tail Report: {probability_tail_report}"
    )
    logging.info(
        f"XGBoost Beat-Benchmark Probability Threshold Report: {probability_threshold_report}"
    )
    logging.info(
        f"XGBoost Beat-Benchmark Validation-Selected Threshold Report: {validation_selected_threshold_report}"
    )
    logging.info(
        f"XGBoost Regressor Predicted Excess Return Quantile Report: {predicted_return_quantile_report}"
    )
    logging.info(
        f"XGBoost Regressor Excess Return Correlation Report: {return_correlation_report}"
    )
    logging.info(f"XGBoost Combined Signal Report: {combined_signal_report}")
    if regressor_validation_selection_report:
        logging.info(
            format_xgboost_regressor_validation_selection_report(
                regressor_validation_selection_report
            )
        )
    logging.info(format_top_n_ranked_selection_summary(top_n_ranked_selection_report))
    logging.info(format_top_n_basket_backtest_summary(top_n_basket_backtest_report))
    if top_n_basket_backtest_by_year_report:
        logging.info(
            format_top_n_basket_backtest_by_year_summary(
                top_n_basket_backtest_by_year_report
            )
        )
    logging.info(
        format_top_n_ranked_selection_summary(
            classifier_top_n_ranked_selection_report,
            title="XGBoost Classifier-Probability Top-N Ranked Selection Summary:",
        )
    )
    logging.info(
        format_top_n_basket_backtest_summary(
            classifier_top_n_basket_backtest_report,
            title="XGBoost Classifier-Probability Top-N Basket Backtest Summary:",
        )
    )
    logging.debug(
        f"XGBoost Top-N Ranked Selection Report: {top_n_ranked_selection_report}"
    )
    logging.debug(
        f"XGBoost Top-N Basket Backtest Report: {top_n_basket_backtest_report}"
    )
    logging.debug(
        "XGBoost Top-N Basket Backtest By-Year Report: "
        f"{top_n_basket_backtest_by_year_report}"
    )
    logging.debug(
        "XGBoost Classifier-Probability Top-N Ranked Selection Report: "
        f"{classifier_top_n_ranked_selection_report}"
    )
    logging.debug(
        "XGBoost Classifier-Probability Top-N Basket Backtest Report: "
        f"{classifier_top_n_basket_backtest_report}"
    )
    logging.debug(
        "XGBoost Classifier-Probability Top-N Basket Backtest By-Year Report: "
        f"{classifier_top_n_basket_backtest_by_year_report}"
    )
    return {
        "regressor_validation_selection": regressor_validation_selection_report or {},
        "ranked_selection": top_n_ranked_selection_report,
        "basket_backtest": top_n_basket_backtest_report,
        "basket_backtest_by_year": top_n_basket_backtest_by_year_report,
        "classifier_probability_ranked_selection": (
            classifier_top_n_ranked_selection_report
        ),
        "classifier_probability_basket_backtest": classifier_top_n_basket_backtest_report,
        "classifier_probability_basket_backtest_by_year": (
            classifier_top_n_basket_backtest_by_year_report
        ),
    }


def build_model_metadata(
    linear_features,
    classifier_features,
    regressor_features,
    prediction_days,
):
    """Build prediction-time metadata needed to align saved artifacts and features."""
    return {
        "linear_features": linear_features,
        "classifier_features": classifier_features,
        "regressor_features": regressor_features,
        "classifier_feature_names": classifier_features,
        "regressor_feature_names": regressor_features,
        "prediction_days": prediction_days,
        "target_type": "spy_relative_excess_forward_return",
        "benchmark_ticker": "SPY",
        "regressor_target": "targetReturns",
        "classifier_target": "beat_benchmark_target",
    }


def build_horizon_model_paths(prediction_days: int) -> dict:
    """Return artifact paths for one prediction horizon."""
    horizon_dir = Path("models") / f"horizon_{prediction_days}"
    return {
        "linear": str(horizon_dir / "linear_regression_model.pkl"),
        "linear_scaler": str(horizon_dir / "linear_regression_scaler.pkl"),
        "classifier": str(horizon_dir / "xgboost_classifier.json"),
        "regressor": str(horizon_dir / "xgboost_regressor.json"),
        "preparator": str(horizon_dir / "data_preparator.pkl"),
        "features": str(horizon_dir / "feature_names.pkl"),
        "model_metadata": str(horizon_dir / "model_metadata.pkl"),
    }


def save_model_artifacts(
    artifact_paths,
    linear_model,
    scaler_lr,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
    regressor,
):
    """Save trained artifacts to one complete path set."""
    for path in artifact_paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(linear_model, artifact_paths["linear"])
    joblib.dump(scaler_lr, artifact_paths["linear_scaler"])
    joblib.dump(data_preparator, artifact_paths["preparator"])
    joblib.dump(all_features, artifact_paths["features"])
    joblib.dump(model_metadata, artifact_paths["model_metadata"])
    classifier.save_model(artifact_paths["classifier"])
    regressor.save_model(artifact_paths["regressor"])


def save_horizon_model_artifacts(
    prediction_days,
    linear_model,
    scaler_lr,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
    regressor,
):
    """Save horizon-specific artifacts and preserve legacy paths for default horizon."""
    horizon_paths = build_horizon_model_paths(prediction_days)
    save_model_artifacts(
        horizon_paths,
        linear_model,
        scaler_lr,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
        regressor,
    )

    if prediction_days == PREDICTION_DAYS:
        save_model_artifacts(
            MODEL_PATHS,
            linear_model,
            scaler_lr,
            data_preparator,
            all_features,
            model_metadata,
            classifier,
            regressor,
        )

    return horizon_paths


def train_models(
    data: pd.DataFrame,
    prediction_days: int = PREDICTION_DAYS,
    random_trials: int = 100,
    random_trial_workers: int = 4,
    classifier_params: dict | None = None,
    regressor_params: dict | None = None,
) -> dict:
    """
    Train the linear baseline, beat-benchmark classifier, and excess-return regressor.

    Validation data is used for XGBoost eval_set and threshold research; test data
    is reserved for final diagnostics.
    """
    logging.info(
        "Training models with explicitly defined feature arrays "
        f"for prediction_days={prediction_days}."
    )

    # DataPreparator owns target creation, chronological splits, and shared scaling.
    data = validate_input_data(data)
    data_preparator = DataPreparator()
    prepared_data = data_preparator.prepare_for_train(
        data, prediction_days=prediction_days, test_size=TEST_SIZE
    )

    # Rebuild DataFrames so explicit feature lists preserve their trained column order.
    all_features = prepared_data["feature_names"]
    x_train_full = pd.DataFrame(prepared_data["x_train"], columns=all_features)
    x_val_full = pd.DataFrame(prepared_data["x_val"], columns=all_features)
    x_test_full = pd.DataFrame(prepared_data["x_test"], columns=all_features)
    y_train = prepared_data["y_train"]
    y_val = prepared_data["y_val"]
    y_test = prepared_data["y_test"]
    logging.info(
        "Prepared data summary: "
        f"x_train={len(x_train_full)}, x_val={len(x_val_full)}, "
        f"x_test={len(x_test_full)}, features={len(all_features)}"
    )

    # Feature lists are intentionally inline for now; prediction metadata mirrors them.
    linear_features = [
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_lag_close_26",
        "chikou_return_26",
        "chikou_above_lag_26",
        "Open",
        "Close",
        "rsi",
        "signalLine",
        "ATR",
        "20_day_avg",
        "macd",
        "BB_Std",
        "obv",
        "dailyReturn",
        "macdHistogram",
        "vma_20",
        "High",
        "Low",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "stoch_k",
        "stoch_d",
        "Volume",
        "10_day_avg",
        "volatility",
        "vma_10",
        "5_day_avg",
    ]
    classifier_features = list(MODEL_FEATURE_COLUMNS)
    regressor_features = list(MODEL_FEATURE_COLUMNS)

    # Linear Regression is a separate scaled baseline, not a stacked XGBoost feature.
    scaler_lr = StandardScaler()
    x_train_lr_scaled = scaler_lr.fit_transform(x_train_full[linear_features])
    x_test_lr_scaled = scaler_lr.transform(x_test_full[linear_features])

    logging.info("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(x_train_lr_scaled, y_train)
    evaluate_model(linear_model, x_test_lr_scaled, y_test, model_type="regression")

    importances_lr = np.abs(linear_model.coef_)
    feature_importance_lr = pd.DataFrame(
        {"feature": linear_features, "importance": importances_lr}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances("Linear Regression", feature_importance_lr)

    classifier_feature_names = classifier_features
    regressor_feature_names = regressor_features

    x_train_classifier = x_train_full[classifier_features]
    x_val_classifier = x_val_full[classifier_features]
    x_test_classifier = x_test_full[classifier_features]
    x_train_regressor = x_train_full[regressor_features]
    x_val_regressor = x_val_full[regressor_features]
    x_test_regressor = x_test_full[regressor_features]

    # The classifier predicts beat-SPY labels; the regressor predicts excess return.
    direction_y_train = prepared_data["direction_y_train"]
    direction_y_val = prepared_data["direction_y_val"]
    direction_y_test = prepared_data["direction_y_test"]

    logging.info("Training XGBoost Classifier...")
    classifier = XGBClassifier(**(classifier_params or XG_PARAMS_CLASSIFIER))
    classifier.fit(
        x_train_classifier,
        direction_y_train,
        eval_set=[(x_val_classifier, direction_y_val)],
        verbose=False,
    )
    evaluate_model(
        classifier, x_test_classifier, direction_y_test, model_type="classification"
    )

    importances_clf = classifier.feature_importances_
    feature_importance_clf = pd.DataFrame(
        {"feature": classifier_feature_names, "importance": importances_clf}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances("XGBoost Classifier", feature_importance_clf)

    logging.info("Training XGBoost Regressor...")
    regressor_candidate_configs = build_xgboost_regressor_candidate_configs(
        regressor_params or XG_PARAMS_REGRESSOR
    )
    (
        regressor,
        regressor_validation_selection_report,
    ) = select_xgboost_regressor_by_validation_top_n(
        x_train_regressor,
        y_train,
        x_val_regressor,
        y_val,
        prepared_data["split_metadata"]["val"],
        candidate_configs=regressor_candidate_configs,
    )
    evaluate_model(regressor, x_test_regressor, y_test, model_type="regression")
    # Validation probabilities choose the beat-benchmark threshold; test evaluates it once.
    xgboost_test_reports = log_xgboost_test_report(
        y_train,
        y_val,
        y_test,
        direction_y_test,
        prepared_data["split_metadata"]["test"],
        classifier.predict(x_test_classifier),
        classifier.predict_proba(x_val_classifier)[:, 1],
        classifier.predict_proba(x_test_classifier)[:, 1],
        regressor.predict(x_test_regressor),
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
        regressor_validation_selection_report=regressor_validation_selection_report,
    )

    importances_reg = regressor.feature_importances_
    feature_importance_reg = pd.DataFrame(
        {"feature": regressor_feature_names, "importance": importances_reg}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances("XGBoost Regressor", feature_importance_reg)

    # Save all preprocessing and feature metadata needed to reproduce training inputs.
    model_metadata = build_model_metadata(
        linear_features,
        classifier_features,
        regressor_features,
        prediction_days,
    )
    model_metadata["regressor_validation_selection"] = (
        regressor_validation_selection_report
    )
    model_metadata["xgboost_regressor_selected_candidate_name"] = (
        regressor_validation_selection_report.get("selected_candidate_name")
    )
    model_metadata["xgboost_regressor_selected_params"] = (
        regressor_validation_selection_report.get("selected_params")
    )
    saved_paths = save_horizon_model_artifacts(
        prediction_days,
        linear_model,
        scaler_lr,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
        regressor,
    )
    logging.info(
        f"Training completed for prediction_days={prediction_days}. "
        f"Models saved to {Path(saved_paths['model_metadata']).parent}."
    )
    return {
        "prediction_days": prediction_days,
        "model_metadata": model_metadata,
        "artifact_paths": saved_paths,
        "regressor_validation_selection": xgboost_test_reports.get(
            "regressor_validation_selection",
            {},
        ),
        "basket_backtest": xgboost_test_reports["basket_backtest"],
    }


def _positive_int(value, argument_name="value"):
    """Parse a command-line value as a positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must be a positive integer"
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must be a positive integer"
        )
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
        default=4,
        help=(
            "Worker count for Top-N random baseline trials. Use 1 for sequential "
            "execution."
        ),
    )

    args = parser.parse_args(argv)
    if args.all_horizons:
        args.horizons = list(ALL_HORIZONS)
    else:
        args.prediction_days = args.prediction_days or PREDICTION_DAYS
        args.horizons = [args.prediction_days]

    return args


def main(argv=None):
    """Run the end-to-end training workflow from CLI arguments."""
    args = parse_args(argv)
    use_cache = not args.no_cache
    logging.info(f"Selected YFinance period: {args.period}")
    logging.info(f"Raw YFinance OHLCV cache enabled: {use_cache}")
    data = prepare_data_parallel(
        TRAINING_TICKERS,
        period=args.period,
        use_cache=use_cache,
    )
    if data.empty:
        logging.error("No data fetched for training. Exiting...")
        return

    horizon_reports = {}
    for prediction_days in args.horizons:
        logging.info(f"Starting model training for prediction_days={prediction_days}.")
        horizon_reports[prediction_days] = train_models(
            data.copy(),
            prediction_days=prediction_days,
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
