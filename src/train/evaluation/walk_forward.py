"""Expanding-window walk-forward helpers for Top-N model diagnostics."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PREDICTION_DAYS
from src.data.data_prep import DataPreparator
from src.features.feature_contract import MODEL_FEATURE_COLUMNS
from src.train.evaluation.validation import _mean_or_nan


DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS = 5
DEFAULT_WALK_FORWARD_VALIDATION_YEARS = 1
DEFAULT_WALK_FORWARD_TEST_YEARS = 1
DEFAULT_WALK_FORWARD_STEP_YEARS = 1
DEFAULT_WALK_FORWARD_TOP_N_BUCKETS = ("top_5", "top_10", "top_20")


def prepare_walk_forward_model_frame(
    data,
    prediction_days=PREDICTION_DAYS,
    data_preparator=None,
):
    """Create the SPY-relative modeling frame used by yearly walk-forward folds."""
    data_preparator = data_preparator or DataPreparator()
    prepared = data_preparator.prepare_features(data)
    prepared = data_preparator._normalize_prediction_date(prepared)
    data_preparator._validate_required_columns(prepared)
    prepared = data_preparator._create_raw_forward_returns(
        prepared,
        prediction_days,
    )
    prepared = data_preparator._add_benchmark_relative_momentum(prepared)
    benchmark_returns = data_preparator._build_benchmark_forward_returns(prepared)
    prepared = data_preparator._create_benchmark_relative_targets(
        prepared,
        benchmark_returns,
    )
    data_preparator.feature_columns = list(MODEL_FEATURE_COLUMNS)
    prepared = data_preparator._drop_unusable_rows(prepared)
    return prepared, list(data_preparator.feature_columns)


def build_expanding_yearly_walk_forward_folds(
    prepared_frame,
    min_train_years=DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
    validation_years=DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
    test_years=DEFAULT_WALK_FORWARD_TEST_YEARS,
    step_years=DEFAULT_WALK_FORWARD_STEP_YEARS,
):
    """Build chronological expanding-window folds from available prediction years."""
    _validate_positive_year_count(min_train_years, "min_train_years")
    _validate_positive_year_count(validation_years, "validation_years")
    _validate_positive_year_count(test_years, "test_years")
    _validate_positive_year_count(step_years, "step_years")

    frame = _frame_with_prediction_year(prepared_frame)
    years = sorted(frame["_prediction_year"].drop_duplicates().astype(int))
    max_train_year_count = len(years) - validation_years - test_years
    if max_train_year_count < min_train_years:
        return []

    folds = []
    train_year_count = int(min_train_years)
    while train_year_count <= max_train_year_count:
        validation_start = train_year_count
        test_start = validation_start + validation_years
        train_years = years[:train_year_count]
        validation_fold_years = years[validation_start:test_start]
        test_fold_years = years[test_start : test_start + test_years]
        fold = {
            "fold_index": len(folds),
            "train_years": [int(year) for year in train_years],
            "validation_years": [int(year) for year in validation_fold_years],
            "test_years": [int(year) for year in test_fold_years],
            "train_date_range": _date_range_for_years(frame, train_years),
            "validation_date_range": _date_range_for_years(
                frame,
                validation_fold_years,
            ),
            "test_date_range": _date_range_for_years(frame, test_fold_years),
        }
        folds.append(fold)
        train_year_count += int(step_years)

    return folds


def build_walk_forward_split(
    prepared_frame,
    fold,
    feature_columns,
    prediction_days=PREDICTION_DAYS,
):
    """Scale and package one walk-forward train/validation/test split."""
    frame = _frame_with_prediction_year(prepared_frame)
    train_df = _rows_for_years(frame, fold["train_years"])
    validation_df = _rows_for_years(frame, fold["validation_years"])
    test_df = _rows_for_years(frame, fold["test_years"])
    train_df = _drop_embargo_tail(train_df, prediction_days)
    validation_df = _drop_embargo_tail(validation_df, prediction_days)
    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("Walk-forward fold contains an empty split.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns].values)
    x_val = scaler.transform(validation_df[feature_columns].values)
    x_test = scaler.transform(test_df[feature_columns].values)
    metadata_builder = DataPreparator()

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": train_df["targetReturns"].values,
        "y_val": validation_df["targetReturns"].values,
        "y_test": test_df["targetReturns"].values,
        "split_metadata": {
            "train": metadata_builder._build_split_metadata(train_df),
            "val": metadata_builder._build_split_metadata(validation_df),
            "test": metadata_builder._build_split_metadata(test_df),
        },
        "split_date_ranges": {
            "train": _date_range_for_frame(train_df),
            "validation": _date_range_for_frame(validation_df),
            "test": _date_range_for_frame(test_df),
        },
        "scaler": scaler,
    }


def build_walk_forward_fold_report(
    fold,
    selection_report,
    basket_backtest_report,
    top_n_buckets=DEFAULT_WALK_FORWARD_TOP_N_BUCKETS,
):
    """Create the compact user-facing report for one walk-forward fold."""
    return {
        "fold_index": int(fold["fold_index"]),
        "train_date_range": fold["train_date_range"],
        "validation_date_range": fold["validation_date_range"],
        "test_date_range": fold["test_date_range"],
        "train_years": list(fold["train_years"]),
        "validation_years": list(fold["validation_years"]),
        "test_years": list(fold["test_years"]),
        "selected_candidate_id": selection_report.get("selected_candidate_id"),
        "selected_candidate_name": selection_report.get("selected_candidate_name"),
        "validation_selection_score": selection_report.get(
            "selected_validation_top_n_mean_excess_return"
        ),
        "top_n": {
            bucket_name: _walk_forward_bucket_metrics(
                basket_backtest_report,
                bucket_name,
            )
            for bucket_name in top_n_buckets
        },
        "basket_backtest": basket_backtest_report,
        "regressor_validation_selection": selection_report,
    }


def build_walk_forward_aggregate_summary(
    fold_reports,
    top_n_buckets=DEFAULT_WALK_FORWARD_TOP_N_BUCKETS,
):
    """Aggregate compact walk-forward metrics across folds."""
    selected_candidate_counts = Counter(
        fold_report.get("selected_candidate_name", "unknown")
        for fold_report in fold_reports
    )
    bucket_summary = {}

    for bucket_name in top_n_buckets:
        bucket_fold_metrics = [
            fold_report.get("top_n", {}).get(bucket_name, {})
            for fold_report in fold_reports
        ]
        model_excess_values = _metric_values(
            bucket_fold_metrics,
            "model_excess",
        )
        model_minus_momentum_values = _metric_values(
            bucket_fold_metrics,
            "model_minus_momentum",
        )
        model_minus_universe_values = _metric_values(
            bucket_fold_metrics,
            "model_minus_universe",
        )
        bucket_summary[bucket_name] = {
            "average_model_excess": _mean_or_nan(model_excess_values),
            "average_model_minus_momentum": _mean_or_nan(
                model_minus_momentum_values
            ),
            "average_model_minus_universe": _mean_or_nan(
                model_minus_universe_values
            ),
            "fold_win_rate_vs_momentum": _positive_rate_or_nan(
                model_minus_momentum_values
            ),
            "fold_win_rate_vs_universe": _positive_rate_or_nan(
                model_minus_universe_values
            ),
            "evaluated_folds": int(len(bucket_fold_metrics)),
        }

    return {
        "fold_count": int(len(fold_reports)),
        "top_n": bucket_summary,
        "selected_candidate_counts": dict(selected_candidate_counts),
    }


def _walk_forward_bucket_metrics(basket_backtest_report, bucket_name):
    """Extract compact fold metrics from one Top-N basket bucket."""
    bucket_report = basket_backtest_report.get(bucket_name, {})
    model = bucket_report.get("model", {})
    momentum = bucket_report.get("momentum_baseline", {})
    universe = bucket_report.get("universe", {})

    model_excess = model.get("average_basket_excess_return", np.nan)
    momentum_excess = momentum.get("average_basket_excess_return", np.nan)
    universe_excess = universe.get("average_basket_excess_return", np.nan)
    return {
        "model_excess": model_excess,
        "model_minus_momentum": _difference_or_nan(
            model_excess,
            momentum_excess,
        ),
        "model_minus_universe": _difference_or_nan(
            model_excess,
            universe_excess,
        ),
    }


def _frame_with_prediction_year(frame):
    """Return a copy with normalized prediction dates and integer years."""
    frame = frame.copy()
    frame["prediction_date"] = pd.to_datetime(
        frame["prediction_date"],
        errors="raise",
    ).dt.normalize()
    frame["_prediction_year"] = frame["prediction_date"].dt.year.astype(int)
    return frame


def _rows_for_years(frame, years):
    """Select rows whose prediction_date year is in years."""
    return frame.loc[frame["_prediction_year"].isin(years)].copy()


def _drop_embargo_tail(frame, prediction_days):
    """Drop trailing prediction dates whose forward target can cross a boundary."""
    prediction_days = int(prediction_days)
    if prediction_days <= 0 or frame.empty:
        return frame

    unique_dates = np.array(sorted(frame["prediction_date"].drop_duplicates()))
    if len(unique_dates) <= prediction_days:
        return frame.iloc[0:0].copy()

    keep_dates = set(unique_dates[:-prediction_days])
    return frame.loc[frame["prediction_date"].isin(keep_dates)].copy()


def _date_range_for_years(frame, years):
    """Return stable string date range metadata for a set of years."""
    rows = _rows_for_years(frame, years)
    if rows.empty:
        return {"start": None, "end": None}

    return {
        "start": rows["prediction_date"].min().strftime("%Y-%m-%d"),
        "end": rows["prediction_date"].max().strftime("%Y-%m-%d"),
    }


def _date_range_for_frame(frame):
    """Return stable string date range metadata for an already-filtered frame."""
    if frame.empty:
        return {"start": None, "end": None}

    return {
        "start": frame["prediction_date"].min().strftime("%Y-%m-%d"),
        "end": frame["prediction_date"].max().strftime("%Y-%m-%d"),
    }


def _metric_values(bucket_fold_metrics, metric_name):
    """Return numeric metric values across folds."""
    return np.asarray(
        [
            metrics.get(metric_name, np.nan)
            for metrics in bucket_fold_metrics
        ],
        dtype=float,
    )


def _positive_rate_or_nan(values):
    """Return the share of finite values greater than zero."""
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return np.nan

    return float(np.mean(finite_values > 0.0))


def _difference_or_nan(left, right):
    """Return left - right unless either side is unavailable."""
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return np.nan

    return float(left) - float(right)


def _validate_positive_year_count(value, argument_name):
    """Validate walk-forward year-count parameters."""
    if int(value) < 1:
        raise ValueError(f"{argument_name} must be at least 1.")
