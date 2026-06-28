"""Shared numeric helpers and input validation for evaluation reports."""

import numpy as np


def _mean_or_nan(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))

def _validate_probability_inputs(probability_up, actual_returns):
    if len(probability_up) == 0:
        raise ValueError("probability_up must contain at least one value.")
    if len(probability_up) != len(actual_returns):
        raise ValueError("probability_up and actual_returns must have the same length.")

def _validate_return_inputs(actual_returns, predicted_returns):
    if len(actual_returns) == 0:
        raise ValueError("actual_returns must contain at least one value.")
    if len(actual_returns) != len(predicted_returns):
        raise ValueError("actual_returns and predicted_returns must have the same length.")

def _up_rate_or_nan(returns):
    if len(returns) == 0:
        return np.nan
    return float(np.mean(np.asarray(returns) > 0))

def _tail_count(values, fraction):
    return max(1, int(np.ceil(len(values) * fraction)))

def _tail_stats(indices, actual_returns):
    selected_returns = actual_returns[indices]
    return {
        "count": int(len(indices)),
        "avg_actual_return": _mean_or_nan(selected_returns),
        "precision": _up_rate_or_nan(selected_returns),
    }

def _return_bucket_stats(indices, actual_returns, predicted_returns):
    selected_actual_returns = actual_returns[indices]
    selected_predicted_returns = predicted_returns[indices]
    return {
        "count": int(len(indices)),
        "avg_actual_return": _mean_or_nan(selected_actual_returns),
        "avg_predicted_return": _mean_or_nan(selected_predicted_returns),
        "precision": _up_rate_or_nan(selected_actual_returns),
    }

def _correlation_or_nan(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])

def _rank_values(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0

    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks

def _validate_ranked_selection_inputs(metadata, predicted_excess_returns):
    required_columns = {
        "Ticker",
        "prediction_date",
        "raw_forward_return",
        "benchmark_forward_return",
        "excess_forward_return",
        "beat_benchmark_target",
    }
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"split_metadata is missing required columns: {missing_columns}")
    if len(metadata) != len(predicted_excess_returns):
        raise ValueError(
            "split_metadata and predicted_excess_returns must have the same length."
        )
    duplicate_candidates = metadata.duplicated(
        subset=["prediction_date", "Ticker"],
        keep=False,
    )
    if duplicate_candidates.any():
        duplicates = (
            metadata.loc[duplicate_candidates, ["prediction_date", "Ticker"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            "split_metadata contains duplicate candidate rows for "
            f"(prediction_date, Ticker): {duplicates}"
        )
