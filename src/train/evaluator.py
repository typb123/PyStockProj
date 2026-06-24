import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def build_classification_report(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "always_up_accuracy": accuracy_score(y_true, np.ones_like(y_true)),
        "always_down_accuracy": accuracy_score(y_true, np.zeros_like(y_true)),
        "predicted_up_count": int(np.sum(y_pred == 1)),
        "predicted_down_count": int(np.sum(y_pred == 0)),
    }


def build_regression_report(y_true, y_pred, y_train=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    zero_pred = np.zeros_like(y_true)

    report = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "zero_return_baseline_mse": mean_squared_error(y_true, zero_pred),
        "zero_return_baseline_mae": mean_absolute_error(y_true, zero_pred),
    }

    if y_train is not None:
        train_mean = float(np.mean(y_train))
        mean_train_pred = np.full_like(y_true, train_mean)
        report["mean_train_return"] = train_mean
        report["mean_train_return_baseline_mse"] = mean_squared_error(
            y_true, mean_train_pred
        )
        report["mean_train_return_baseline_mae"] = mean_absolute_error(
            y_true, mean_train_pred
        )

    return report


def build_trading_relevance_report(actual_returns, predicted_returns, predicted_direction):
    actual_returns = np.asarray(actual_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    predicted_direction = np.asarray(predicted_direction).astype(int)

    predicted_up = predicted_direction == 1
    predicted_down = predicted_direction == 0

    return {
        "avg_actual_return_when_predicted_up": _mean_or_nan(
            actual_returns[predicted_up]
        ),
        "avg_actual_return_when_predicted_down": _mean_or_nan(
            actual_returns[predicted_down]
        ),
        "avg_predicted_return_when_predicted_up": _mean_or_nan(
            predicted_returns[predicted_up]
        ),
        "avg_predicted_return_when_predicted_down": _mean_or_nan(
            predicted_returns[predicted_down]
        ),
    }


def build_probability_summary(probability_up):
    probability_up = np.asarray(probability_up, dtype=float)
    if len(probability_up) == 0:
        raise ValueError("probability_up must contain at least one value.")

    return {
        "count": int(len(probability_up)),
        "min": float(np.min(probability_up)),
        "max": float(np.max(probability_up)),
        "mean": float(np.mean(probability_up)),
        "median": float(np.median(probability_up)),
        "std": float(np.std(probability_up)),
        "p10": float(np.percentile(probability_up, 10)),
        "p25": float(np.percentile(probability_up, 25)),
        "p75": float(np.percentile(probability_up, 75)),
        "p90": float(np.percentile(probability_up, 90)),
    }


def build_probability_tail_report(probability_up, actual_returns):
    probability_up = np.asarray(probability_up, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    _validate_probability_inputs(probability_up, actual_returns)
    order = np.argsort(probability_up)

    return {
        "top_10_pct": _tail_stats(order[-_tail_count(probability_up, 0.10):], actual_returns),
        "top_20_pct": _tail_stats(order[-_tail_count(probability_up, 0.20):], actual_returns),
        "bottom_10_pct": _tail_stats(order[:_tail_count(probability_up, 0.10)], actual_returns),
        "bottom_20_pct": _tail_stats(order[:_tail_count(probability_up, 0.20)], actual_returns),
    }


def build_probability_threshold_report(
    probability_up,
    actual_returns,
    thresholds=(0.50, 0.55, 0.60, 0.65, 0.70),
):
    probability_up = np.asarray(probability_up, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    _validate_probability_inputs(probability_up, actual_returns)
    report = {}

    for threshold in thresholds:
        selected = probability_up >= threshold
        selected_returns = actual_returns[selected]
        report[threshold] = {
            "selected_count": int(np.sum(selected)),
            "selected_fraction": float(np.mean(selected)),
            "precision": _up_rate_or_nan(selected_returns),
            "avg_actual_return": _mean_or_nan(selected_returns),
        }

    return report


def build_validation_selected_threshold_report(
    validation_probability_up,
    validation_actual_returns,
    test_probability_up,
    test_actual_returns,
    thresholds=(0.50, 0.55, 0.60, 0.65, 0.70),
    min_selected_count=25,
    selection_metric="avg_actual_return",
):
    if min_selected_count < 1:
        raise ValueError("min_selected_count must be at least 1.")
    if selection_metric != "avg_actual_return":
        raise ValueError("Unsupported selection_metric. Use 'avg_actual_return'.")

    thresholds = tuple(thresholds)

    # Select thresholds on validation only so test stays out-of-sample.
    validation_threshold_report = build_probability_threshold_report(
        validation_probability_up,
        validation_actual_returns,
        thresholds=thresholds,
    )
    # Validate test inputs even when no validation threshold is eligible.
    build_probability_threshold_report(
        test_probability_up,
        test_actual_returns,
        thresholds=thresholds,
    )

    # Ignore tiny validation tails because they are often just noise.
    eligible_thresholds = [
        threshold
        for threshold, stats in validation_threshold_report.items()
        if stats["selected_count"] >= min_selected_count
    ]

    selected_threshold = None
    selected_validation_stats = None
    selected_test_stats = None

    if eligible_thresholds:
        selected_threshold = max(
            eligible_thresholds,
            key=lambda threshold: validation_threshold_report[threshold][selection_metric],
        )
        selected_validation_stats = validation_threshold_report[selected_threshold]
        # Apply the selected rule to test once; test results must not affect selection.
        selected_test_stats = build_probability_threshold_report(
            test_probability_up,
            test_actual_returns,
            thresholds=(selected_threshold,),
        )[selected_threshold]

    return {
        "thresholds": thresholds,
        "min_selected_count": min_selected_count,
        "selection_metric": selection_metric,
        "validation_threshold_report": validation_threshold_report,
        "selected_threshold": selected_threshold,
        "selected_validation_stats": selected_validation_stats,
        "selected_test_stats": selected_test_stats,
    }


def build_predicted_return_quantile_report(actual_returns, predicted_returns):
    actual_returns = np.asarray(actual_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    _validate_return_inputs(actual_returns, predicted_returns)
    order = np.argsort(predicted_returns)

    return {
        "top_10_pct": _return_bucket_stats(
            order[-_tail_count(predicted_returns, 0.10):],
            actual_returns,
            predicted_returns,
        ),
        "top_20_pct": _return_bucket_stats(
            order[-_tail_count(predicted_returns, 0.20):],
            actual_returns,
            predicted_returns,
        ),
        "bottom_10_pct": _return_bucket_stats(
            order[:_tail_count(predicted_returns, 0.10)],
            actual_returns,
            predicted_returns,
        ),
        "bottom_20_pct": _return_bucket_stats(
            order[:_tail_count(predicted_returns, 0.20)],
            actual_returns,
            predicted_returns,
        ),
    }


def build_return_correlation_report(actual_returns, predicted_returns):
    actual_returns = np.asarray(actual_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    _validate_return_inputs(actual_returns, predicted_returns)

    return {
        "pearson": _correlation_or_nan(actual_returns, predicted_returns),
        "spearman": _correlation_or_nan(
            _rank_values(actual_returns),
            _rank_values(predicted_returns),
        ),
    }


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
