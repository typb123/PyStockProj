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


def _mean_or_nan(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def _validate_probability_inputs(probability_up, actual_returns):
    if len(probability_up) == 0:
        raise ValueError("probability_up must contain at least one value.")
    if len(probability_up) != len(actual_returns):
        raise ValueError("probability_up and actual_returns must have the same length.")


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
