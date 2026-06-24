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


def _mean_or_nan(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))
