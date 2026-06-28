"""Basic classification, regression, probability, and signal reports."""

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

from src.train.evaluation.validation import (
    _correlation_or_nan,
    _mean_or_nan,
    _rank_values,
    _return_bucket_stats,
    _tail_count,
    _tail_stats,
    _up_rate_or_nan,
    _validate_probability_inputs,
    _validate_return_inputs,
)


def build_classification_report(y_true, y_pred):
    """Measure beat-benchmark classification quality against simple baselines.

    Inputs are one-dimensional true and predicted labels, where 1 means the stock
    beat the benchmark and 0 means it did not.
    """
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
    """Measure SPY-relative excess-return error against simple baselines.

    y_true and y_pred are one-dimensional excess-return arrays. y_train is optional
    and is used only to build the train-mean excess-return baseline.
    """
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
    """Summarize excess returns for rows classified as beat or not beat benchmark.

    Empty groups return NaN for their averages, which means that side had no
    selected rows rather than zero excess return.
    """
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

def build_actual_return_baseline_report(actual_returns):
    """Summarize full test-set excess returns before signal filtering."""
    actual_returns = np.asarray(actual_returns, dtype=float)
    if len(actual_returns) == 0:
        raise ValueError("actual_returns must contain at least one value.")

    positive_return_rate = _up_rate_or_nan(actual_returns)
    return {
        "count": int(len(actual_returns)),
        "avg_actual_return": _mean_or_nan(actual_returns),
        "positive_return_rate": positive_return_rate,
        "precision": positive_return_rate,
    }

def build_probability_summary(probability_up):
    """Describe predicted probabilities of beating the benchmark.

    probability_up is the legacy parameter name for predict_proba(...)[..., 1],
    now interpreted as probability of beating SPY.
    """
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
    """Compare excess returns in highest/lowest beat-benchmark probability buckets.

    This tests whether classifier probabilities rank SPY-relative opportunities,
    even when the default hard class prediction is weak.
    """
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
    """Evaluate rows above each beat-benchmark probability threshold.

    Returns a report dictionary keyed by threshold.
    NaN precision or average excess return means a threshold selected no rows.
    """
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
    """Choose a beat-benchmark probability threshold on validation, then test it.

    This avoids tuning on the test set: validation selects the rule, and test is
    only the out-of-sample check of that already-selected rule.
    Returns validation candidate stats plus the selected threshold and selected
    test stats, or None fields when no threshold is eligible.
    """
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
    """Measure whether predicted excess returns rank SPY-relative outcomes.

    Ranking diagnostics are separate from MSE/MAE/R2 because a return model can
    rank opportunities usefully even when exact excess-return magnitudes are
    noisy. Returns top and bottom predicted-excess-return bucket stats.
    """
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
    """Report correlation between predicted and actual excess returns.

    Pearson measures excess-return fit; Spearman measures ranking signal.
    Returns both values in a small dictionary.
    """
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

def build_combined_signal_report(
    probability_up,
    actual_returns,
    predicted_returns,
    probability_threshold=0.60,
    top_return_fraction=0.20,
):
    """Evaluate rows selected by beat-benchmark probability and excess-return rank."""
    probability_up = np.asarray(probability_up, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    predicted_returns = np.asarray(predicted_returns, dtype=float)
    _validate_probability_inputs(probability_up, actual_returns)
    _validate_return_inputs(actual_returns, predicted_returns)

    probability_mask = probability_up >= probability_threshold
    order = np.argsort(predicted_returns)
    top_return_indices = order[-_tail_count(predicted_returns, top_return_fraction):]
    top_return_mask = np.zeros(len(predicted_returns), dtype=bool)
    top_return_mask[top_return_indices] = True

    selected = probability_mask & top_return_mask
    selected_actual_returns = actual_returns[selected]
    selected_predicted_returns = predicted_returns[selected]

    return {
        "probability_threshold": probability_threshold,
        "top_return_fraction": top_return_fraction,
        "selected_count": int(np.sum(selected)),
        "selected_fraction": float(np.mean(selected)),
        "avg_actual_return": _mean_or_nan(selected_actual_returns),
        "avg_predicted_return": _mean_or_nan(selected_predicted_returns),
        "precision": _up_rate_or_nan(selected_actual_returns),
    }
