from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
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


def build_top_n_ranked_selection_report(
    split_metadata,
    predicted_excess_returns,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    parallel_random_trials=False,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
):
    """Evaluate ranked stock selection within each prediction date.

    Rows are ranked by predicted SPY-relative excess return separately for each
    date, then the top min(N, available candidates) are aggregated across dates.
    Random and momentum baselines use the same dates and candidate rows.
    """
    return build_top_n_selection_reports(
        split_metadata,
        predicted_excess_returns,
        top_n_values=top_n_values,
        random_seed=random_seed,
        random_trials=random_trials,
        parallel_random_trials=parallel_random_trials,
        random_trial_workers=random_trial_workers,
        momentum_score_column=momentum_score_column,
        prediction_days=prediction_days,
    )["ranked_selection"]


def build_top_n_basket_backtest_report(
    split_metadata,
    ranked_predictions,
    top_ns=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    parallel_random_trials=False,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
):
    """Summarize equal-weight top-N baskets selected within each prediction date.

    This is a simple no-lookahead basket diagnostic: each prediction date
    contributes one equal-weight basket outcome per bucket. Returns are averaged
    across dates without compounding or overlapping-position modeling.
    """
    return build_top_n_selection_reports(
        split_metadata,
        ranked_predictions,
        top_n_values=top_ns,
        random_seed=random_seed,
        random_trials=random_trials,
        parallel_random_trials=parallel_random_trials,
        random_trial_workers=random_trial_workers,
        momentum_score_column=momentum_score_column,
        prediction_days=prediction_days,
    )["basket_backtest"]


def build_top_n_selection_reports(
    split_metadata,
    ranked_predictions,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    parallel_random_trials=False,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
):
    """Build ranked-selection and basket-backtest Top-N reports in one pass."""
    if random_trials < 1:
        raise ValueError("random_trials must be at least 1.")
    if random_trial_workers < 1:
        raise ValueError("random_trial_workers must be at least 1.")

    metadata = split_metadata.copy()
    ranked_predictions = np.asarray(ranked_predictions, dtype=float)
    _validate_ranked_selection_inputs(metadata, ranked_predictions)

    score_column = "predicted_excess_return"
    metadata[score_column] = ranked_predictions
    metadata = metadata[metadata["Ticker"] != "SPY"].copy()
    momentum_score_column = _resolve_momentum_score_column(
        metadata,
        momentum_score_column,
        prediction_days,
    )
    relative_momentum_score_column = _resolve_relative_momentum_score_column(
        metadata,
        prediction_days,
    )
    metadata_columns = set(metadata.columns)
    grouped_metadata = list(metadata.groupby("prediction_date", sort=True))
    ranked_selection_report = {}
    basket_backtest_report = {}
    universe_ranked_stats = _universe_ranked_selection_stats(grouped_metadata)
    universe_basket_stats = _basket_backtest_stats(
        [date_group for _, date_group in grouped_metadata]
    )
    benchmark_basket_stats = _benchmark_basket_backtest_stats(grouped_metadata)

    for top_n in top_n_values:
        key = f"top_{top_n}"
        selected_count_by_date = {
            prediction_date: min(int(top_n), len(date_group))
            for prediction_date, date_group in grouped_metadata
        }
        model_selected_groups = _select_top_n_by_score(
            grouped_metadata,
            selected_count_by_date,
            score_column,
        )
        random_ranked_stats, random_basket_stats = _random_top_n_selection_stats(
            grouped_metadata,
            selected_count_by_date,
            random_seed=random_seed,
            random_trials=random_trials,
            parallel_random_trials=parallel_random_trials,
            random_trial_workers=random_trial_workers,
        )
        momentum_selected_groups = _select_available_top_n_groups(
            grouped_metadata,
            selected_count_by_date,
            momentum_score_column,
            metadata_columns,
        )
        relative_momentum_selected_groups = _select_available_top_n_groups(
            grouped_metadata,
            selected_count_by_date,
            relative_momentum_score_column,
            metadata_columns,
        )

        ranked_selection_report[key] = {
            "model": _ranked_selection_stats(
                model_selected_groups,
                grouped_metadata,
                score_column=score_column,
            ),
            "random_baseline": random_ranked_stats,
            "momentum_baseline": _combined_momentum_ranked_selection_stats(
                momentum_selected_groups,
                grouped_metadata,
                momentum_score_column,
            ),
            "relative_momentum_baseline": (
                _combined_relative_momentum_ranked_selection_stats(
                    relative_momentum_selected_groups,
                    grouped_metadata,
                    relative_momentum_score_column,
                )
            ),
            "universe": universe_ranked_stats,
        }
        basket_backtest_report[key] = {
            "model": _basket_backtest_stats(model_selected_groups),
            "random_baseline": random_basket_stats,
            "momentum_baseline": _combined_momentum_basket_backtest_stats(
                momentum_selected_groups,
                momentum_score_column,
            ),
            "relative_momentum_baseline": _combined_relative_momentum_basket_backtest_stats(
                relative_momentum_selected_groups,
                relative_momentum_score_column,
            ),
            "universe": universe_basket_stats,
            "benchmark": benchmark_basket_stats,
        }

    return {
        "ranked_selection": ranked_selection_report,
        "basket_backtest": basket_backtest_report,
    }


def _select_top_n_by_score(grouped_metadata, selected_count_by_date, score_column):
    selected_groups = []

    for prediction_date, date_group in grouped_metadata:
        selected_count = selected_count_by_date[prediction_date]
        if selected_count == 0:
            continue
        selected_groups.append(date_group.nlargest(selected_count, score_column))

    return selected_groups


def _select_available_top_n_groups(
    grouped_metadata,
    selected_count_by_date,
    score_column,
    metadata_columns,
):
    if score_column not in metadata_columns:
        return None

    return _select_top_n_by_score(
        grouped_metadata,
        selected_count_by_date,
        score_column,
    )


def _random_top_n_selection_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
    parallel_random_trials=False,
    random_trial_workers=4,
):
    if parallel_random_trials and random_trials > 1:
        trial_seeds = _random_trial_seeds(random_seed, random_trials)
        worker_count = min(int(random_trial_workers), int(random_trials))
        seed_chunks = np.array_split(
            np.asarray(trial_seeds, dtype=np.uint64),
            worker_count,
        )
        tasks = [
            (
                grouped_metadata,
                selected_count_by_date,
                [int(seed) for seed in seed_chunk],
            )
            for seed_chunk in seed_chunks
            if len(seed_chunk) > 0
        ]
        ranked_trial_stats = []
        basket_trial_stats = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for ranked_chunk, basket_chunk in executor.map(
                _random_top_n_selection_stats_worker,
                tasks,
            ):
                ranked_trial_stats.extend(ranked_chunk)
                basket_trial_stats.extend(basket_chunk)
    else:
        # Preserve the legacy default path: one RNG seeded once, consumed across
        # dates and trials. With one trial, explicit parallel mode uses this
        # no-pool path as well.
        ranked_trial_stats, basket_trial_stats = (
            _sequential_random_top_n_selection_trial_stats(
                grouped_metadata,
                selected_count_by_date,
                random_seed,
                random_trials,
            )
        )

    ranked_stats = _average_random_trial_stats(
        ranked_trial_stats,
        random_seed,
        random_trials,
    )
    basket_stats = _average_random_basket_trial_stats(basket_trial_stats)
    basket_stats["random_seed"] = int(random_seed)
    basket_stats["random_trials"] = int(random_trials)
    return ranked_stats, basket_stats


def _sequential_random_top_n_selection_trial_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
):
    rng = np.random.default_rng(random_seed)
    ranked_trial_stats = []
    basket_trial_stats = []

    for _ in range(random_trials):
        selected_groups = []
        for prediction_date, date_group in grouped_metadata:
            selected_count = selected_count_by_date[prediction_date]
            if selected_count == 0:
                continue

            selected_positions = rng.choice(
                len(date_group),
                size=selected_count,
                replace=False,
            )
            selected_groups.append(date_group.iloc[selected_positions])

        ranked_trial_stats.append(
            _ranked_selection_stats(selected_groups, grouped_metadata)
        )
        basket_trial_stats.append(_basket_backtest_stats(selected_groups))

    return ranked_trial_stats, basket_trial_stats


def _random_trial_seeds(random_seed, random_trials):
    rng = np.random.default_rng(random_seed)
    return [
        int(seed)
        for seed in rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=int(random_trials),
            dtype=np.uint32,
        )
    ]


def _random_top_n_selection_stats_worker(args):
    grouped_metadata, selected_count_by_date, trial_seeds = args
    ranked_trial_stats = []
    basket_trial_stats = []

    for trial_seed in trial_seeds:
        selected_groups = _select_random_trial_groups(
            grouped_metadata,
            selected_count_by_date,
            trial_seed,
        )
        ranked_trial_stats.append(
            _ranked_selection_stats(selected_groups, grouped_metadata)
        )
        basket_trial_stats.append(_basket_backtest_stats(selected_groups))

    return ranked_trial_stats, basket_trial_stats


def _select_random_trial_groups(
    grouped_metadata,
    selected_count_by_date,
    trial_seed,
):
    rng = np.random.default_rng(trial_seed)
    selected_groups = []
    for prediction_date, date_group in grouped_metadata:
        selected_count = selected_count_by_date[prediction_date]
        if selected_count == 0:
            continue

        selected_positions = rng.choice(
            len(date_group),
            size=selected_count,
            replace=False,
        )
        selected_groups.append(date_group.iloc[selected_positions])

    return selected_groups


def _combined_momentum_ranked_selection_stats(
    selected_groups,
    grouped_metadata,
    momentum_score_column,
):
    if selected_groups is None:
        stats = _empty_ranked_selection_stats(score_column=momentum_score_column)
        stats["available"] = False
        stats["momentum_score_column"] = momentum_score_column
        stats["reason"] = (
            f"Momentum score column '{momentum_score_column}' is not present in split_metadata."
        )
        return stats

    stats = _ranked_selection_stats(
        selected_groups,
        grouped_metadata,
        score_column=momentum_score_column,
    )
    stats["available"] = True
    stats["momentum_score_column"] = momentum_score_column
    return stats


def _combined_relative_momentum_ranked_selection_stats(
    selected_groups,
    grouped_metadata,
    relative_momentum_score_column,
):
    if selected_groups is None:
        stats = _empty_ranked_selection_stats(
            score_column=relative_momentum_score_column
        )
        stats["available"] = False
        stats["relative_momentum_score_column"] = relative_momentum_score_column
        stats["reason"] = (
            f"Relative momentum score column '{relative_momentum_score_column}' "
            "is not present in split_metadata."
        )
        return stats

    stats = _ranked_selection_stats(
        selected_groups,
        grouped_metadata,
        score_column=relative_momentum_score_column,
    )
    stats["available"] = True
    stats["relative_momentum_score_column"] = relative_momentum_score_column
    return stats


def _combined_momentum_basket_backtest_stats(
    selected_groups,
    momentum_score_column,
):
    if selected_groups is None:
        stats = _empty_basket_backtest_stats()
        stats["available"] = False
        stats["momentum_score_column"] = momentum_score_column
        stats["reason"] = (
            f"Momentum score column '{momentum_score_column}' is not present in split_metadata."
        )
        return stats

    stats = _basket_backtest_stats(selected_groups)
    stats["available"] = True
    stats["momentum_score_column"] = momentum_score_column
    return stats


def _combined_relative_momentum_basket_backtest_stats(
    selected_groups,
    relative_momentum_score_column,
):
    if selected_groups is None:
        stats = _empty_basket_backtest_stats()
        stats["available"] = False
        stats["relative_momentum_score_column"] = relative_momentum_score_column
        stats["reason"] = (
            f"Relative momentum score column '{relative_momentum_score_column}' "
            "is not present in split_metadata."
        )
        return stats

    stats = _basket_backtest_stats(selected_groups)
    stats["available"] = True
    stats["relative_momentum_score_column"] = relative_momentum_score_column
    return stats


def _resolve_momentum_score_column(metadata, momentum_score_column, prediction_days):
    if momentum_score_column is not None:
        return momentum_score_column

    if prediction_days is not None:
        horizon_column = f"momentum_{int(prediction_days)}d"
        if horizon_column in metadata.columns:
            return horizon_column

    return "dailyReturn"


def _resolve_relative_momentum_score_column(metadata, prediction_days):
    if prediction_days is None:
        return "relative_momentum"

    return f"relative_momentum_{int(prediction_days)}d"


def _basket_backtest_stats(selected_groups):
    if not selected_groups:
        return _empty_basket_backtest_stats()

    date_stats = pd.DataFrame(
        [
            {
                "basket_raw_return": float(selected_group["raw_forward_return"].mean()),
                "basket_benchmark_return": float(
                    selected_group["benchmark_forward_return"].mean()
                ),
                "basket_excess_return": float(
                    selected_group["excess_forward_return"].mean()
                ),
                "selected_count": len(selected_group),
            }
            for selected_group in selected_groups
        ]
    )

    return _summarize_basket_date_stats(date_stats)


def _random_basket_backtest_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
):
    rng = np.random.default_rng(random_seed)
    trial_stats = []

    for _ in range(random_trials):
        selected_groups = []
        for prediction_date, date_group in grouped_metadata:
            selected_count = selected_count_by_date[prediction_date]
            if selected_count == 0:
                continue

            selected_positions = rng.choice(
                len(date_group),
                size=selected_count,
                replace=False,
            )
            selected_groups.append(date_group.iloc[selected_positions])

        trial_stats.append(_basket_backtest_stats(selected_groups))

    stats = _average_random_basket_trial_stats(trial_stats)
    stats["random_seed"] = int(random_seed)
    stats["random_trials"] = int(random_trials)
    return stats


def _average_random_basket_trial_stats(trial_stats):
    if not trial_stats:
        return _empty_basket_backtest_stats()

    stats = {}
    for metric in trial_stats[0]:
        values = np.asarray([trial[metric] for trial in trial_stats], dtype=float)
        stats[metric] = _mean_or_nan(values)

    stats["evaluated_dates"] = int(round(stats["evaluated_dates"]))
    return stats


def _momentum_basket_backtest_stats(
    grouped_metadata,
    selected_count_by_date,
    momentum_score_column,
):
    metadata_columns = set()
    for _, date_group in grouped_metadata:
        metadata_columns.update(date_group.columns)

    if momentum_score_column not in metadata_columns:
        stats = _empty_basket_backtest_stats()
        stats["available"] = False
        stats["momentum_score_column"] = momentum_score_column
        stats["reason"] = (
            f"Momentum score column '{momentum_score_column}' is not present in split_metadata."
        )
        return stats

    selected_groups = _select_top_n_by_score(
        grouped_metadata,
        selected_count_by_date,
        momentum_score_column,
    )
    stats = _basket_backtest_stats(selected_groups)
    stats["available"] = True
    stats["momentum_score_column"] = momentum_score_column
    return stats


def _relative_momentum_basket_backtest_stats(
    grouped_metadata,
    selected_count_by_date,
    relative_momentum_score_column,
):
    metadata_columns = set()
    for _, date_group in grouped_metadata:
        metadata_columns.update(date_group.columns)

    if relative_momentum_score_column not in metadata_columns:
        stats = _empty_basket_backtest_stats()
        stats["available"] = False
        stats["relative_momentum_score_column"] = relative_momentum_score_column
        stats["reason"] = (
            f"Relative momentum score column '{relative_momentum_score_column}' "
            "is not present in split_metadata."
        )
        return stats

    selected_groups = _select_top_n_by_score(
        grouped_metadata,
        selected_count_by_date,
        relative_momentum_score_column,
    )
    stats = _basket_backtest_stats(selected_groups)
    stats["available"] = True
    stats["relative_momentum_score_column"] = relative_momentum_score_column
    return stats


def _benchmark_basket_backtest_stats(grouped_metadata):
    if not grouped_metadata:
        return _empty_basket_backtest_stats()

    date_stats = pd.DataFrame(
        [
            {
                "basket_raw_return": float(
                    date_group["benchmark_forward_return"].mean()
                ),
                "basket_benchmark_return": float(
                    date_group["benchmark_forward_return"].mean()
                ),
                "basket_excess_return": 0.0,
                "selected_count": 1,
            }
            for _, date_group in grouped_metadata
        ]
    )
    return _summarize_basket_date_stats(date_stats)


def _summarize_basket_date_stats(date_stats):
    basket_raw_returns = date_stats["basket_raw_return"].to_numpy()
    basket_excess_returns = date_stats["basket_excess_return"].to_numpy()

    return {
        "average_basket_raw_return": _mean_or_nan(basket_raw_returns),
        "average_basket_benchmark_return": _mean_or_nan(
            date_stats["basket_benchmark_return"].to_numpy()
        ),
        "average_basket_excess_return": _mean_or_nan(basket_excess_returns),
        "positive_basket_return_rate": _up_rate_or_nan(basket_raw_returns),
        "beat_benchmark_rate": _up_rate_or_nan(basket_excess_returns),
        "average_selected_count": _mean_or_nan(
            date_stats["selected_count"].to_numpy()
        ),
        "evaluated_dates": int(len(date_stats)),
    }


def _empty_basket_backtest_stats():
    return {
        "average_basket_raw_return": np.nan,
        "average_basket_benchmark_return": np.nan,
        "average_basket_excess_return": np.nan,
        "positive_basket_return_rate": np.nan,
        "beat_benchmark_rate": np.nan,
        "average_selected_count": np.nan,
        "evaluated_dates": 0,
    }


def _ranked_selection_stats(selected_groups, grouped_metadata, score_column=None):
    if not selected_groups:
        return _empty_ranked_selection_stats(score_column=score_column)

    selected = pd.concat(selected_groups, ignore_index=True)
    selected_by_date = {
        prediction_date: selected_group
        for prediction_date, selected_group in zip(
            [prediction_date for prediction_date, _ in grouped_metadata],
            selected_groups,
        )
    }
    date_stats = []
    candidate_counts = []

    for prediction_date, date_group in grouped_metadata:
        if prediction_date not in selected_by_date:
            continue

        selected_group = selected_by_date[prediction_date]
        selected_raw_return = float(selected_group["raw_forward_return"].mean())
        universe_return = float(date_group["raw_forward_return"].mean())
        date_stat = {
            "selected_raw_forward_return": selected_raw_return,
            "selected_benchmark_forward_return": float(
                selected_group["benchmark_forward_return"].mean()
            ),
            "selected_excess_return_vs_benchmark": float(
                selected_group["excess_forward_return"].mean()
            ),
            "equal_weight_universe_forward_return": universe_return,
            "selected_return_minus_universe_return": (
                selected_raw_return - universe_return
            ),
        }
        if score_column is not None:
            date_stat[score_column] = float(selected_group[score_column].mean())

        candidate_counts.append(len(date_group))
        date_stats.append(date_stat)

    date_stats = pd.DataFrame(date_stats)
    stats = {
        "date_count": int(len(selected_groups)),
        "selected_row_count": int(len(selected)),
        "average_selected_raw_forward_return": _mean_or_nan(
            date_stats["selected_raw_forward_return"].to_numpy()
        ),
        "average_selected_benchmark_forward_return": _mean_or_nan(
            date_stats["selected_benchmark_forward_return"].to_numpy()
        ),
        "average_selected_excess_return_vs_benchmark": _mean_or_nan(
            date_stats["selected_excess_return_vs_benchmark"].to_numpy()
        ),
        "beat_benchmark_rate": _mean_or_nan(
            selected["beat_benchmark_target"].to_numpy()
        ),
        "average_equal_weight_universe_forward_return": _mean_or_nan(
            date_stats["equal_weight_universe_forward_return"].to_numpy()
        ),
        "average_selected_return_minus_universe_return": _mean_or_nan(
            date_stats["selected_return_minus_universe_return"].to_numpy()
        ),
        "median_selected_excess_return_vs_benchmark": float(
            np.median(selected["excess_forward_return"].to_numpy())
        ),
        "positive_raw_return_rate": _up_rate_or_nan(
            selected["raw_forward_return"].to_numpy()
        ),
        "average_number_of_candidates_per_date": _mean_or_nan(
            np.asarray(candidate_counts, dtype=float)
        ),
    }
    if score_column is not None:
        stats[f"average_{score_column}"] = _mean_or_nan(
            date_stats[score_column].to_numpy()
        )

    return stats


def _random_ranked_selection_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
):
    rng = np.random.default_rng(random_seed)
    trial_stats = []

    for _ in range(random_trials):
        selected_groups = []
        for prediction_date, date_group in grouped_metadata:
            selected_count = selected_count_by_date[prediction_date]
            if selected_count == 0:
                continue

            selected_positions = rng.choice(
                len(date_group),
                size=selected_count,
                replace=False,
            )
            selected_groups.append(date_group.iloc[selected_positions])

        trial_stats.append(_ranked_selection_stats(selected_groups, grouped_metadata))

    return _average_random_trial_stats(trial_stats, random_seed, random_trials)


def _average_random_trial_stats(trial_stats, random_seed, random_trials):
    if not trial_stats:
        stats = _empty_ranked_selection_stats()
    else:
        stats = {}
        for metric in trial_stats[0]:
            values = np.asarray([trial[metric] for trial in trial_stats], dtype=float)
            stats[metric] = _mean_or_nan(values)

        for count_metric in ["date_count", "selected_row_count"]:
            stats[count_metric] = int(round(stats[count_metric]))

    stats["random_baseline_trials"] = int(random_trials)
    stats["random_seed"] = int(random_seed)
    return stats


def _momentum_ranked_selection_stats(
    grouped_metadata,
    selected_count_by_date,
    momentum_score_column,
):
    metadata_columns = set()
    for _, date_group in grouped_metadata:
        metadata_columns.update(date_group.columns)

    if momentum_score_column not in metadata_columns:
        stats = _empty_ranked_selection_stats(score_column=momentum_score_column)
        stats["available"] = False
        stats["momentum_score_column"] = momentum_score_column
        stats["reason"] = (
            f"Momentum score column '{momentum_score_column}' is not present in split_metadata."
        )
        return stats

    selected_groups = _select_top_n_by_score(
        grouped_metadata,
        selected_count_by_date,
        momentum_score_column,
    )
    stats = _ranked_selection_stats(
        selected_groups,
        grouped_metadata,
        score_column=momentum_score_column,
    )
    stats["available"] = True
    stats["momentum_score_column"] = momentum_score_column
    return stats


def _relative_momentum_ranked_selection_stats(
    grouped_metadata,
    selected_count_by_date,
    relative_momentum_score_column,
):
    metadata_columns = set()
    for _, date_group in grouped_metadata:
        metadata_columns.update(date_group.columns)

    if relative_momentum_score_column not in metadata_columns:
        stats = _empty_ranked_selection_stats(
            score_column=relative_momentum_score_column
        )
        stats["available"] = False
        stats["relative_momentum_score_column"] = relative_momentum_score_column
        stats["reason"] = (
            f"Relative momentum score column '{relative_momentum_score_column}' "
            "is not present in split_metadata."
        )
        return stats

    selected_groups = _select_top_n_by_score(
        grouped_metadata,
        selected_count_by_date,
        relative_momentum_score_column,
    )
    stats = _ranked_selection_stats(
        selected_groups,
        grouped_metadata,
        score_column=relative_momentum_score_column,
    )
    stats["available"] = True
    stats["relative_momentum_score_column"] = relative_momentum_score_column
    return stats


def _universe_ranked_selection_stats(grouped_metadata):
    if not grouped_metadata:
        return {
            "date_count": 0,
            "candidate_row_count": 0,
            "average_equal_weight_universe_forward_return": np.nan,
            "average_benchmark_forward_return": np.nan,
            "average_excess_return_vs_benchmark": np.nan,
            "beat_benchmark_rate": np.nan,
            "positive_raw_return_rate": np.nan,
            "average_number_of_candidates_per_date": np.nan,
        }

    rows = [date_group for _, date_group in grouped_metadata]
    metadata = pd.concat(rows, ignore_index=True)
    date_stats = pd.DataFrame(
        [
            {
                "raw_forward_return": float(date_group["raw_forward_return"].mean()),
                "benchmark_forward_return": float(
                    date_group["benchmark_forward_return"].mean()
                ),
                "excess_forward_return": float(
                    date_group["excess_forward_return"].mean()
                ),
                "candidate_count": len(date_group),
            }
            for _, date_group in grouped_metadata
        ]
    )
    return {
        "date_count": int(len(grouped_metadata)),
        "candidate_row_count": int(len(metadata)),
        "average_equal_weight_universe_forward_return": _mean_or_nan(
            date_stats["raw_forward_return"].to_numpy()
        ),
        "average_benchmark_forward_return": _mean_or_nan(
            date_stats["benchmark_forward_return"].to_numpy()
        ),
        "average_excess_return_vs_benchmark": _mean_or_nan(
            date_stats["excess_forward_return"].to_numpy()
        ),
        "beat_benchmark_rate": _mean_or_nan(
            metadata["beat_benchmark_target"].to_numpy()
        ),
        "positive_raw_return_rate": _up_rate_or_nan(
            metadata["raw_forward_return"].to_numpy()
        ),
        "average_number_of_candidates_per_date": _mean_or_nan(
            date_stats["candidate_count"].to_numpy()
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


def _empty_ranked_selection_stats(score_column=None):
    stats = {
        "date_count": 0,
        "selected_row_count": 0,
        "average_selected_raw_forward_return": np.nan,
        "average_selected_benchmark_forward_return": np.nan,
        "average_selected_excess_return_vs_benchmark": np.nan,
        "beat_benchmark_rate": np.nan,
        "average_equal_weight_universe_forward_return": np.nan,
        "average_selected_return_minus_universe_return": np.nan,
        "average_predicted_excess_return": np.nan,
        "median_selected_excess_return_vs_benchmark": np.nan,
        "positive_raw_return_rate": np.nan,
        "average_number_of_candidates_per_date": np.nan,
    }
    if score_column is not None:
        stats[f"average_{score_column}"] = np.nan

    return stats
