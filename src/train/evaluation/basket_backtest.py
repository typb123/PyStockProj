"""Basket backtest helpers for equal-weight Top-N reports."""

import numpy as np
import pandas as pd

from src.train.evaluation.validation import _mean_or_nan, _up_rate_or_nan


def _basket_backtest_stats(selected_groups):
    """Summarize equal-weight basket returns across selected prediction dates."""
    if not selected_groups:
        return _empty_basket_backtest_stats()

    return _summarize_basket_date_stats(_basket_backtest_date_stats(selected_groups))


def _bootstrap_basket_confidence_intervals(
    model_date_stats,
    momentum_date_stats=None,
    relative_momentum_date_stats=None,
    universe_date_stats=None,
    bootstrap_trials=500,
    bootstrap_seed=42,
    confidence_level=0.95,
):
    """Bootstrap Top-N basket excess returns by prediction_date."""
    if bootstrap_trials < 1:
        raise ValueError("bootstrap_trials must be at least 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    model_date_stats = _prepare_bootstrap_date_stats(model_date_stats)
    if len(model_date_stats) == 0:
        return _empty_bootstrap_confidence_intervals(
            bootstrap_trials,
            confidence_level,
            reason="No prediction dates are available for bootstrap resampling.",
        )

    bootstrap_data = pd.DataFrame(
        {
            "prediction_date": model_date_stats["prediction_date"],
            "model": model_date_stats["basket_excess_return"].astype(float),
        }
    )
    optional_sources = {
        "momentum_baseline": momentum_date_stats,
        "relative_momentum_baseline": relative_momentum_date_stats,
        "universe": universe_date_stats,
    }
    for source_name, source_date_stats in optional_sources.items():
        prepared_source = _prepare_bootstrap_date_stats(source_date_stats)
        if len(prepared_source) == 0:
            continue

        bootstrap_data = bootstrap_data.merge(
            prepared_source[["prediction_date", "basket_excess_return"]].rename(
                columns={"basket_excess_return": source_name}
            ),
            on="prediction_date",
            how="left",
        )

    bootstrap_data = bootstrap_data.sort_values("prediction_date").reset_index(
        drop=True
    )
    alpha = (1.0 - confidence_level) / 2.0
    rng = np.random.default_rng(bootstrap_seed)
    report = {}

    for source_name in [
        "model",
        "momentum_baseline",
        "relative_momentum_baseline",
        "universe",
    ]:
        metric_name = f"{source_name}_average_basket_excess_return"
        if source_name not in bootstrap_data.columns:
            report[metric_name] = _unavailable_bootstrap_interval(
                bootstrap_trials,
                confidence_level,
                reason=f"{source_name} date-level basket stats are unavailable.",
            )
            continue

        report[metric_name] = _bootstrap_mean_interval(
            bootstrap_data[[source_name]].dropna(),
            source_name,
            rng,
            bootstrap_trials,
            confidence_level,
            alpha,
        )

    difference_pairs = [
        ("momentum_baseline", "model_minus_momentum_baseline"),
        (
            "relative_momentum_baseline",
            "model_minus_relative_momentum_baseline",
        ),
        ("universe", "model_minus_universe"),
    ]
    for baseline_name, metric_prefix in difference_pairs:
        metric_name = f"{metric_prefix}_average_basket_excess_return"
        if baseline_name not in bootstrap_data.columns:
            report[metric_name] = _unavailable_bootstrap_interval(
                bootstrap_trials,
                confidence_level,
                reason=f"{baseline_name} date-level basket stats are unavailable.",
            )
            continue

        pair_data = bootstrap_data[["model", baseline_name]].dropna()
        if len(pair_data) == 0:
            report[metric_name] = _unavailable_bootstrap_interval(
                bootstrap_trials,
                confidence_level,
                reason=(
                    f"No overlapping prediction dates are available for model and "
                    f"{baseline_name}."
                ),
            )
            continue

        diff_data = pd.DataFrame(
            {metric_prefix: pair_data["model"] - pair_data[baseline_name]}
        )
        report[metric_name] = _bootstrap_mean_interval(
            diff_data,
            metric_prefix,
            rng,
            bootstrap_trials,
            confidence_level,
            alpha,
        )

    return report


def _prepare_bootstrap_date_stats(date_stats):
    """Normalize per-date basket stats for bootstrap merging."""
    if date_stats is None or len(date_stats) == 0:
        return pd.DataFrame(columns=["prediction_date", "basket_excess_return"])

    prepared = date_stats.copy()
    prepared["prediction_date"] = pd.to_datetime(
        prepared["prediction_date"],
        errors="raise",
    )
    return prepared

def _basket_backtest_date_stats(selected_groups):
    """Build per-date equal-weight basket outcomes for selected groups."""
    if not selected_groups:
        return pd.DataFrame(
            columns=[
                "prediction_date",
                "prediction_year",
                "basket_raw_return",
                "basket_benchmark_return",
                "basket_excess_return",
                "selected_count",
            ]
        )

    date_stats = pd.DataFrame(
        [
            _basket_backtest_date_stat(selected_group)
            for selected_group in selected_groups
        ]
    )
    return date_stats

def _basket_backtest_date_stat(selected_group):
    """Build one selected date group's equal-weight basket outcome."""
    prediction_date = pd.to_datetime(
        selected_group["prediction_date"].iloc[0],
        errors="raise",
    )
    return {
        "prediction_date": prediction_date,
        "prediction_year": str(prediction_date.year),
        "basket_raw_return": float(selected_group["raw_forward_return"].mean()),
        "basket_benchmark_return": float(
            selected_group["benchmark_forward_return"].mean()
        ),
        "basket_excess_return": float(selected_group["excess_forward_return"].mean()),
        "selected_count": len(selected_group),
    }

def _summarize_basket_date_stats(date_stats):
    """Convert per-date basket outcomes into basket backtest report metrics."""
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


def _bootstrap_mean_interval(
    date_values,
    value_column,
    rng,
    bootstrap_trials,
    confidence_level,
    alpha,
):
    """Build one percentile bootstrap interval over date-level values."""
    values = date_values[value_column].to_numpy(dtype=float)
    if len(values) == 0:
        return _unavailable_bootstrap_interval(
            bootstrap_trials,
            confidence_level,
            reason="No prediction dates are available for bootstrap resampling.",
        )

    sampled_positions = rng.integers(
        0,
        len(values),
        size=(int(bootstrap_trials), len(values)),
    )
    bootstrap_means = values[sampled_positions].mean(axis=1)
    interval = {
        "mean": float(np.mean(values)),
        "ci_lower": float(np.quantile(bootstrap_means, alpha)),
        "ci_upper": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "bootstrap_trials": int(bootstrap_trials),
        "confidence_level": float(confidence_level),
        "resampled_dates": int(len(values)),
        "available": True,
    }
    if len(values) < 2:
        interval["note"] = (
            "Fewer than two prediction dates; percentile interval is degenerate."
        )
    return interval


def _empty_bootstrap_confidence_intervals(
    bootstrap_trials,
    confidence_level,
    reason,
):
    """Return the empty bootstrap CI shape for all supported metrics."""
    return {
        metric_name: _unavailable_bootstrap_interval(
            bootstrap_trials,
            confidence_level,
            reason=reason,
        )
        for metric_name in [
            "model_average_basket_excess_return",
            "momentum_baseline_average_basket_excess_return",
            "relative_momentum_baseline_average_basket_excess_return",
            "universe_average_basket_excess_return",
            "model_minus_momentum_baseline_average_basket_excess_return",
            "model_minus_relative_momentum_baseline_average_basket_excess_return",
            "model_minus_universe_average_basket_excess_return",
        ]
    }


def _unavailable_bootstrap_interval(
    bootstrap_trials,
    confidence_level,
    reason,
):
    """Return an explicit unavailable CI entry."""
    return {
        "mean": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "bootstrap_trials": int(bootstrap_trials),
        "confidence_level": float(confidence_level),
        "resampled_dates": 0,
        "available": False,
        "reason": reason,
    }

def _empty_basket_backtest_stats():
    """Return the empty-result shape for Top-N basket backtest reports."""
    return {
        "average_basket_raw_return": np.nan,
        "average_basket_benchmark_return": np.nan,
        "average_basket_excess_return": np.nan,
        "positive_basket_return_rate": np.nan,
        "beat_benchmark_rate": np.nan,
        "average_selected_count": np.nan,
        "evaluated_dates": 0,
    }
