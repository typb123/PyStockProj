"""Basket backtest helpers for equal-weight Top-N reports."""

import numpy as np
import pandas as pd

from src.train.evaluation.validation import _mean_or_nan, _up_rate_or_nan


def _basket_backtest_stats(selected_groups):
    """Summarize equal-weight basket returns across selected prediction dates."""
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
