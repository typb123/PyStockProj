"""Evaluation report builders for SPY-relative model diagnostics."""

from src.train.evaluation.basic_reports import (
    build_actual_return_baseline_report,
    build_classification_report,
    build_combined_signal_report,
    build_predicted_return_quantile_report,
    build_probability_summary,
    build_probability_tail_report,
    build_probability_threshold_report,
    build_regression_report,
    build_return_correlation_report,
    build_trading_relevance_report,
    build_validation_selected_threshold_report,
)
from src.train.evaluation.top_n import (
    build_top_n_basket_backtest_report,
    build_top_n_ranked_selection_report,
    build_top_n_selection_reports,
)

__all__ = [
    "build_actual_return_baseline_report",
    "build_classification_report",
    "build_combined_signal_report",
    "build_predicted_return_quantile_report",
    "build_probability_summary",
    "build_probability_tail_report",
    "build_probability_threshold_report",
    "build_regression_report",
    "build_return_correlation_report",
    "build_trading_relevance_report",
    "build_validation_selected_threshold_report",
    "build_top_n_basket_backtest_report",
    "build_top_n_ranked_selection_report",
    "build_top_n_selection_reports",
]
