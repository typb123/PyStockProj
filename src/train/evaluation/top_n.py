"""Public Top-N ranked-selection and basket-backtest orchestration."""

import numpy as np

from src.train.evaluation.baselines import (
    _benchmark_basket_backtest_stats,
    _combined_momentum_basket_backtest_stats,
    _combined_momentum_ranked_selection_stats,
    _combined_relative_momentum_basket_backtest_stats,
    _combined_relative_momentum_ranked_selection_stats,
    _random_top_n_selection_stats,
    _resolve_momentum_score_column,
    _resolve_relative_momentum_score_column,
    _universe_ranked_selection_stats,
)
from src.train.evaluation.basket_backtest import _basket_backtest_stats
from src.train.evaluation.ranked_selection import (
    _ranked_selection_stats,
    _select_top_n_by_score,
)
from src.train.evaluation.validation import _validate_ranked_selection_inputs


def build_top_n_ranked_selection_report(
    split_metadata,
    predicted_excess_returns,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
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

def _select_available_top_n_groups(
    grouped_metadata,
    selected_count_by_date,
    score_column,
    metadata_columns,
):
    """Select Top-N groups when a baseline score column is available."""
    if score_column not in metadata_columns:
        return None

    return _select_top_n_by_score(
        grouped_metadata,
        selected_count_by_date,
        score_column,
    )
