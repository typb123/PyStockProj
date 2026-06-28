"""Ranked-selection helpers for per-date Top-N reports."""

import numpy as np
import pandas as pd

from src.train.evaluation.validation import _mean_or_nan, _up_rate_or_nan


def _select_top_n_by_score(grouped_metadata, selected_count_by_date, score_column):
    """Select each date's highest-scoring rows for a Top-N candidate bucket."""
    selected_groups = []

    for prediction_date, date_group in grouped_metadata:
        selected_count = selected_count_by_date[prediction_date]
        if selected_count == 0:
            continue
        selected_groups.append(date_group.nlargest(selected_count, score_column))

    return selected_groups

def _ranked_selection_stats(selected_groups, grouped_metadata, score_column=None):
    """Aggregate per-row and per-date outcomes for selected ranked candidates."""
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

def _empty_ranked_selection_stats(score_column=None):
    """Return the empty-result shape for Top-N ranked selection reports."""
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
