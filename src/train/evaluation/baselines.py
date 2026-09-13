"""Random, momentum, universe, and benchmark baseline helpers."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.train.evaluation.basket_backtest import (
    _basket_backtest_stats,
    _empty_basket_backtest_stats,
    _summarize_basket_date_stats,
)
from src.train.evaluation.ranked_selection import (
    _empty_ranked_selection_stats,
    _ranked_selection_stats,
    _select_top_n_by_score,
)
from src.train.evaluation.validation import _mean_or_nan, _up_rate_or_nan


@dataclass(frozen=True)
class _RandomDateArrays:
    """One ordered candidate date group prepared for random trial sampling."""

    prediction_year: str
    raw_forward_return: np.ndarray
    benchmark_forward_return: np.ndarray
    excess_forward_return: np.ndarray
    beat_benchmark_target: np.ndarray
    universe_raw_forward_return: float


@dataclass(frozen=True)
class _RandomTrialOutcome:
    """Compact per-trial summaries reused for overall and by-year reports."""

    ranked_stats: dict
    basket_stats: dict
    prediction_years: tuple[str, ...]
    basket_raw_returns: np.ndarray
    basket_benchmark_returns: np.ndarray
    basket_excess_returns: np.ndarray
    selected_counts: np.ndarray


def _random_top_n_selection_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
    random_trial_workers=4,
):
    """Build random ranked-selection and basket baselines for Top-N buckets."""
    ranked_stats, basket_stats, _ = _random_top_n_selection_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed=random_seed,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    return ranked_stats, basket_stats


def _random_top_n_selection_reports(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
    random_trial_workers=4,
    prepared_input: tuple[_RandomDateArrays, ...] | None = None,
):
    """Build overall and by-year random baselines from one random trial run."""
    if random_trial_workers < 1:
        raise ValueError("random_trial_workers must be at least 1.")

    trial_results = _random_top_n_selection_trial_results(
        grouped_metadata,
        selected_count_by_date,
        random_seed,
        random_trials,
        random_trial_workers,
        prepared_input=prepared_input,
    )
    ranked_trial_stats = [trial.ranked_stats for trial in trial_results]
    basket_trial_stats = [trial.basket_stats for trial in trial_results]

    ranked_stats = _average_random_trial_stats(
        ranked_trial_stats,
        random_seed,
        random_trials,
    )
    basket_stats = _average_random_basket_trial_stats(basket_trial_stats)
    basket_stats["random_seed"] = int(random_seed)
    basket_stats["random_trials"] = int(random_trials)
    basket_by_year_stats = _average_random_basket_trial_stats_by_year(
        trial_results,
        random_seed,
        random_trials,
    )
    return ranked_stats, basket_stats, basket_by_year_stats


def _random_top_n_selection_trial_results(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
    random_trial_workers,
    prepared_input: tuple[_RandomDateArrays, ...] | None = None,
):
    """Generate per-trial random selections through the centralized worker path."""
    trial_seeds = _random_trial_seeds(random_seed, random_trials)
    prepared_input = (
        _prepare_random_trial_input(grouped_metadata)
        if prepared_input is None
        else prepared_input
    )
    selected_counts = tuple(
        selected_count_by_date[prediction_date]
        for prediction_date, _ in grouped_metadata
    )
    if random_trial_workers > 1 and random_trials > 1:
        worker_count = min(int(random_trial_workers), int(random_trials))
        seed_chunks = np.array_split(
            np.asarray(trial_seeds, dtype=np.uint64),
            worker_count,
        )
        tasks = [
            (
                prepared_input,
                selected_counts,
                [int(seed) for seed in seed_chunk],
            )
            for seed_chunk in seed_chunks
            if len(seed_chunk) > 0
        ]
        trial_results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result_chunk in executor.map(
                _random_top_n_selection_results_worker,
                tasks,
            ):
                trial_results.extend(result_chunk)
        return trial_results

    return [
        _random_top_n_selection_trial_result(
            prepared_input, selected_counts, trial_seed
        )
        for trial_seed in trial_seeds
    ]


def _random_trial_seeds(random_seed, random_trials):
    """Derive independent per-trial seeds for random baseline evaluation."""
    if int(random_trials) == 1:
        return [int(random_seed)]

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


def _random_top_n_selection_results_worker(args):
    """Run a chunk of random Top-N trials inside a process-pool worker."""
    prepared_input, selected_counts, trial_seeds = args

    return [
        _random_top_n_selection_trial_result(
            prepared_input, selected_counts, trial_seed
        )
        for trial_seed in trial_seeds
    ]


def _random_top_n_selection_trial_result(
    prepared_input,
    selected_counts,
    trial_seed,
):
    """Run one random trial with the legacy RNG sequence and array reductions."""
    rng = np.random.default_rng(trial_seed)
    selected_raw_means = []
    selected_benchmark_means = []
    selected_excess_means = []
    selected_raw_values = []
    selected_excess_values = []
    selected_beat_targets = []
    prediction_years = []
    selected_counts_by_date = []

    for date_arrays, selected_count in zip(prepared_input, selected_counts):
        if selected_count == 0:
            continue
        # Even a full selection must advance the per-trial generator exactly as
        # before so later date samples remain identical.
        selected_positions = rng.choice(
            len(date_arrays.raw_forward_return),
            size=selected_count,
            replace=False,
        )
        selected_raw = date_arrays.raw_forward_return[selected_positions]
        selected_benchmark = date_arrays.benchmark_forward_return[selected_positions]
        selected_excess = date_arrays.excess_forward_return[selected_positions]
        selected_beat = date_arrays.beat_benchmark_target[selected_positions]

        selected_raw_mean = _pandas_skipna_mean(selected_raw)
        selected_raw_means.append(selected_raw_mean)
        selected_benchmark_means.append(_pandas_skipna_mean(selected_benchmark))
        selected_excess_means.append(_pandas_skipna_mean(selected_excess))
        selected_raw_values.append(selected_raw)
        selected_excess_values.append(selected_excess)
        selected_beat_targets.append(selected_beat)
        prediction_years.append(date_arrays.prediction_year)
        selected_counts_by_date.append(selected_count)

    if not selected_raw_means:
        return _RandomTrialOutcome(
            ranked_stats=_empty_ranked_selection_stats(),
            basket_stats=_empty_basket_backtest_stats(),
            prediction_years=(),
            basket_raw_returns=np.asarray([]),
            basket_benchmark_returns=np.asarray([]),
            basket_excess_returns=np.asarray([]),
            selected_counts=np.asarray([]),
        )

    selected_raw_means = np.asarray(selected_raw_means)
    selected_benchmark_means = np.asarray(selected_benchmark_means)
    selected_excess_means = np.asarray(selected_excess_means)
    # Preserve the legacy private-helper alignment behavior for mixed zero and
    # nonzero selections: its selected-group list was zipped to the first N
    # candidate date groups. Normal Top-N production paths select every date.
    ranked_reference_dates = prepared_input[: len(selected_raw_means)]
    universe_raw_means = np.asarray(
        [
            date_arrays.universe_raw_forward_return
            for date_arrays in ranked_reference_dates
        ]
    )
    candidate_counts = np.asarray(
        [len(date_arrays.raw_forward_return) for date_arrays in ranked_reference_dates],
        dtype=float,
    )
    selected_raw_values = np.concatenate(selected_raw_values)
    selected_excess_values = np.concatenate(selected_excess_values)
    selected_beat_targets = np.concatenate(selected_beat_targets)
    selected_counts_by_date = np.asarray(selected_counts_by_date)

    ranked_stats = {
        "date_count": int(len(selected_raw_means)),
        "selected_row_count": int(len(selected_raw_values)),
        "average_selected_raw_forward_return": _mean_or_nan(selected_raw_means),
        "average_selected_benchmark_forward_return": _mean_or_nan(
            selected_benchmark_means
        ),
        "average_selected_excess_return_vs_benchmark": _mean_or_nan(
            selected_excess_means
        ),
        "beat_benchmark_rate": _mean_or_nan(selected_beat_targets),
        "average_equal_weight_universe_forward_return": _mean_or_nan(
            universe_raw_means
        ),
        "average_selected_return_minus_universe_return": _mean_or_nan(
            selected_raw_means - universe_raw_means
        ),
        "median_selected_excess_return_vs_benchmark": float(
            np.median(selected_excess_values)
        ),
        "positive_raw_return_rate": _up_rate_or_nan(selected_raw_values),
        "average_number_of_candidates_per_date": _mean_or_nan(candidate_counts),
    }
    basket_stats = _array_basket_stats(
        selected_raw_means,
        selected_benchmark_means,
        selected_excess_means,
        selected_counts_by_date,
    )
    return _RandomTrialOutcome(
        ranked_stats=ranked_stats,
        basket_stats=basket_stats,
        prediction_years=tuple(prediction_years),
        basket_raw_returns=selected_raw_means,
        basket_benchmark_returns=selected_benchmark_means,
        basket_excess_returns=selected_excess_means,
        selected_counts=selected_counts_by_date,
    )


def _prepare_random_trial_input(grouped_metadata) -> tuple[_RandomDateArrays, ...]:
    """Extract ordered date-local arrays once for all random trials and Top-Ns."""
    prepared_dates = []
    for _, date_group in grouped_metadata:
        raw_forward_return = date_group["raw_forward_return"].to_numpy(copy=False)
        prepared_dates.append(
            _RandomDateArrays(
                prediction_year=str(
                    pd.to_datetime(
                        date_group["prediction_date"].iloc[0], errors="raise"
                    ).year
                ),
                raw_forward_return=raw_forward_return,
                benchmark_forward_return=date_group[
                    "benchmark_forward_return"
                ].to_numpy(copy=False),
                excess_forward_return=date_group["excess_forward_return"].to_numpy(
                    copy=False
                ),
                beat_benchmark_target=date_group["beat_benchmark_target"].to_numpy(
                    copy=False
                ),
                universe_raw_forward_return=_pandas_skipna_mean(raw_forward_return),
            )
        )
    return tuple(prepared_dates)


def _pandas_skipna_mean(values):
    """Match pandas Series.mean()'s NaN-skipping behavior for numeric arrays."""
    valid_values = np.asarray(values)[pd.notna(values)]
    if len(valid_values) == 0:
        return np.nan
    return float(np.mean(valid_values))


def _array_basket_stats(
    raw_returns,
    benchmark_returns,
    excess_returns,
    selected_counts,
):
    """Match basket date-stat summarization without building a DataFrame."""
    return {
        "average_basket_raw_return": _mean_or_nan(raw_returns),
        "average_basket_benchmark_return": _mean_or_nan(benchmark_returns),
        "average_basket_excess_return": _mean_or_nan(excess_returns),
        "positive_basket_return_rate": _up_rate_or_nan(raw_returns),
        "beat_benchmark_rate": _up_rate_or_nan(excess_returns),
        "average_selected_count": _mean_or_nan(selected_counts),
        "evaluated_dates": int(len(raw_returns)),
    }


def _combined_momentum_ranked_selection_stats(
    selected_groups,
    grouped_metadata,
    momentum_score_column,
):
    """Attach availability metadata to absolute-momentum ranked stats."""
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
    """Attach availability metadata to SPY-relative momentum ranked stats."""
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
    """Attach availability metadata to absolute-momentum basket stats."""
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
    """Attach availability metadata to SPY-relative momentum basket stats."""
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
    """Choose the absolute-momentum baseline column for this horizon."""
    if momentum_score_column is not None:
        return momentum_score_column

    if prediction_days is not None:
        return f"momentum_{int(prediction_days)}d"

    return "dailyReturn"


def _resolve_relative_momentum_score_column(metadata, prediction_days):
    """Choose the SPY-relative momentum baseline column for this horizon."""
    if prediction_days is None:
        return "relative_momentum"

    return f"relative_momentum_{int(prediction_days)}d"


def _random_basket_backtest_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
):
    """Build a sequential random basket backtest baseline."""
    _, basket_stats, _ = _random_top_n_selection_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed=random_seed,
        random_trials=random_trials,
        random_trial_workers=1,
    )
    return basket_stats


def _average_random_basket_trial_stats(trial_stats):
    """Average basket backtest metrics across random trials."""
    if not trial_stats:
        return _empty_basket_backtest_stats()

    stats = {}
    for metric in trial_stats[0]:
        values = np.asarray([trial[metric] for trial in trial_stats], dtype=float)
        stats[metric] = _mean_or_nan(values)

    stats["evaluated_dates"] = int(round(stats["evaluated_dates"]))
    return stats


def _average_random_basket_trial_stats_by_year(
    trial_outcomes,
    random_seed,
    random_trials,
):
    """Average random basket metrics across trials, grouped by prediction year."""
    years = sorted(
        {year for outcome in trial_outcomes for year in outcome.prediction_years}
    )
    by_year_stats = {}

    for year in years:
        trial_stats = []
        for outcome in trial_outcomes:
            year_mask = np.asarray(
                [outcome_year == year for outcome_year in outcome.prediction_years]
            )
            if not np.any(year_mask):
                trial_stats.append(_empty_basket_backtest_stats())
            else:
                trial_stats.append(
                    _array_basket_stats(
                        outcome.basket_raw_returns[year_mask],
                        outcome.basket_benchmark_returns[year_mask],
                        outcome.basket_excess_returns[year_mask],
                        outcome.selected_counts[year_mask],
                    )
                )

        stats = _average_random_basket_trial_stats(trial_stats)
        stats["random_seed"] = int(random_seed)
        stats["random_trials"] = int(random_trials)
        by_year_stats[year] = stats

    return by_year_stats


def _momentum_basket_backtest_stats(
    grouped_metadata,
    selected_count_by_date,
    momentum_score_column,
):
    """Build the absolute-momentum basket baseline when its score exists."""
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
    """Build the SPY-relative momentum basket baseline when its score exists."""
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
    """Represent the SPY benchmark as the per-date basket comparator."""
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


def _random_ranked_selection_stats(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
):
    """Build a sequential random ranked-selection baseline."""
    ranked_stats, _, _ = _random_top_n_selection_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed=random_seed,
        random_trials=random_trials,
        random_trial_workers=1,
    )
    return ranked_stats


def _average_random_trial_stats(trial_stats, random_seed, random_trials):
    """Average ranked-selection metrics across random trials."""
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
    """Build the absolute-momentum ranked-selection baseline."""
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
    """Build the SPY-relative momentum ranked-selection baseline."""
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
    """Summarize the full non-SPY candidate universe by prediction date."""
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
