"""Public Top-N ranked-selection and basket-backtest orchestration."""

from copy import deepcopy
import hashlib
import time

import numpy as np
import pandas as pd

from src.train.evaluation.baselines import (
    _benchmark_basket_backtest_stats,
    _combined_momentum_basket_backtest_stats,
    _combined_momentum_ranked_selection_stats,
    _combined_relative_momentum_basket_backtest_stats,
    _combined_relative_momentum_ranked_selection_stats,
    _random_top_n_selection_reports,
    _resolve_momentum_score_column,
    _resolve_relative_momentum_score_column,
    _universe_ranked_selection_stats,
)
from src.train.evaluation.basket_backtest import (
    _basket_backtest_date_stats,
    _basket_backtest_stats,
    _bootstrap_basket_confidence_intervals,
    _summarize_basket_date_stats,
)
from src.train.evaluation.ranked_selection import (
    _ranked_selection_stats,
    _select_top_n_by_score,
)
from src.train.evaluation.validation import _validate_ranked_selection_inputs


_PREDICTION_YEAR_COLUMN = "_prediction_year"


def build_top_n_ranked_selection_report(
    split_metadata,
    predicted_excess_returns,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
    bootstrap_trials=500,
    bootstrap_seed=42,
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
        bootstrap_trials=bootstrap_trials,
        bootstrap_seed=bootstrap_seed,
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
    bootstrap_trials=500,
    bootstrap_seed=42,
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
        bootstrap_trials=bootstrap_trials,
        bootstrap_seed=bootstrap_seed,
    )["basket_backtest"]


def build_model_only_top_n_basket_backtest_report(
    split_metadata,
    ranked_predictions,
    top_n_values=(5, 10, 20),
):
    """Summarize model-selected Top-N baskets without baseline evaluation.

    This intentionally skips random and momentum baselines so validation model
    selection can optimize the same basket metric used in final reports without
    running expensive random trials for every candidate.
    """
    metadata = split_metadata.copy()
    ranked_predictions = np.asarray(ranked_predictions, dtype=float)
    _validate_ranked_selection_inputs(metadata, ranked_predictions)

    score_column = "predicted_excess_return"
    metadata[score_column] = ranked_predictions
    metadata = metadata[metadata["Ticker"] != "SPY"].copy()
    grouped_metadata = list(metadata.groupby("prediction_date", sort=True))
    basket_backtest_report = {}

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
        basket_backtest_report[key] = {
            "model": _basket_backtest_stats(model_selected_groups),
        }

    return basket_backtest_report


def build_top_n_selection_reports(
    split_metadata,
    ranked_predictions,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
    bootstrap_trials=500,
    bootstrap_seed=42,
    prepared_evaluation_state=None,
    include_phase_timings=False,
):
    """Build regressor-ranked Top-N reports using predicted excess return."""
    return _build_top_n_selection_reports_for_score(
        split_metadata,
        ranked_predictions,
        "predicted_excess_return",
        top_n_values=top_n_values,
        random_seed=random_seed,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
        momentum_score_column=momentum_score_column,
        prediction_days=prediction_days,
        bootstrap_trials=bootstrap_trials,
        bootstrap_seed=bootstrap_seed,
        prepared_evaluation_state=prepared_evaluation_state,
        include_phase_timings=include_phase_timings,
    )


def prepare_random_top_n_evaluation_state(
    split_metadata,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    prediction_days=None,
):
    """Precompute model-independent random Top-N summaries for one test fold."""
    if random_trials < 1:
        raise ValueError("random_trials must be at least 1.")
    metadata = _evaluation_metadata(split_metadata)
    grouped_metadata = list(metadata.groupby("prediction_date", sort=True))
    started_at = time.perf_counter()
    random_baselines = {}
    for top_n in top_n_values:
        selected_count_by_date = {
            prediction_date: min(int(top_n), len(date_group))
            for prediction_date, date_group in grouped_metadata
        }
        random_baselines[f"top_{top_n}"] = _random_top_n_selection_reports(
            grouped_metadata,
            selected_count_by_date,
            random_seed=random_seed,
            random_trials=random_trials,
            # Context construction runs in the parent before outer workers.
            # Keep the deterministic legacy serial order and avoid a new pool.
            random_trial_workers=1,
        )
    return {
        "identity": _evaluation_input_identity(metadata),
        "top_n_values": tuple(int(value) for value in top_n_values),
        "random_seed": int(random_seed),
        "random_trials": int(random_trials),
        "prediction_days": None if prediction_days is None else int(prediction_days),
        "random_baselines": random_baselines,
        "preparation_seconds": time.perf_counter() - started_at,
    }


def build_probability_ranked_top_n_selection_reports(
    split_metadata,
    classifier_probabilities,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
    bootstrap_trials=500,
    bootstrap_seed=42,
):
    """Build classifier-probability-ranked Top-N reports."""
    return _build_top_n_selection_reports_for_score(
        split_metadata,
        classifier_probabilities,
        "predicted_beat_benchmark_probability",
        top_n_values=top_n_values,
        random_seed=random_seed,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
        momentum_score_column=momentum_score_column,
        prediction_days=prediction_days,
        bootstrap_trials=bootstrap_trials,
        bootstrap_seed=bootstrap_seed,
    )


def _build_top_n_selection_reports_for_score(
    split_metadata,
    ranking_scores,
    score_column,
    top_n_values=(5, 10, 20),
    random_seed=42,
    random_trials=100,
    random_trial_workers=4,
    momentum_score_column=None,
    prediction_days=None,
    bootstrap_trials=500,
    bootstrap_seed=42,
    prepared_evaluation_state=None,
    include_phase_timings=False,
):
    """Build ranked-selection and basket-backtest Top-N reports for a score column."""
    if random_trials < 1:
        raise ValueError("random_trials must be at least 1.")
    if random_trial_workers < 1:
        raise ValueError("random_trial_workers must be at least 1.")
    if bootstrap_trials < 1:
        raise ValueError("bootstrap_trials must be at least 1.")

    setup_started_at = time.perf_counter()
    raw_metadata = split_metadata.copy()
    ranking_scores = np.asarray(ranking_scores, dtype=float)
    _validate_ranked_selection_inputs(raw_metadata, ranking_scores)
    raw_metadata[score_column] = ranking_scores
    metadata = _evaluation_metadata(raw_metadata)
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
    basket_backtest_by_year_report = {}
    grouped_metadata_by_year = _grouped_metadata_by_year(grouped_metadata)
    universe_started_at = time.perf_counter()
    universe_ranked_stats = _universe_ranked_selection_stats(grouped_metadata)
    universe_date_stats = _basket_backtest_date_stats(
        [date_group for _, date_group in grouped_metadata]
    )
    universe_basket_stats = _summarize_basket_date_stats(universe_date_stats)
    benchmark_basket_stats = _benchmark_basket_backtest_stats(grouped_metadata)
    universe_baseline_seconds = time.perf_counter() - universe_started_at
    phase_timings = {
        "evaluation_setup_seconds": time.perf_counter() - setup_started_at,
        "model_top_n_seconds": 0.0,
        "momentum_baseline_seconds": 0.0,
        "universe_baseline_seconds": universe_baseline_seconds,
        "random_baseline_seconds": 0.0,
        "bootstrap_seconds": 0.0,
        "report_assembly_seconds": 0.0,
    }
    _validate_prepared_evaluation_state(
        prepared_evaluation_state,
        metadata.drop(columns=[score_column]),
        top_n_values,
        random_seed,
        random_trials,
        prediction_days,
    )

    for top_n in top_n_values:
        key = f"top_{top_n}"
        selected_count_by_date = {
            prediction_date: min(int(top_n), len(date_group))
            for prediction_date, date_group in grouped_metadata
        }
        model_started_at = time.perf_counter()
        model_selected_groups = _select_top_n_by_score(
            grouped_metadata,
            selected_count_by_date,
            score_column,
        )
        model_date_stats = _basket_backtest_date_stats(model_selected_groups)
        model_ranked_stats = _ranked_selection_stats(
            model_selected_groups,
            grouped_metadata,
            score_column=score_column,
        )
        phase_timings["model_top_n_seconds"] += time.perf_counter() - model_started_at
        random_started_at = time.perf_counter()
        if prepared_evaluation_state is None:
            random_reports = _random_top_n_selection_reports(
                grouped_metadata,
                selected_count_by_date,
                random_seed=random_seed,
                random_trials=random_trials,
                random_trial_workers=random_trial_workers,
            )
        else:
            random_reports = prepared_evaluation_state["random_baselines"][key]
        (
            random_ranked_stats,
            random_basket_stats,
            random_basket_by_year_stats,
        ) = deepcopy(random_reports)
        if prepared_evaluation_state is None:
            phase_timings["random_baseline_seconds"] += (
                time.perf_counter() - random_started_at
            )
        momentum_started_at = time.perf_counter()
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
        momentum_date_stats = _available_basket_date_stats(momentum_selected_groups)
        relative_momentum_date_stats = _available_basket_date_stats(
            relative_momentum_selected_groups
        )
        phase_timings["momentum_baseline_seconds"] += (
            time.perf_counter() - momentum_started_at
        )

        ranked_selection_report[key] = {
            "model": model_ranked_stats,
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
        bootstrap_started_at = time.perf_counter()
        bootstrap_confidence_intervals = _bootstrap_basket_confidence_intervals(
            model_date_stats,
            momentum_date_stats=momentum_date_stats,
            relative_momentum_date_stats=relative_momentum_date_stats,
            universe_date_stats=universe_date_stats,
            bootstrap_trials=bootstrap_trials,
            bootstrap_seed=bootstrap_seed,
        )
        phase_timings["bootstrap_seconds"] += time.perf_counter() - bootstrap_started_at
        assembly_started_at = time.perf_counter()
        basket_backtest_report[key] = {
            "model": _summarize_basket_date_stats(model_date_stats),
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
            "bootstrap_confidence_intervals": bootstrap_confidence_intervals,
        }
        basket_backtest_by_year_report[key] = _build_basket_backtest_by_year_report(
            grouped_metadata_by_year,
            model_selected_groups,
            momentum_selected_groups,
            relative_momentum_selected_groups,
            momentum_score_column,
            relative_momentum_score_column,
            random_basket_by_year_stats,
        )
        phase_timings["report_assembly_seconds"] += (
            time.perf_counter() - assembly_started_at
        )

    report = {
        "ranked_selection": ranked_selection_report,
        "basket_backtest": basket_backtest_report,
        "basket_backtest_by_year": basket_backtest_by_year_report,
    }
    if include_phase_timings:
        report["phase_timings"] = phase_timings
    return report


def _evaluation_metadata(split_metadata):
    """Normalize the exact model-independent candidate input used by Top-N."""
    metadata = split_metadata.copy()
    _validate_ranked_selection_inputs(metadata, np.zeros(len(metadata)))
    metadata = metadata[metadata["Ticker"] != "SPY"].copy()
    metadata[_PREDICTION_YEAR_COLUMN] = _prediction_year_strings(
        metadata["prediction_date"]
    )
    return metadata


def _evaluation_input_identity(metadata):
    """Hash ordered candidate identity and every realized input used by baselines."""
    row_hashes = pd.util.hash_pandas_object(
        metadata, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return {
        "row_count": int(len(metadata)),
        "columns": list(metadata.columns),
        "ordered_row_digest": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }


def _validate_prepared_evaluation_state(
    state,
    metadata,
    top_n_values,
    random_seed,
    random_trials,
    prediction_days,
):
    """Reject cached random summaries unless their full evaluation inputs match."""
    if state is None:
        return
    expected = {
        "identity": _evaluation_input_identity(metadata),
        "top_n_values": tuple(int(value) for value in top_n_values),
        "random_seed": int(random_seed),
        "random_trials": int(random_trials),
        "prediction_days": None if prediction_days is None else int(prediction_days),
    }
    actual = {key: state.get(key) for key in expected}
    if actual != expected:
        raise ValueError("Prepared Top-N evaluation state does not match inputs/settings.")


def _prediction_year_strings(prediction_dates):
    """Convert prediction dates to stable string year labels."""
    parsed_dates = pd.to_datetime(prediction_dates, errors="raise")
    if parsed_dates.isna().any():
        raise ValueError("prediction_date must not contain null values.")

    return parsed_dates.dt.year.astype(int).astype(str)


def _grouped_metadata_by_year(grouped_metadata):
    """Group date-level metadata groups by prediction_date year."""
    grouped_by_year = {}
    for prediction_date, date_group in grouped_metadata:
        year = _prediction_year_from_group(date_group)
        grouped_by_year.setdefault(year, []).append((prediction_date, date_group))
    return grouped_by_year


def _prediction_year_from_group(date_group):
    """Return the string prediction year for one selected or candidate group."""
    return str(date_group[_PREDICTION_YEAR_COLUMN].iloc[0])


def _selected_groups_by_year(selected_groups):
    """Group selected date baskets by prediction_date year."""
    if selected_groups is None:
        return None

    grouped_by_year = {}
    for selected_group in selected_groups:
        year = _prediction_year_from_group(selected_group)
        grouped_by_year.setdefault(year, []).append(selected_group)
    return grouped_by_year


def _available_basket_date_stats(selected_groups):
    """Return per-date basket stats only when a baseline is available."""
    if selected_groups is None:
        return None

    return _basket_backtest_date_stats(selected_groups)


def _build_basket_backtest_by_year_report(
    grouped_metadata_by_year,
    model_selected_groups,
    momentum_selected_groups,
    relative_momentum_selected_groups,
    momentum_score_column,
    relative_momentum_score_column,
    random_basket_by_year_stats,
):
    """Build equal-weight Top-N basket diagnostics split by prediction year."""
    model_selected_groups_by_year = _selected_groups_by_year(model_selected_groups)
    momentum_selected_groups_by_year = _selected_groups_by_year(momentum_selected_groups)
    relative_momentum_selected_groups_by_year = _selected_groups_by_year(
        relative_momentum_selected_groups
    )
    by_year_report = {}

    for year in sorted(grouped_metadata_by_year):
        year_grouped_metadata = grouped_metadata_by_year[year]
        by_year_report[year] = {
            "model": _basket_backtest_stats(
                model_selected_groups_by_year.get(year, [])
            ),
            "random_baseline": random_basket_by_year_stats.get(year, {}),
            "momentum_baseline": _combined_momentum_basket_backtest_stats(
                (
                    None
                    if momentum_selected_groups_by_year is None
                    else momentum_selected_groups_by_year.get(year, [])
                ),
                momentum_score_column,
            ),
            "relative_momentum_baseline": _combined_relative_momentum_basket_backtest_stats(
                (
                    None
                    if relative_momentum_selected_groups_by_year is None
                    else relative_momentum_selected_groups_by_year.get(year, [])
                ),
                relative_momentum_score_column,
            ),
            "universe": _basket_backtest_stats(
                [date_group for _, date_group in year_grouped_metadata]
            ),
            "benchmark": _benchmark_basket_backtest_stats(year_grouped_metadata),
        }

    return by_year_report

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
