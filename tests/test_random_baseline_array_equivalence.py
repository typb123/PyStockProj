"""Strict reference checks for array-oriented random-baseline evaluation."""

import numpy as np
import pandas as pd

import src.train.evaluation.baselines as baselines
import src.train.evaluation.top_n as top_n
from src.train.evaluation.basket_backtest import (
    _basket_backtest_date_stats,
    _summarize_basket_date_stats,
)
from src.train.evaluation.ranked_selection import _ranked_selection_stats


def _shuffled_candidate_groups():
    metadata = pd.DataFrame(
        {
            "Ticker": ["SPY", "BBB", "AAA", "CCC", "AAA", "BBB", "AAA", "BBB"],
            "prediction_date": pd.to_datetime(
                [
                    "2023-01-03",
                    "2022-12-30",
                    "2023-01-03",
                    "2023-06-01",
                    "2022-12-30",
                    "2023-06-01",
                    "2023-06-01",
                    "2023-01-03",
                ]
            ),
            "raw_forward_return": [
                0.0,
                np.nan,
                -0.04,
                0.0,
                0.10,
                -0.02,
                0.03,
                0.08,
            ],
            "benchmark_forward_return": [
                0.0,
                0.01,
                0.02,
                0.01,
                0.01,
                np.nan,
                0.01,
                0.02,
            ],
            "excess_forward_return": [
                0.0,
                np.nan,
                -0.06,
                -0.01,
                0.09,
                np.nan,
                0.02,
                0.06,
            ],
            "beat_benchmark_target": [0.0, np.nan, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "score": [99.0, 0.5, 0.5, 0.2, 0.5, 0.2, 0.2, 0.5],
        }
    )
    candidates = metadata.loc[metadata["Ticker"] != "SPY"].copy()
    candidates["_prediction_year"] = candidates["prediction_date"].dt.year.astype(str)
    return list(candidates.groupby("prediction_date", sort=True))


def _reference_random_reports(
    grouped_metadata,
    selected_count_by_date,
    random_seed,
    random_trials,
):
    """Literal pre-optimization pandas reference used only by these tests."""
    trial_ranked_stats = []
    trial_basket_stats = []
    trial_date_stats = []
    sampled_positions = []
    for trial_seed in baselines._random_trial_seeds(random_seed, random_trials):
        rng = np.random.default_rng(trial_seed)
        selected_groups = []
        trial_positions = []
        for prediction_date, date_group in grouped_metadata:
            selected_count = selected_count_by_date[prediction_date]
            if selected_count == 0:
                continue
            positions = rng.choice(len(date_group), size=selected_count, replace=False)
            trial_positions.append(positions)
            selected_groups.append(date_group.iloc[positions])
        date_stats = _basket_backtest_date_stats(selected_groups)
        trial_ranked_stats.append(
            _ranked_selection_stats(selected_groups, grouped_metadata)
        )
        trial_basket_stats.append(_summarize_basket_date_stats(date_stats))
        trial_date_stats.append(date_stats)
        sampled_positions.append(trial_positions)

    ranked = baselines._average_random_trial_stats(
        trial_ranked_stats,
        random_seed,
        random_trials,
    )
    basket = baselines._average_random_basket_trial_stats(trial_basket_stats)
    basket["random_seed"] = int(random_seed)
    basket["random_trials"] = int(random_trials)
    by_year = {}
    years = sorted(
        {
            year
            for date_stats in trial_date_stats
            for year in date_stats.get("prediction_year", pd.Series(dtype=str)).astype(str)
        }
    )
    for year in years:
        year_trial_stats = []
        for date_stats in trial_date_stats:
            selected = date_stats.loc[
                date_stats["prediction_year"].astype(str) == year
            ]
            year_trial_stats.append(
                _summarize_basket_date_stats(selected)
                if len(selected)
                else baselines._empty_basket_backtest_stats()
            )
        stats = baselines._average_random_basket_trial_stats(year_trial_stats)
        stats["random_seed"] = int(random_seed)
        stats["random_trials"] = int(random_trials)
        by_year[year] = stats
    return ranked, basket, by_year, sampled_positions


def _assert_nested_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if isinstance(actual_value, dict):
            _assert_nested_close(actual_value, expected_value)
        elif isinstance(actual_value, (float, np.floating)):
            assert np.isclose(actual_value, expected_value, rtol=1e-12, atol=1e-14, equal_nan=True)
        else:
            assert actual_value == expected_value


def test_array_random_baseline_matches_pandas_reference_for_seeds_trials_and_workers():
    grouped_metadata = _shuffled_candidate_groups()
    selected_count_by_date = {
        prediction_date: min(2, len(date_group))
        for prediction_date, date_group in grouped_metadata
    }

    for random_seed, random_trials, worker_count in (
        (7, 1, 1),
        (7, 5, 1),
        (11, 6, 2),
    ):
        expected_ranked, expected_basket, expected_by_year, _ = (
            _reference_random_reports(
                grouped_metadata,
                selected_count_by_date,
                random_seed,
                random_trials,
            )
        )
        actual_ranked, actual_basket, actual_by_year = (
            baselines._random_top_n_selection_reports(
                grouped_metadata,
                selected_count_by_date,
                random_seed=random_seed,
                random_trials=random_trials,
                random_trial_workers=worker_count,
            )
        )

        _assert_nested_close(actual_ranked, expected_ranked)
        _assert_nested_close(actual_basket, expected_basket)
        _assert_nested_close(actual_by_year, expected_by_year)


def test_full_selection_still_consumes_choice_before_later_partial_selection(monkeypatch):
    grouped_metadata = _shuffled_candidate_groups()
    # The first sorted date has two candidates and is fully selected; the later
    # three-candidate date remains partial. Skipping either full choice changes
    # that later sample.
    selected_count_by_date = {
        grouped_metadata[0][0]: 2,
        grouped_metadata[1][0]: 2,
        grouped_metadata[2][0]: 2,
    }
    expected = np.random.default_rng(42)
    expected_positions = [
        expected.choice(2, size=2, replace=False),
        expected.choice(2, size=2, replace=False),
        expected.choice(3, size=2, replace=False),
    ]
    recorded_positions = []
    original_default_rng = np.random.default_rng

    class RecordingGenerator:
        def __init__(self, seed):
            self._generator = original_default_rng(seed)

        def choice(self, *args, **kwargs):
            positions = self._generator.choice(*args, **kwargs)
            recorded_positions.append(positions.copy())
            return positions

    monkeypatch.setattr(
        baselines.np.random,
        "default_rng",
        lambda seed: RecordingGenerator(seed),
    )
    baselines._random_top_n_selection_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed=42,
        random_trials=1,
        random_trial_workers=1,
    )

    assert len(recorded_positions) == 3
    for actual, expected_positions_for_date in zip(
        recorded_positions,
        expected_positions,
    ):
        np.testing.assert_array_equal(actual, expected_positions_for_date)


def test_array_random_baseline_preserves_mixed_zero_selection_private_behavior():
    grouped_metadata = _shuffled_candidate_groups()
    selected_count_by_date = {
        grouped_metadata[0][0]: 0,
        grouped_metadata[1][0]: 2,
        grouped_metadata[2][0]: 1,
    }
    expected_ranked, expected_basket, expected_by_year, _ = _reference_random_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed=7,
        random_trials=3,
    )
    actual_ranked, actual_basket, actual_by_year = (
        baselines._random_top_n_selection_reports(
            grouped_metadata,
            selected_count_by_date,
            random_seed=7,
            random_trials=3,
            random_trial_workers=1,
        )
    )

    _assert_nested_close(actual_ranked, expected_ranked)
    _assert_nested_close(actual_basket, expected_basket)
    _assert_nested_close(actual_by_year, expected_by_year)


def test_complete_top_n_report_matches_pandas_random_reference(monkeypatch):
    metadata = pd.concat(
        [date_group for _, date_group in _shuffled_candidate_groups()],
        ignore_index=True,
    )
    spy_row = pd.DataFrame(
        {
            "Ticker": ["SPY"],
            "prediction_date": [pd.Timestamp("2023-01-03")],
            "raw_forward_return": [0.0],
            "benchmark_forward_return": [0.0],
            "excess_forward_return": [0.0],
            "beat_benchmark_target": [0.0],
            "score": [99.0],
        }
    )
    metadata = pd.concat([metadata.drop(columns=["_prediction_year"]), spy_row], ignore_index=True)
    scores = metadata["score"].to_numpy()
    optimized_report = top_n.build_top_n_selection_reports(
        metadata,
        scores,
        top_n_values=(1, 2, 5),
        random_seed=11,
        random_trials=5,
        random_trial_workers=1,
        bootstrap_trials=20,
        bootstrap_seed=17,
    )

    def pandas_reference_random_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed,
        random_trials,
        random_trial_workers=4,
        prepared_input=None,
    ):
        del random_trial_workers, prepared_input
        ranked, basket, by_year, _ = _reference_random_reports(
            grouped_metadata,
            selected_count_by_date,
            random_seed,
            random_trials,
        )
        return ranked, basket, by_year

    monkeypatch.setattr(
        top_n,
        "_random_top_n_selection_reports",
        pandas_reference_random_reports,
    )
    reference_report = top_n.build_top_n_selection_reports(
        metadata,
        scores,
        top_n_values=(1, 2, 5),
        random_seed=11,
        random_trials=5,
        random_trial_workers=1,
        bootstrap_trials=20,
        bootstrap_seed=17,
    )

    _assert_nested_close(optimized_report, reference_report)


def test_array_random_baseline_preserves_date_weighting_median_and_sign_rules():
    grouped_metadata = _shuffled_candidate_groups()
    selected_count_by_date = {
        prediction_date: len(date_group)
        for prediction_date, date_group in grouped_metadata
    }
    ranked, basket, _ = baselines._random_top_n_selection_reports(
        grouped_metadata,
        selected_count_by_date,
        random_seed=42,
        random_trials=1,
        random_trial_workers=1,
    )

    date_means = np.array([0.10, 0.02, (0.00 - 0.02 + 0.03) / 3.0])
    assert np.isclose(
        ranked["average_selected_raw_forward_return"],
        np.mean(date_means),
    )
    # This deliberately differs from a pooled-row mean because date candidate
    # counts vary and the report is an equal-weight mean of date baskets.
    assert not np.isclose(ranked["average_selected_raw_forward_return"], 0.025)
    assert np.isnan(ranked["median_selected_excess_return_vs_benchmark"])
    assert ranked["positive_raw_return_rate"] == 3 / 7
    assert basket["positive_basket_return_rate"] == 1.0
    assert basket["beat_benchmark_rate"] == 2 / 3
