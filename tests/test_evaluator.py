"""Evaluation report tests for SPY-relative ranked watchlist diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.train.evaluator import (
    build_actual_return_baseline_report,
    build_classification_report,
    build_combined_signal_report,
    build_model_only_top_n_basket_backtest_report,
    build_probability_ranked_top_n_selection_reports,
    build_probability_summary,
    build_probability_tail_report,
    build_probability_threshold_report,
    build_validation_selected_threshold_report,
    build_predicted_return_quantile_report,
    build_regression_report,
    build_return_correlation_report,
    build_same_date_ranking_diagnostics,
    build_top_n_basket_backtest_report,
    build_top_n_ranked_selection_report,
    build_top_n_selection_reports,
    build_trading_relevance_report,
)
import src.train.evaluation as evaluation
import src.train.evaluation.baselines as baselines
from src.train.evaluation.basket_backtest import _bootstrap_basket_confidence_intervals
from src.train.evaluation.walk_forward import (
    build_expanding_yearly_walk_forward_folds,
    build_walk_forward_aggregate_summary,
    build_walk_forward_fold_report,
    build_walk_forward_split,
)
import src.train.evaluator as evaluator


def test_build_classification_report_includes_beat_benchmark_baselines_and_counts():
    report = build_classification_report(
        y_true=np.array([1, 0, 1, 0]),
        y_pred=np.array([1, 1, 0, 0]),
    )

    assert report["accuracy"] == 0.5
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["f1"] == 0.5
    assert report["always_up_accuracy"] == 0.5
    assert report["always_down_accuracy"] == 0.5
    assert report["predicted_up_count"] == 2
    assert report["predicted_down_count"] == 2


def test_build_regression_report_includes_excess_return_baselines():
    report = build_regression_report(
        y_true=np.array([0.10, -0.10]),
        y_pred=np.array([0.05, -0.05]),
        y_train=np.array([0.02, 0.04]),
    )

    assert report["mse"] == 0.0025000000000000005
    assert report["mae"] == 0.05
    assert report["zero_return_baseline_mse"] == 0.010000000000000002
    assert report["zero_return_baseline_mae"] == 0.1
    assert report["mean_train_return"] == 0.03
    assert report["mean_train_return_baseline_mse"] == 0.010900000000000002
    assert report["mean_train_return_baseline_mae"] == 0.1


def test_build_trading_relevance_report_splits_by_beat_benchmark_classification():
    report = build_trading_relevance_report(
        actual_returns=np.array([0.10, -0.05, 0.03, -0.02]),
        predicted_returns=np.array([0.08, 0.01, 0.04, -0.03]),
        predicted_direction=np.array([1, 1, 0, 0]),
    )

    assert report["avg_actual_return_when_predicted_up"] == 0.025
    assert report["avg_actual_return_when_predicted_down"] == 0.004999999999999999
    assert report["avg_predicted_return_when_predicted_up"] == 0.045
    assert report["avg_predicted_return_when_predicted_down"] == 0.005000000000000001


def test_build_actual_return_baseline_report_summarizes_full_test_excess_returns():
    report = build_actual_return_baseline_report(np.array([0.10, -0.05, 0.00, 0.03]))

    assert report["count"] == 4
    assert report["avg_actual_return"] == 0.02
    assert report["positive_return_rate"] == 0.5
    assert report["precision"] == 0.5


def test_build_actual_return_baseline_report_rejects_empty_input():
    with pytest.raises(ValueError, match="actual_returns"):
        build_actual_return_baseline_report(np.array([]))


def test_build_actual_return_baseline_report_preserves_nan_inf_behavior():
    report = build_actual_return_baseline_report(np.array([np.nan, np.inf, -0.01]))

    assert report["count"] == 3
    assert np.isnan(report["avg_actual_return"])
    assert report["positive_return_rate"] == 1 / 3
    assert report["precision"] == 1 / 3


def test_build_probability_summary_reports_beat_benchmark_distribution():
    probabilities = np.array([0.1, 0.2, 0.5, 0.8, 0.9])

    report = build_probability_summary(probabilities)

    assert report["count"] == 5
    assert report["min"] == 0.1
    assert report["max"] == 0.9
    assert report["mean"] == 0.5
    assert report["median"] == 0.5
    assert np.isclose(report["std"], np.std(probabilities))
    assert np.isclose(report["p10"], 0.14)
    assert report["p25"] == 0.2
    assert report["p75"] == 0.8
    assert np.isclose(report["p90"], 0.86)


def test_build_probability_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="probability_up"):
        build_probability_summary(np.array([]))


def test_build_probability_tail_report_summarizes_beat_benchmark_tails():
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    actual_returns = np.array(
        [-0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    )

    report = build_probability_tail_report(probabilities, actual_returns)

    assert report["top_10_pct"]["count"] == 1
    assert report["top_10_pct"]["avg_actual_return"] == 0.07
    assert report["top_10_pct"]["precision"] == 1.0
    assert report["top_20_pct"]["count"] == 2
    assert report["top_20_pct"]["avg_actual_return"] == 0.065
    assert report["top_20_pct"]["precision"] == 1.0
    assert report["bottom_10_pct"]["count"] == 1
    assert report["bottom_10_pct"]["avg_actual_return"] == -0.10
    assert report["bottom_10_pct"]["precision"] == 0.0
    assert report["bottom_20_pct"]["count"] == 2
    assert report["bottom_20_pct"]["avg_actual_return"] == -0.07500000000000001
    assert report["bottom_20_pct"]["precision"] == 0.0


def test_probability_tail_and_threshold_reports_reject_mismatched_lengths():
    probabilities = np.array([0.1, 0.2, 0.3])
    actual_returns = np.array([0.01, -0.02])

    for report_builder in [
        build_probability_tail_report,
        build_probability_threshold_report,
    ]:
        with pytest.raises(ValueError, match="same length"):
            report_builder(probabilities, actual_returns)


def test_build_probability_threshold_report_summarizes_beat_benchmark_selected_rows():
    probabilities = np.array([0.40, 0.50, 0.56, 0.61, 0.72])
    actual_returns = np.array([-0.03, 0.01, -0.02, 0.04, 0.05])

    report = build_probability_threshold_report(probabilities, actual_returns)

    assert report[0.50]["selected_count"] == 4
    assert report[0.50]["selected_fraction"] == 0.8
    assert report[0.50]["precision"] == 0.75
    assert report[0.50]["avg_actual_return"] == 0.02
    assert report[0.60]["selected_count"] == 2
    assert report[0.60]["selected_fraction"] == 0.4
    assert report[0.60]["precision"] == 1.0
    assert report[0.60]["avg_actual_return"] == 0.045
    assert report[0.70]["selected_count"] == 1
    assert report[0.70]["selected_fraction"] == 0.2
    assert report[0.70]["precision"] == 1.0
    assert report[0.70]["avg_actual_return"] == 0.05


def test_validation_selected_threshold_report_selects_best_beat_benchmark_threshold():
    report = build_validation_selected_threshold_report(
        validation_probability_up=np.array([0.40, 0.52, 0.56, 0.62, 0.72]),
        validation_actual_returns=np.array([-0.03, 0.01, -0.02, 0.06, 0.08]),
        test_probability_up=np.array([0.51, 0.63, 0.74]),
        test_actual_returns=np.array([0.01, 0.02, -0.03]),
        thresholds=(0.50, 0.55, 0.60, 0.70),
        min_selected_count=1,
    )

    assert report["thresholds"] == (0.50, 0.55, 0.60, 0.70)
    assert report["min_selected_count"] == 1
    assert report["selection_metric"] == "avg_actual_return"
    assert report["selected_threshold"] == 0.70
    assert (
        report["selected_validation_stats"]
        == report["validation_threshold_report"][0.70]
    )


def test_validation_selected_threshold_report_chooses_eligible_threshold():
    report = build_validation_selected_threshold_report(
        validation_probability_up=np.array([0.40, 0.52, 0.56, 0.62, 0.72]),
        validation_actual_returns=np.array([-0.03, 0.01, -0.02, 0.06, 0.08]),
        test_probability_up=np.array([0.51, 0.63, 0.74]),
        test_actual_returns=np.array([0.01, 0.02, -0.03]),
        thresholds=(0.50, 0.55, 0.60, 0.70),
        min_selected_count=2,
    )

    assert report["selected_threshold"] == 0.60
    assert report["selected_validation_stats"]["selected_count"] == 2


def test_validation_selected_threshold_report_handles_no_eligible_threshold():
    report = build_validation_selected_threshold_report(
        validation_probability_up=np.array([0.40, 0.52, 0.56]),
        validation_actual_returns=np.array([-0.03, 0.01, -0.02]),
        test_probability_up=np.array([0.51, 0.63, 0.74]),
        test_actual_returns=np.array([0.01, 0.02, -0.03]),
        thresholds=(0.50, 0.55, 0.60),
        min_selected_count=10,
    )

    assert report["selected_threshold"] is None
    assert report["selected_validation_stats"] is None
    assert report["selected_test_stats"] is None


def test_validation_selected_threshold_report_evaluates_selected_threshold_on_test():
    report = build_validation_selected_threshold_report(
        validation_probability_up=np.array([0.40, 0.52, 0.56, 0.62, 0.72]),
        validation_actual_returns=np.array([-0.03, 0.01, -0.02, 0.06, 0.08]),
        test_probability_up=np.array([0.51, 0.63, 0.74]),
        test_actual_returns=np.array([0.01, 0.02, -0.03]),
        thresholds=(0.50, 0.55, 0.60, 0.70),
        min_selected_count=2,
    )

    assert report["selected_threshold"] == 0.60
    assert report["selected_test_stats"]["selected_count"] == 2
    assert report["selected_test_stats"]["selected_fraction"] == 2 / 3
    assert report["selected_test_stats"]["precision"] == 0.5
    assert np.isclose(report["selected_test_stats"]["avg_actual_return"], -0.005)


def test_validation_selected_threshold_report_rejects_invalid_inputs():
    valid_probabilities = np.array([0.50, 0.60])
    valid_returns = np.array([0.01, -0.02])

    with pytest.raises(ValueError, match="probability_up"):
        build_validation_selected_threshold_report(
            np.array([]), np.array([]), valid_probabilities, valid_returns
        )
    with pytest.raises(ValueError, match="probability_up"):
        build_validation_selected_threshold_report(
            valid_probabilities, valid_returns, np.array([]), np.array([])
        )
    with pytest.raises(ValueError, match="same length"):
        build_validation_selected_threshold_report(
            np.array([0.50, 0.60]), np.array([0.01]), valid_probabilities, valid_returns
        )
    with pytest.raises(ValueError, match="same length"):
        build_validation_selected_threshold_report(
            valid_probabilities, valid_returns, np.array([0.50]), np.array([0.01, 0.02])
        )
    with pytest.raises(ValueError, match="min_selected_count"):
        build_validation_selected_threshold_report(
            valid_probabilities,
            valid_returns,
            valid_probabilities,
            valid_returns,
            min_selected_count=0,
        )
    with pytest.raises(ValueError, match="selection_metric"):
        build_validation_selected_threshold_report(
            valid_probabilities,
            valid_returns,
            valid_probabilities,
            valid_returns,
            selection_metric="precision",
        )


def test_build_predicted_return_quantile_report_summarizes_excess_return_ranking_buckets():
    predicted_returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    actual_returns = np.array(
        [-0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    )

    report = build_predicted_return_quantile_report(actual_returns, predicted_returns)

    assert report["top_10_pct"]["count"] == 1
    assert report["top_10_pct"]["avg_actual_return"] == 0.07
    assert report["top_10_pct"]["avg_predicted_return"] == 1.0
    assert report["top_10_pct"]["precision"] == 1.0
    assert report["top_20_pct"]["count"] == 2
    assert report["top_20_pct"]["avg_actual_return"] == 0.065
    assert report["top_20_pct"]["avg_predicted_return"] == 0.95
    assert report["bottom_10_pct"]["count"] == 1
    assert report["bottom_10_pct"]["avg_actual_return"] == -0.10
    assert report["bottom_10_pct"]["avg_predicted_return"] == 0.1
    assert report["bottom_10_pct"]["precision"] == 0.0
    assert report["bottom_20_pct"]["count"] == 2
    assert report["bottom_20_pct"]["avg_actual_return"] == -0.07500000000000001
    assert report["bottom_20_pct"]["avg_predicted_return"] == 0.15000000000000002


def test_build_return_correlation_report_summarizes_excess_return_signal():
    actual_returns = np.array([10.0, 20.0, 30.0, 40.0])
    predicted_returns = np.array([1.0, 2.0, 4.0, 3.0])

    report = build_return_correlation_report(actual_returns, predicted_returns)

    assert np.isclose(
        report["pearson"], np.corrcoef(actual_returns, predicted_returns)[0, 1]
    )
    assert np.isclose(report["spearman"], 0.8)


def test_build_combined_signal_report_selects_beat_probability_and_excess_return_rank_overlap():
    probability_up = np.array([0.55, 0.65, 0.70, 0.80, 0.50])
    actual_returns = np.array([-0.02, 0.03, -0.01, 0.05, 0.02])
    predicted_returns = np.array([0.01, 0.04, 0.03, 0.05, 0.02])

    report = build_combined_signal_report(
        probability_up,
        actual_returns,
        predicted_returns,
        probability_threshold=0.60,
        top_return_fraction=0.40,
    )

    assert report["probability_threshold"] == 0.60
    assert report["top_return_fraction"] == 0.40
    assert report["selected_count"] == 2
    assert report["selected_fraction"] == 0.4
    assert np.isclose(report["avg_actual_return"], 0.04)
    assert np.isclose(report["avg_predicted_return"], 0.045)
    assert report["precision"] == 1.0


def test_build_combined_signal_report_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        build_combined_signal_report(
            np.array([0.60, 0.70]),
            np.array([0.01, -0.02]),
            np.array([0.03]),
        )


def make_ranked_selection_metadata():
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "raw_forward_return": [0.10, 0.02, -0.01, 0.03, 0.20, -0.04],
            "benchmark_forward_return": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
            "excess_forward_return": [0.09, 0.01, -0.02, 0.01, 0.18, -0.06],
            "beat_benchmark_target": [1, 1, 0, 1, 1, 0],
            "dailyReturn": [0.01, 0.05, -0.02, 0.04, 0.03, 0.10],
        }
    )


def make_by_year_top_n_metadata():
    return pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "prediction_date": [
                "2022-12-30",
                "2022-12-30",
                "2023-01-03",
                "2023-01-03",
                "2023-06-01",
                "2023-06-01",
            ],
            "raw_forward_return": [0.10, 0.01, -0.02, 0.08, 0.04, 0.01],
            "benchmark_forward_return": [0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
            "excess_forward_return": [0.08, -0.01, -0.03, 0.07, 0.03, 0.00],
            "beat_benchmark_target": [1, 0, 0, 1, 1, 0],
            "dailyReturn": [0.09, 0.01, 0.02, 0.08, 0.04, 0.01],
            "relative_momentum": [0.07, -0.01, 0.01, 0.07, 0.03, 0.00],
        }
    )


def test_same_date_ranking_diagnostics_rank_ic_is_per_date_not_global():
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "excess_forward_return": [0.0, 1.0, 0.0, 1.0],
            "excess_return_rank_pct_by_date": [0.0, 1.0, 0.0, 1.0],
        }
    )
    model_scores = np.array([0.1, 0.9, 0.9, 0.1])

    report = build_same_date_ranking_diagnostics(
        metadata,
        model_scores,
        top_n_values=(1,),
    )

    assert report["available"] is True
    assert report["rank_ic"]["available"] is True
    assert report["rank_ic"]["date_count"] == 2
    assert np.isclose(report["rank_ic"]["mean_rank_ic"], 0.0)
    assert np.isclose(report["rank_ic"]["median_rank_ic"], 0.0)
    assert np.isclose(report["rank_ic"]["rank_ic_positive_rate"], 0.5)


def test_same_date_ranking_diagnostics_ignores_unusable_rank_ic_dates():
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "excess_forward_return": [0.1, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0],
            "excess_return_rank_pct_by_date": [1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0],
        }
    )
    model_scores = np.array([0.8, 0.4, 0.4, 0.1, 0.9, 0.2, 0.7])

    report = build_same_date_ranking_diagnostics(
        metadata,
        model_scores,
        top_n_values=(1,),
    )

    assert report["rank_ic"]["available"] is True
    assert report["rank_ic"]["date_count"] == 1
    assert np.isclose(report["rank_ic"]["mean_rank_ic"], 1.0)
    assert np.isclose(report["rank_ic"]["rank_ic_positive_rate"], 1.0)


def test_same_date_ranking_diagnostics_selected_realized_rank_distribution():
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "excess_forward_return": [0.3, 0.1, -0.2, -0.1, 0.0, 0.2],
            "excess_return_rank_pct_by_date": [0.9, 0.5, 0.1, 0.1, 0.5, 0.9],
        }
    )
    model_scores = np.array([0.9, 0.8, 0.1, 0.8, 0.9, 0.1])

    report = build_same_date_ranking_diagnostics(
        metadata,
        model_scores,
        top_n_values=(1, 2),
    )

    distribution = report["selected_realized_rank_distribution"]
    assert distribution["available"] is True
    assert distribution["top_1"]["selected_count"] == 2
    assert np.isclose(distribution["top_1"]["top_20_realized_rate"], 0.5)
    assert np.isclose(distribution["top_1"]["middle_60_realized_rate"], 0.5)
    assert np.isclose(distribution["top_1"]["bottom_20_realized_rate"], 0.0)
    assert np.isclose(distribution["top_1"]["average_realized_rank_pct"], 0.7)
    assert distribution["top_2"]["selected_count"] == 4
    assert np.isclose(distribution["top_2"]["top_20_realized_rate"], 0.25)
    assert np.isclose(distribution["top_2"]["middle_60_realized_rate"], 0.5)
    assert np.isclose(distribution["top_2"]["bottom_20_realized_rate"], 0.25)
    assert np.isclose(distribution["top_2"]["average_realized_rank_pct"], 0.5)


def test_same_date_ranking_diagnostics_missing_columns_are_unavailable():
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "excess_forward_return": [0.1, -0.1],
        }
    )

    report = build_same_date_ranking_diagnostics(metadata, np.array([0.8, 0.2]))

    assert report["available"] is False
    assert report["rank_ic"]["available"] is False
    assert report["selected_realized_rank_distribution"]["available"] is False
    assert "excess_return_rank_pct_by_date" in report["reason"]


def make_walk_forward_yearly_frame(start_year=2015, end_year=2023):
    rows = []
    for year in range(start_year, end_year + 1):
        rows.append(
            {
                "Ticker": "AAA",
                "prediction_date": pd.Timestamp(f"{year}-06-30"),
                "raw_forward_return": 0.02,
                "benchmark_forward_return": 0.01,
                "excess_forward_return": 0.01,
                "targetReturns": 0.01,
                "beat_benchmark_target": 1,
                "dailyReturn": 0.01,
            }
        )
    return pd.DataFrame(rows)


def assert_nested_reports_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if isinstance(actual_value, dict):
            assert_nested_reports_close(actual_value, expected_value)
        elif isinstance(actual_value, float):
            assert np.isclose(actual_value, expected_value, equal_nan=True)
        else:
            assert actual_value == expected_value


def test_walk_forward_fold_generation_uses_expanding_chronological_windows():
    frame = make_walk_forward_yearly_frame(2015, 2023)

    folds = build_expanding_yearly_walk_forward_folds(
        frame,
        min_train_years=3,
        validation_years=1,
        test_years=1,
    )

    assert len(folds) == 5
    assert folds[0]["train_years"] == [2015, 2016, 2017]
    assert folds[0]["validation_years"] == [2018]
    assert folds[0]["test_years"] == [2019]
    assert folds[1]["train_years"] == [2015, 2016, 2017, 2018]
    assert folds[1]["validation_years"] == [2019]
    assert folds[1]["test_years"] == [2020]
    assert folds[-1]["test_years"] == [2023]
    assert folds[0]["train_date_range"] == {
        "start": "2015-06-30",
        "end": "2017-06-30",
    }
    assert folds[0]["validation_date_range"] == {
        "start": "2018-06-30",
        "end": "2018-06-30",
    }
    assert folds[0]["test_date_range"] == {
        "start": "2019-06-30",
        "end": "2019-06-30",
    }


def test_walk_forward_fold_generation_has_no_within_fold_date_leakage():
    frame = make_walk_forward_yearly_frame(2015, 2022)

    folds = build_expanding_yearly_walk_forward_folds(
        frame,
        min_train_years=3,
        validation_years=1,
        test_years=1,
    )

    for fold in folds:
        train_years = set(fold["train_years"])
        validation_years = set(fold["validation_years"])
        test_years = set(fold["test_years"])
        assert train_years.isdisjoint(validation_years)
        assert train_years.isdisjoint(test_years)
        assert validation_years.isdisjoint(test_years)
        assert max(train_years) < min(validation_years)
        assert max(validation_years) < min(test_years)


def test_walk_forward_split_embargoes_boundary_prediction_dates():
    rows = []
    for year in [2020, 2021, 2022]:
        for day in range(1, 6):
            rows.append(
                {
                    "_source_index": f"{year}-{day}",
                    "Ticker": "AAA",
                    "prediction_date": pd.Timestamp(f"{year}-01-0{day}"),
                    "feature_a": float(day),
                    "targetReturns": 0.01,
                    "raw_forward_return": 0.02,
                    "benchmark_forward_return": 0.01,
                    "excess_forward_return": 0.01,
                    "beat_benchmark_target": 1,
                    "dailyReturn": 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    fold = {
        "train_years": [2020],
        "validation_years": [2021],
        "test_years": [2022],
    }

    split = build_walk_forward_split(
        frame,
        fold,
        feature_columns=["feature_a"],
        prediction_days=2,
    )

    assert split["split_metadata"]["train"]["prediction_date"].max() == pd.Timestamp(
        "2020-01-03"
    )
    assert split["split_metadata"]["val"]["prediction_date"].max() == pd.Timestamp(
        "2021-01-03"
    )
    assert split["split_metadata"]["test"]["prediction_date"].min() == pd.Timestamp(
        "2022-01-01"
    )
    assert len(split["x_train"]) == 3
    assert len(split["x_val"]) == 3
    assert len(split["x_test"]) == 5


def test_walk_forward_aggregate_summary_computes_means_win_rates_and_candidates():
    fold_reports = [
        {
            "selected_candidate_name": "candidate_0_baseline",
            "top_n": {
                "top_5": {
                    "model_excess": 0.02,
                    "model_minus_momentum": 0.01,
                    "model_minus_universe": 0.03,
                },
                "top_10": {
                    "model_excess": 0.01,
                    "model_minus_momentum": -0.01,
                    "model_minus_universe": 0.02,
                },
            },
        },
        {
            "selected_candidate_name": "candidate_1",
            "top_n": {
                "top_5": {
                    "model_excess": 0.04,
                    "model_minus_momentum": -0.02,
                    "model_minus_universe": 0.01,
                },
                "top_10": {
                    "model_excess": 0.03,
                    "model_minus_momentum": 0.02,
                    "model_minus_universe": -0.01,
                },
            },
        },
    ]

    summary = build_walk_forward_aggregate_summary(
        fold_reports,
        top_n_buckets=("top_5", "top_10"),
    )

    assert summary["fold_count"] == 2
    assert summary["selected_candidate_counts"] == {
        "candidate_0_baseline": 1,
        "candidate_1": 1,
    }
    assert np.isclose(summary["top_n"]["top_5"]["average_model_excess"], 0.03)
    assert np.isclose(
        summary["top_n"]["top_5"]["average_model_minus_momentum"],
        -0.005,
    )
    assert summary["top_n"]["top_5"]["fold_win_rate_vs_momentum"] == 0.5
    assert summary["top_n"]["top_5"]["fold_win_rate_vs_universe"] == 1.0
    assert summary["top_n"]["top_10"]["fold_win_rate_vs_momentum"] == 0.5
    assert summary["top_n"]["top_10"]["fold_win_rate_vs_universe"] == 0.5


def test_walk_forward_fold_report_records_selected_candidate_and_bucket_metrics():
    fold = {
        "fold_index": 0,
        "train_date_range": {"start": "2015-01-01", "end": "2019-12-31"},
        "validation_date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "test_date_range": {"start": "2021-01-01", "end": "2021-12-31"},
        "train_years": [2015, 2016, 2017, 2018, 2019],
        "validation_years": [2020],
        "test_years": [2021],
    }
    selection_report = {
        "selected_candidate_id": 1,
        "selected_candidate_name": "candidate_1",
        "selected_validation_top_n_mean_excess_return": 0.03,
    }
    basket_backtest = {
        "top_5": {
            "model": {"average_basket_excess_return": 0.04},
            "momentum_baseline": {"average_basket_excess_return": 0.01},
            "universe": {"average_basket_excess_return": 0.02},
        }
    }

    report = build_walk_forward_fold_report(
        fold,
        selection_report,
        basket_backtest,
        top_n_buckets=("top_5",),
    )

    assert report["selected_candidate_id"] == 1
    assert report["selected_candidate_name"] == "candidate_1"
    assert report["validation_selection_score"] == 0.03
    assert report["top_n"]["top_5"]["model_excess"] == 0.04
    assert np.isclose(report["top_n"]["top_5"]["model_minus_momentum"], 0.03)
    assert np.isclose(report["top_n"]["top_5"]["model_minus_universe"], 0.02)


def test_top_n_ranked_selection_is_per_date_not_global():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
    )

    top_1 = report["top_1"]["model"]
    assert top_1["date_count"] == 2
    assert top_1["selected_row_count"] == 2
    assert np.isclose(top_1["average_selected_raw_forward_return"], 0.03)
    assert np.isclose(top_1["average_selected_benchmark_forward_return"], 0.015)
    assert np.isclose(top_1["average_selected_excess_return_vs_benchmark"], 0.015)
    assert np.isclose(top_1["beat_benchmark_rate"], 0.5)
    assert np.isclose(top_1["average_predicted_excess_return"], 0.60)


def test_top_n_selection_reports_match_compatibility_wrappers():
    metadata = make_ranked_selection_metadata()
    metadata["relative_momentum"] = [0.01, 0.05, -0.02, 0.04, 0.03, 0.10]
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    combined_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
    )
    ranked_report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
    )
    basket_report = build_top_n_basket_backtest_report(
        metadata,
        predicted_excess_returns,
        top_ns=(1, 2),
        random_seed=7,
        random_trials=5,
    )

    assert combined_report["ranked_selection"] == ranked_report
    assert combined_report["basket_backtest"] == basket_report


def test_model_only_top_n_basket_backtest_report_skips_baselines():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_model_only_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_n_values=(1,),
    )

    assert set(report) == {"top_1"}
    assert set(report["top_1"]) == {"model"}
    assert np.isclose(
        report["top_1"]["model"]["average_basket_excess_return"],
        (0.09 + -0.06) / 2,
    )


def test_top_n_basket_bootstrap_confidence_intervals_resample_prediction_dates():
    metadata = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "AAA", "BBB"],
            "prediction_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "raw_forward_return": [0.10, 0.00, 0.20, 0.00],
            "benchmark_forward_return": [0.00, 0.00, 0.00, 0.00],
            "excess_forward_return": [0.10, 0.00, 0.20, 0.00],
            "beat_benchmark_target": [1, 0, 1, 0],
            "dailyReturn": [0.10, 0.00, 0.20, 0.00],
        }
    )
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.60])

    report = build_top_n_selection_reports(
        metadata,
        ranked_predictions,
        top_n_values=(2,),
        random_trials=2,
        bootstrap_trials=200,
        bootstrap_seed=7,
    )

    model_ci = report["basket_backtest"]["top_2"][
        "bootstrap_confidence_intervals"
    ]["model_average_basket_excess_return"]
    assert np.isclose(model_ci["mean"], 0.075)
    assert 0.05 <= model_ci["ci_lower"] <= model_ci["mean"]
    assert model_ci["mean"] <= model_ci["ci_upper"] <= 0.10
    assert model_ci["resampled_dates"] == 2


def test_top_n_basket_bootstrap_confidence_intervals_are_deterministic():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    first_report = build_top_n_selection_reports(
        metadata,
        ranked_predictions,
        top_n_values=(1,),
        random_seed=11,
        random_trials=3,
        bootstrap_trials=100,
        bootstrap_seed=123,
    )
    second_report = build_top_n_selection_reports(
        metadata,
        ranked_predictions,
        top_n_values=(1,),
        random_seed=11,
        random_trials=3,
        bootstrap_trials=100,
        bootstrap_seed=123,
    )

    assert (
        first_report["basket_backtest"]["top_1"]["bootstrap_confidence_intervals"]
        == second_report["basket_backtest"]["top_1"]["bootstrap_confidence_intervals"]
    )


def test_top_n_basket_bootstrap_confidence_intervals_are_additive():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_selection_reports(
        metadata,
        ranked_predictions,
        top_n_values=(1,),
        random_seed=7,
        random_trials=2,
        bootstrap_trials=50,
    )
    basket_top_1 = report["basket_backtest"]["top_1"]

    assert {
        "model",
        "random_baseline",
        "momentum_baseline",
        "relative_momentum_baseline",
        "universe",
        "benchmark",
        "bootstrap_confidence_intervals",
    }.issubset(basket_top_1)
    ci_report = basket_top_1["bootstrap_confidence_intervals"]
    assert set(ci_report) == {
        "model_average_basket_excess_return",
        "momentum_baseline_average_basket_excess_return",
        "relative_momentum_baseline_average_basket_excess_return",
        "universe_average_basket_excess_return",
        "model_minus_momentum_baseline_average_basket_excess_return",
        "model_minus_relative_momentum_baseline_average_basket_excess_return",
        "model_minus_universe_average_basket_excess_return",
    }
    model_ci = ci_report["model_average_basket_excess_return"]
    assert model_ci["bootstrap_trials"] == 50
    assert model_ci["confidence_level"] == 0.95
    assert model_ci["ci_lower"] <= model_ci["mean"] <= model_ci["ci_upper"]


def test_basket_bootstrap_confidence_intervals_handle_empty_and_single_date():
    empty_report = _bootstrap_basket_confidence_intervals(
        pd.DataFrame(columns=["prediction_date", "basket_excess_return"]),
        bootstrap_trials=10,
    )
    empty_model_ci = empty_report["model_average_basket_excess_return"]
    assert empty_model_ci["available"] is False
    assert empty_model_ci["resampled_dates"] == 0
    assert "No prediction dates" in empty_model_ci["reason"]

    single_date_report = _bootstrap_basket_confidence_intervals(
        pd.DataFrame(
            {
                "prediction_date": pd.to_datetime(["2024-01-01"]),
                "basket_excess_return": [0.04],
            }
        ),
        bootstrap_trials=10,
    )
    single_model_ci = single_date_report["model_average_basket_excess_return"]
    assert single_model_ci["mean"] == 0.04
    assert single_model_ci["ci_lower"] == 0.04
    assert single_model_ci["ci_upper"] == 0.04
    assert "Fewer than two prediction dates" in single_model_ci["note"]


def test_probability_ranked_top_n_selection_reports_rank_by_probability():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])
    classifier_probabilities = np.array([0.10, 0.90, 0.20, 0.30, 0.80, 0.70])

    regressor_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=7,
        random_trials=5,
    )
    probability_report = build_probability_ranked_top_n_selection_reports(
        metadata,
        classifier_probabilities,
        top_n_values=(1,),
        random_seed=7,
        random_trials=5,
    )

    probability_top_1 = probability_report["ranked_selection"]["top_1"]["model"]
    regressor_top_1 = regressor_report["ranked_selection"]["top_1"]["model"]
    assert np.isclose(
        probability_top_1["average_selected_raw_forward_return"],
        (0.02 + 0.20) / 2,
    )
    assert np.isclose(
        probability_top_1["average_predicted_beat_benchmark_probability"],
        (0.90 + 0.80) / 2,
    )
    assert not np.isclose(
        probability_top_1["average_selected_raw_forward_return"],
        regressor_top_1["average_selected_raw_forward_return"],
    )
    assert probability_report.keys() == {
        "ranked_selection",
        "basket_backtest",
        "basket_backtest_by_year",
    }


def test_top_n_selection_reports_include_by_year_basket_report_and_existing_keys():
    metadata = make_by_year_top_n_metadata()
    predicted_excess_returns = np.array([0.90, 0.10, 0.20, 0.80, 0.70, 0.60])

    report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        random_seed=7,
        random_trials=2,
        random_trial_workers=1,
    )

    assert report.keys() == {
        "ranked_selection",
        "basket_backtest",
        "basket_backtest_by_year",
    }
    assert set(report["basket_backtest_by_year"]) == {"top_5", "top_10", "top_20"}
    assert set(report["basket_backtest_by_year"]["top_5"]) == {"2022", "2023"}
    assert all(
        isinstance(year, str)
        for year in report["basket_backtest_by_year"]["top_5"]
    )
    assert set(report["basket_backtest_by_year"]["top_5"]["2023"]) == {
        "model",
        "random_baseline",
        "momentum_baseline",
        "relative_momentum_baseline",
        "universe",
        "benchmark",
    }


def test_top_n_basket_backtest_by_year_uses_prediction_date_year_and_selected_groups():
    metadata = make_by_year_top_n_metadata()
    predicted_excess_returns = np.array([0.90, 0.10, 0.20, 0.80, 0.70, 0.60])

    report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=7,
        random_trials=2,
        random_trial_workers=1,
    )

    by_year = report["basket_backtest_by_year"]["top_1"]
    assert set(by_year) == {"2022", "2023"}
    assert by_year["2022"]["model"]["evaluated_dates"] == 1
    assert by_year["2023"]["model"]["evaluated_dates"] == 2
    assert np.isclose(
        by_year["2022"]["model"]["average_basket_excess_return"],
        0.08,
    )
    assert np.isclose(
        by_year["2023"]["model"]["average_basket_excess_return"],
        (0.07 + 0.03) / 2,
    )
    assert np.isclose(
        by_year["2023"]["universe"]["average_basket_excess_return"],
        ((-0.03 + 0.07) / 2 + (0.03 + 0.00) / 2) / 2,
    )
    assert by_year["2022"]["benchmark"]["average_basket_excess_return"] == 0.0


def test_top_n_basket_backtest_by_year_random_baseline_is_deterministic_with_parallel_workers():
    metadata = make_by_year_top_n_metadata()
    predicted_excess_returns = np.array([0.90, 0.10, 0.20, 0.80, 0.70, 0.60])

    first_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=11,
        random_trials=6,
        random_trial_workers=2,
    )
    second_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=11,
        random_trials=6,
        random_trial_workers=2,
    )

    assert_nested_reports_close(
        second_report["basket_backtest_by_year"],
        first_report["basket_backtest_by_year"],
    )
    random_baseline = first_report["basket_backtest_by_year"]["top_1"]["2023"][
        "random_baseline"
    ]
    assert random_baseline["random_seed"] == 11
    assert random_baseline["random_trials"] == 6


def test_top_n_basket_backtest_by_year_random_baseline_is_deterministic_with_single_worker():
    metadata = make_by_year_top_n_metadata()
    predicted_excess_returns = np.array([0.90, 0.10, 0.20, 0.80, 0.70, 0.60])

    first_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=11,
        random_trials=6,
        random_trial_workers=1,
    )
    second_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=11,
        random_trials=6,
        random_trial_workers=1,
    )

    assert_nested_reports_close(
        second_report["basket_backtest_by_year"],
        first_report["basket_backtest_by_year"],
    )


def test_top_n_random_baselines_match_across_worker_counts_with_independent_trial_seeds():
    # The random baseline uses deterministic independent per-trial seeds, so
    # worker count changes scheduling only, not selected trial baskets.
    metadata = make_by_year_top_n_metadata()
    predicted_excess_returns = np.array([0.90, 0.10, 0.20, 0.80, 0.70, 0.60])

    single_worker_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=11,
        random_trials=6,
        random_trial_workers=1,
    )
    parallel_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=11,
        random_trials=6,
        random_trial_workers=2,
    )

    assert_nested_reports_close(
        parallel_report["basket_backtest_by_year"]["top_1"]["2023"][
            "random_baseline"
        ],
        single_worker_report["basket_backtest_by_year"]["top_1"]["2023"][
            "random_baseline"
        ],
    )
    assert_nested_reports_close(
        parallel_report["basket_backtest"]["top_1"]["random_baseline"],
        single_worker_report["basket_backtest"]["top_1"]["random_baseline"],
    )


def test_probability_ranked_top_n_selection_reports_include_by_year_basket_report():
    metadata = make_by_year_top_n_metadata()
    classifier_probabilities = np.array([0.10, 0.90, 0.20, 0.80, 0.70, 0.60])

    report = build_probability_ranked_top_n_selection_reports(
        metadata,
        classifier_probabilities,
        top_n_values=(1,),
        random_seed=7,
        random_trials=2,
        random_trial_workers=1,
    )

    assert "basket_backtest_by_year" in report
    assert set(report["basket_backtest_by_year"]["top_1"]) == {"2022", "2023"}


def test_probability_ranked_top_n_selection_reports_are_exported():
    assert (
        evaluation.build_probability_ranked_top_n_selection_reports
        is build_probability_ranked_top_n_selection_reports
    )
    assert (
        evaluator.build_probability_ranked_top_n_selection_reports
        is build_probability_ranked_top_n_selection_reports
    )
    assert "build_probability_ranked_top_n_selection_reports" in evaluation.__all__
    assert "build_probability_ranked_top_n_selection_reports" in evaluator.__all__


def test_same_date_ranking_diagnostics_are_exported():
    assert (
        evaluation.build_same_date_ranking_diagnostics
        is build_same_date_ranking_diagnostics
    )
    assert (
        evaluator.build_same_date_ranking_diagnostics
        is build_same_date_ranking_diagnostics
    )
    assert "build_same_date_ranking_diagnostics" in evaluation.__all__
    assert "build_same_date_ranking_diagnostics" in evaluator.__all__


def test_top_n_selection_reports_returns_direct_expected_values():
    metadata = make_ranked_selection_metadata()
    metadata["relative_momentum"] = [0.01, 0.05, -0.02, 0.04, 0.03, 0.10]
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
    )

    ranked_top_1 = report["ranked_selection"]["top_1"]
    basket_top_1 = report["basket_backtest"]["top_1"]
    ranked_top_2 = report["ranked_selection"]["top_2"]
    basket_top_2 = report["basket_backtest"]["top_2"]

    assert np.isclose(
        ranked_top_1["model"]["average_selected_raw_forward_return"],
        (0.10 + -0.04) / 2,
    )
    assert np.isclose(
        ranked_top_1["model"]["average_selected_excess_return_vs_benchmark"],
        (0.09 + -0.06) / 2,
    )
    assert np.isclose(
        ranked_top_1["model"]["average_predicted_excess_return"],
        (0.90 + 0.30) / 2,
    )
    assert np.isclose(
        basket_top_1["model"]["average_basket_raw_return"],
        (0.10 + -0.04) / 2,
    )
    assert np.isclose(
        basket_top_1["model"]["average_basket_excess_return"],
        (0.09 + -0.06) / 2,
    )

    assert np.isclose(
        ranked_top_2["model"]["average_selected_raw_forward_return"],
        ((0.10 + 0.02) / 2 + (-0.04 + 0.20) / 2) / 2,
    )
    assert np.isclose(
        ranked_top_2["model"]["average_selected_excess_return_vs_benchmark"],
        ((0.09 + 0.01) / 2 + (-0.06 + 0.18) / 2) / 2,
    )
    assert np.isclose(
        basket_top_2["model"]["average_basket_raw_return"],
        ((0.10 + 0.02) / 2 + (-0.04 + 0.20) / 2) / 2,
    )
    assert np.isclose(
        basket_top_2["model"]["average_basket_excess_return"],
        ((0.09 + 0.01) / 2 + (-0.06 + 0.18) / 2) / 2,
    )

    expected_universe_raw = (
        (0.10 + 0.02 + -0.01) / 3 + (0.03 + 0.20 + -0.04) / 3
    ) / 2
    expected_universe_excess = (
        (0.09 + 0.01 + -0.02) / 3 + (0.01 + 0.18 + -0.06) / 3
    ) / 2
    assert np.isclose(
        ranked_top_1["universe"]["average_equal_weight_universe_forward_return"],
        expected_universe_raw,
    )
    assert np.isclose(
        ranked_top_1["universe"]["average_excess_return_vs_benchmark"],
        expected_universe_excess,
    )
    assert np.isclose(
        basket_top_1["universe"]["average_basket_raw_return"],
        expected_universe_raw,
    )
    assert np.isclose(
        basket_top_1["universe"]["average_basket_excess_return"],
        expected_universe_excess,
    )
    assert np.isclose(
        basket_top_1["benchmark"]["average_basket_raw_return"],
        (0.01 + 0.02) / 2,
    )
    assert basket_top_1["benchmark"]["average_basket_excess_return"] == 0.0

    assert ranked_top_1["random_baseline"]["random_seed"] == 7
    assert ranked_top_1["random_baseline"]["random_baseline_trials"] == 5
    assert basket_top_1["random_baseline"]["random_seed"] == 7
    assert basket_top_1["random_baseline"]["random_trials"] == 5
    assert ranked_top_1["momentum_baseline"]["available"] is True
    assert ranked_top_1["momentum_baseline"]["momentum_score_column"] == "dailyReturn"
    assert basket_top_1["relative_momentum_baseline"]["available"] is True
    assert (
        basket_top_1["relative_momentum_baseline"][
            "relative_momentum_score_column"
        ]
        == "relative_momentum"
    )


def test_top_n_random_trials_workers_greater_than_one_are_deterministic():
    metadata = make_ranked_selection_metadata()
    metadata["relative_momentum"] = [0.01, 0.05, -0.02, 0.04, 0.03, 0.10]
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    first_parallel_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
        random_trial_workers=2,
    )
    second_parallel_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
        random_trial_workers=2,
    )

    assert_nested_reports_close(second_parallel_report, first_parallel_report)
    ranked_random = first_parallel_report["ranked_selection"]["top_1"][
        "random_baseline"
    ]
    basket_random = first_parallel_report["basket_backtest"]["top_1"][
        "random_baseline"
    ]
    assert ranked_random["random_seed"] == 7
    assert ranked_random["random_baseline_trials"] == 5
    assert basket_random["random_seed"] == 7
    assert basket_random["random_trials"] == 5


def test_top_n_random_trials_worker_one_is_deterministic_with_independent_trial_seeds():
    metadata = make_ranked_selection_metadata()
    metadata["relative_momentum"] = [0.01, 0.05, -0.02, 0.04, 0.03, 0.10]
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    first_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
        random_trial_workers=1,
    )
    second_report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
        random_seed=7,
        random_trials=5,
        random_trial_workers=1,
    )

    assert_nested_reports_close(second_report, first_report)


def test_top_n_random_trial_worker_one_uses_sequential_no_pool_path(monkeypatch):
    class RaisingProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ProcessPoolExecutor should not be used")

    monkeypatch.setattr(
        baselines,
        "ProcessPoolExecutor",
        RaisingProcessPoolExecutor,
    )
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=7,
        random_trials=5,
        random_trial_workers=1,
    )

    ranked_random = report["ranked_selection"]["top_1"]["random_baseline"]
    basket_random = report["basket_backtest"]["top_1"]["random_baseline"]
    assert ranked_random["date_count"] == 2
    assert ranked_random["selected_row_count"] == 2
    assert ranked_random["random_seed"] == 7
    assert ranked_random["random_baseline_trials"] == 5
    assert basket_random["random_seed"] == 7
    assert basket_random["random_trials"] == 5


def test_top_n_random_trial_workers_greater_than_one_uses_process_pool(monkeypatch):
    pool_calls = []

    class RecordingProcessPoolExecutor:
        def __init__(self, max_workers):
            pool_calls.append(("init", max_workers))

        def __enter__(self):
            pool_calls.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, traceback):
            pool_calls.append(("exit", None))

        def map(self, worker, tasks):
            task_list = list(tasks)
            pool_calls.append(("map", len(task_list)))
            return [worker(task) for task in task_list]

    monkeypatch.setattr(
        baselines,
        "ProcessPoolExecutor",
        RecordingProcessPoolExecutor,
    )
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    build_top_n_selection_reports(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=7,
        random_trials=5,
        random_trial_workers=2,
    )

    assert pool_calls == [("init", 2), ("enter", None), ("map", 2), ("exit", None)]


def test_top_n_ranked_selection_uses_min_available_candidates_per_date():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(5,),
    )

    assert report["top_5"]["model"]["date_count"] == 2
    assert report["top_5"]["model"]["selected_row_count"] == 6
    assert report["top_5"]["model"]["average_number_of_candidates_per_date"] == 3.0
    assert report["top_5"]["random_baseline"]["selected_row_count"] == 6


def test_top_n_ranked_selection_computes_universe_relative_return():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
    )

    selected_raw_mean = (0.10 + -0.04) / 2
    universe_date_1 = (0.10 + 0.02 + -0.01) / 3
    universe_date_2 = (0.03 + 0.20 + -0.04) / 3
    expected_universe_mean = (universe_date_1 + universe_date_2) / 2
    assert np.isclose(
        report["top_1"]["model"]["average_equal_weight_universe_forward_return"],
        expected_universe_mean,
    )
    assert np.isclose(
        report["top_1"]["model"]["average_selected_return_minus_universe_return"],
        selected_raw_mean - expected_universe_mean,
    )
    assert np.isclose(
        report["top_1"]["universe"]["average_equal_weight_universe_forward_return"],
        expected_universe_mean,
    )


def test_top_n_ranked_selection_reports_multiple_buckets():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1, 2),
    )

    assert set(report) == {"top_1", "top_2"}
    assert set(report["top_1"]) == {
        "model",
        "random_baseline",
        "momentum_baseline",
        "relative_momentum_baseline",
        "universe",
    }
    assert report["top_1"]["model"]["selected_row_count"] == 2
    assert report["top_2"]["model"]["selected_row_count"] == 4
    assert np.isclose(
        report["top_2"]["model"]["median_selected_excess_return_vs_benchmark"],
        0.05,
    )
    assert report["top_2"]["model"]["positive_raw_return_rate"] == 0.75


def test_top_n_ranked_selection_ignores_benchmark_rows_if_present():
    metadata = make_ranked_selection_metadata()
    spy_row = pd.DataFrame(
        {
            "Ticker": ["SPY"],
            "prediction_date": [pd.Timestamp("2024-01-01")],
            "raw_forward_return": [0.50],
            "benchmark_forward_return": [0.50],
            "excess_forward_return": [0.00],
            "beat_benchmark_target": [0],
            "dailyReturn": [1.00],
        }
    )
    metadata = pd.concat([metadata, spy_row], ignore_index=True)
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30, 9.99])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
    )

    assert report["top_1"]["model"]["selected_row_count"] == 2
    assert report["top_1"]["random_baseline"]["selected_row_count"] == 2
    assert report["top_1"]["universe"]["candidate_row_count"] == 6
    assert np.isclose(
        report["top_1"]["model"]["average_selected_raw_forward_return"],
        0.03,
    )


def test_top_n_random_baseline_is_deterministic_with_fixed_seed():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    first_report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=7,
        random_trials=5,
    )
    second_report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=7,
        random_trials=5,
    )

    assert (
        first_report["top_1"]["random_baseline"]
        == second_report["top_1"]["random_baseline"]
    )
    assert first_report["top_1"]["random_baseline"]["random_seed"] == 7
    assert first_report["top_1"]["random_baseline"]["random_baseline_trials"] == 5


def test_top_n_random_baseline_selects_within_date_not_globally():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        random_seed=42,
        random_trials=1,
    )

    random_baseline = report["top_1"]["random_baseline"]
    assert random_baseline["date_count"] == 2
    assert random_baseline["selected_row_count"] == 2
    assert random_baseline["average_number_of_candidates_per_date"] == 3.0
    assert np.isclose(
        random_baseline["average_selected_raw_forward_return"],
        (0.10 + -0.04) / 2,
    )


def test_top_n_momentum_baseline_ranks_within_date():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
    )

    momentum_baseline = report["top_1"]["momentum_baseline"]
    assert momentum_baseline["available"] is True
    assert momentum_baseline["momentum_score_column"] == "dailyReturn"
    assert momentum_baseline["date_count"] == 2
    assert momentum_baseline["selected_row_count"] == 2
    assert np.isclose(
        momentum_baseline["average_selected_raw_forward_return"],
        (0.02 + -0.04) / 2,
    )
    assert np.isclose(
        momentum_baseline["average_dailyReturn"],
        (0.05 + 0.10) / 2,
    )


def test_top_n_ranked_selection_uses_horizon_matched_momentum_column_when_available():
    metadata = make_ranked_selection_metadata()
    metadata["momentum_10d"] = [0.99, 0.01, 0.02, 0.01, 0.02, 0.99]
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        prediction_days=10,
    )

    momentum_baseline = report["top_1"]["momentum_baseline"]
    assert momentum_baseline["available"] is True
    assert momentum_baseline["momentum_score_column"] == "momentum_10d"
    assert np.isclose(
        momentum_baseline["average_selected_raw_forward_return"],
        (0.10 + -0.04) / 2,
    )
    assert np.isclose(
        momentum_baseline["average_momentum_10d"],
        (0.99 + 0.99) / 2,
    )


def test_top_n_ranked_selection_includes_relative_momentum_baseline_when_available():
    metadata = make_ranked_selection_metadata()
    metadata["relative_momentum_10d"] = [0.99, 0.01, 0.02, 0.01, 0.02, 0.99]
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        prediction_days=10,
    )

    relative_momentum = report["top_1"]["relative_momentum_baseline"]
    assert relative_momentum["available"] is True
    assert relative_momentum["relative_momentum_score_column"] == "relative_momentum_10d"
    assert np.isclose(
        relative_momentum["average_selected_raw_forward_return"],
        (0.10 + -0.04) / 2,
    )
    assert np.isclose(
        relative_momentum["average_relative_momentum_10d"],
        (0.99 + 0.99) / 2,
    )


def test_top_n_ranked_selection_marks_horizon_momentum_unavailable_when_missing():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        prediction_days=10,
    )

    momentum_baseline = report["top_1"]["momentum_baseline"]
    assert momentum_baseline["available"] is False
    assert momentum_baseline["momentum_score_column"] == "momentum_10d"
    assert "not present" in momentum_baseline["reason"]
    assert np.isnan(momentum_baseline["average_selected_raw_forward_return"])


def test_top_n_ranked_selection_relative_momentum_unavailable_when_horizon_column_missing():
    metadata = make_ranked_selection_metadata()
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
        prediction_days=10,
    )

    relative_momentum = report["top_1"]["relative_momentum_baseline"]
    assert relative_momentum["available"] is False
    assert relative_momentum["relative_momentum_score_column"] == "relative_momentum_10d"
    assert "not present" in relative_momentum["reason"]
    assert np.isnan(relative_momentum["average_selected_raw_forward_return"])


def test_top_n_momentum_baseline_reports_missing_score_column_explicitly():
    metadata = make_ranked_selection_metadata().drop(columns=["dailyReturn"])
    predicted_excess_returns = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_ranked_selection_report(
        metadata,
        predicted_excess_returns,
        top_n_values=(1,),
    )

    momentum_baseline = report["top_1"]["momentum_baseline"]
    assert momentum_baseline["available"] is False
    assert momentum_baseline["momentum_score_column"] == "dailyReturn"
    assert "not present" in momentum_baseline["reason"]
    assert np.isnan(momentum_baseline["average_selected_raw_forward_return"])


def test_top_n_ranked_selection_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        build_top_n_ranked_selection_report(
            make_ranked_selection_metadata(),
            np.array([0.10, 0.20]),
        )


def test_top_n_ranked_selection_rejects_missing_metadata_columns():
    metadata = make_ranked_selection_metadata().drop(columns=["raw_forward_return"])

    with pytest.raises(ValueError, match="required columns"):
        build_top_n_ranked_selection_report(
            metadata,
            np.arange(len(metadata), dtype=float),
        )


def test_top_n_ranked_selection_rejects_duplicate_date_ticker_candidates():
    metadata = pd.concat(
        [
            make_ranked_selection_metadata(),
            make_ranked_selection_metadata().iloc[[0]],
        ],
        ignore_index=True,
    )
    predicted_excess_returns = np.arange(len(metadata), dtype=float)

    with pytest.raises(ValueError, match="duplicate candidate rows"):
        build_top_n_ranked_selection_report(
            metadata,
            predicted_excess_returns,
            top_n_values=(1,),
        )


def test_top_n_basket_backtest_selects_model_top_n_within_each_date():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
    )

    model = report["top_1"]["model"]
    assert model["evaluated_dates"] == 2
    assert model["average_selected_count"] == 1.0
    assert np.isclose(model["average_basket_raw_return"], (0.10 + -0.04) / 2)
    assert np.isclose(model["average_basket_benchmark_return"], 0.015)
    assert np.isclose(model["average_basket_excess_return"], (0.09 + -0.06) / 2)
    assert model["positive_basket_return_rate"] == 0.5
    assert model["beat_benchmark_rate"] == 0.5


def test_top_n_basket_backtest_averages_equal_weight_baskets_per_date():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(2,),
    )

    date_1_basket_raw = (0.10 + 0.02) / 2
    date_2_basket_raw = (0.20 + -0.04) / 2
    date_1_basket_excess = (0.09 + 0.01) / 2
    date_2_basket_excess = (0.18 + -0.06) / 2
    model = report["top_2"]["model"]
    assert model["evaluated_dates"] == 2
    assert model["average_selected_count"] == 2.0
    assert np.isclose(
        model["average_basket_raw_return"],
        (date_1_basket_raw + date_2_basket_raw) / 2,
    )
    assert np.isclose(
        model["average_basket_excess_return"],
        (date_1_basket_excess + date_2_basket_excess) / 2,
    )


def test_top_n_basket_backtest_random_baseline_is_deterministic():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    first_report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
        random_seed=7,
        random_trials=5,
    )
    second_report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
        random_seed=7,
        random_trials=5,
    )

    assert (
        first_report["top_1"]["random_baseline"]
        == second_report["top_1"]["random_baseline"]
    )
    assert first_report["top_1"]["random_baseline"]["random_seed"] == 7
    assert first_report["top_1"]["random_baseline"]["random_trials"] == 5


def test_top_n_basket_backtest_momentum_baseline_ranks_by_daily_return():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
    )

    momentum = report["top_1"]["momentum_baseline"]
    assert momentum["available"] is True
    assert momentum["momentum_score_column"] == "dailyReturn"
    assert momentum["evaluated_dates"] == 2
    assert np.isclose(momentum["average_basket_raw_return"], (0.02 + -0.04) / 2)
    assert np.isclose(momentum["average_basket_excess_return"], (0.01 + -0.06) / 2)


def test_top_n_basket_backtest_uses_horizon_matched_momentum_column_when_available():
    metadata = make_ranked_selection_metadata()
    metadata["momentum_20d"] = [0.99, 0.01, 0.02, 0.01, 0.02, 0.99]
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
        prediction_days=20,
    )

    momentum = report["top_1"]["momentum_baseline"]
    assert momentum["available"] is True
    assert momentum["momentum_score_column"] == "momentum_20d"
    assert np.isclose(momentum["average_basket_raw_return"], (0.10 + -0.04) / 2)


def test_top_n_basket_backtest_includes_relative_momentum_baseline_when_available():
    metadata = make_ranked_selection_metadata()
    metadata["relative_momentum_20d"] = [0.99, 0.01, 0.02, 0.01, 0.02, 0.99]
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
        prediction_days=20,
    )

    relative_momentum = report["top_1"]["relative_momentum_baseline"]
    assert relative_momentum["available"] is True
    assert relative_momentum["relative_momentum_score_column"] == "relative_momentum_20d"
    assert np.isclose(
        relative_momentum["average_basket_raw_return"],
        (0.10 + -0.04) / 2,
    )


def test_top_n_basket_backtest_marks_horizon_momentum_unavailable_when_missing():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
        prediction_days=20,
    )

    momentum = report["top_1"]["momentum_baseline"]
    assert momentum["available"] is False
    assert momentum["momentum_score_column"] == "momentum_20d"
    assert "not present" in momentum["reason"]
    assert np.isnan(momentum["average_basket_raw_return"])


def test_top_n_basket_backtest_relative_momentum_unavailable_when_horizon_column_missing():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
        prediction_days=20,
    )

    relative_momentum = report["top_1"]["relative_momentum_baseline"]
    assert relative_momentum["available"] is False
    assert relative_momentum["relative_momentum_score_column"] == "relative_momentum_20d"
    assert "not present" in relative_momentum["reason"]
    assert np.isnan(relative_momentum["average_basket_raw_return"])


def test_top_n_basket_backtest_momentum_baseline_reports_missing_score_column():
    metadata = make_ranked_selection_metadata().drop(columns=["dailyReturn"])
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
    )

    momentum = report["top_1"]["momentum_baseline"]
    assert momentum["available"] is False
    assert momentum["momentum_score_column"] == "dailyReturn"
    assert "not present" in momentum["reason"]
    assert np.isnan(momentum["average_basket_raw_return"])


def test_top_n_basket_backtest_excludes_spy_from_candidate_selections():
    metadata = make_ranked_selection_metadata()
    spy_row = pd.DataFrame(
        {
            "Ticker": ["SPY"],
            "prediction_date": [pd.Timestamp("2024-01-01")],
            "raw_forward_return": [0.50],
            "benchmark_forward_return": [0.50],
            "excess_forward_return": [0.00],
            "beat_benchmark_target": [0],
            "dailyReturn": [1.00],
        }
    )
    metadata = pd.concat([metadata, spy_row], ignore_index=True)
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30, 9.99])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
    )

    assert np.isclose(
        report["top_1"]["model"]["average_basket_raw_return"],
        (0.10 + -0.04) / 2,
    )
    assert np.isclose(report["top_1"]["universe"]["average_selected_count"], 3.0)
    assert np.isclose(report["top_1"]["universe"]["average_basket_raw_return"], 0.05)


def test_top_n_basket_backtest_uses_all_candidates_when_fewer_than_n():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(5,),
    )

    model = report["top_5"]["model"]
    assert model["evaluated_dates"] == 2
    assert model["average_selected_count"] == 3.0
    assert np.isclose(model["average_basket_raw_return"], 0.05)
    assert np.isclose(model["average_basket_excess_return"], 0.035)


def test_top_n_basket_backtest_includes_universe_and_benchmark_summaries():
    metadata = make_ranked_selection_metadata()
    ranked_predictions = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.30])

    report = build_top_n_basket_backtest_report(
        metadata,
        ranked_predictions,
        top_ns=(1,),
    )

    assert set(report["top_1"]) == {
        "model",
        "random_baseline",
        "momentum_baseline",
        "relative_momentum_baseline",
        "universe",
        "benchmark",
        "bootstrap_confidence_intervals",
    }
    assert np.isclose(report["top_1"]["universe"]["average_basket_raw_return"], 0.05)
    assert np.isclose(
        report["top_1"]["benchmark"]["average_basket_raw_return"],
        0.015,
    )
    assert report["top_1"]["benchmark"]["average_basket_excess_return"] == 0.0


def test_top_n_basket_backtest_rejects_invalid_inputs():
    metadata = make_ranked_selection_metadata()

    with pytest.raises(ValueError, match="same length"):
        build_top_n_basket_backtest_report(metadata, np.array([0.10, 0.20]))

    with pytest.raises(ValueError, match="required columns"):
        build_top_n_basket_backtest_report(
            metadata.drop(columns=["raw_forward_return"]),
            np.arange(len(metadata), dtype=float),
        )

    with pytest.raises(ValueError, match="random_trials"):
        build_top_n_basket_backtest_report(
            metadata,
            np.arange(len(metadata), dtype=float),
            random_trials=0,
        )

    with pytest.raises(ValueError, match="random_trial_workers"):
        build_top_n_basket_backtest_report(
            metadata,
            np.arange(len(metadata), dtype=float),
            random_trial_workers=0,
        )


def test_predicted_return_reports_reject_empty_and_mismatched_inputs():
    with pytest.raises(ValueError, match="actual_returns"):
        build_predicted_return_quantile_report(np.array([]), np.array([]))

    for report_builder in [
        build_predicted_return_quantile_report,
        build_return_correlation_report,
    ]:
        with pytest.raises(ValueError, match="same length"):
            report_builder(np.array([0.01, -0.02]), np.array([0.01]))
