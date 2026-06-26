import numpy as np
import pandas as pd
import pytest

from src.train.evaluator import (
    build_actual_return_baseline_report,
    build_classification_report,
    build_combined_signal_report,
    build_probability_summary,
    build_probability_tail_report,
    build_probability_threshold_report,
    build_validation_selected_threshold_report,
    build_predicted_return_quantile_report,
    build_regression_report,
    build_return_correlation_report,
    build_top_n_basket_backtest_report,
    build_top_n_ranked_selection_report,
    build_trading_relevance_report,
)


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
        "universe",
        "benchmark",
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


def test_predicted_return_reports_reject_empty_and_mismatched_inputs():
    with pytest.raises(ValueError, match="actual_returns"):
        build_predicted_return_quantile_report(np.array([]), np.array([]))

    for report_builder in [
        build_predicted_return_quantile_report,
        build_return_correlation_report,
    ]:
        with pytest.raises(ValueError, match="same length"):
            report_builder(np.array([0.01, -0.02]), np.array([0.01]))
