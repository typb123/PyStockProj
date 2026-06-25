import numpy as np
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
    build_trading_relevance_report,
)


def test_build_classification_report_includes_baselines_and_counts():
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


def test_build_regression_report_includes_zero_and_train_mean_baselines():
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


def test_build_trading_relevance_report_splits_by_classifier_direction():
    report = build_trading_relevance_report(
        actual_returns=np.array([0.10, -0.05, 0.03, -0.02]),
        predicted_returns=np.array([0.08, 0.01, 0.04, -0.03]),
        predicted_direction=np.array([1, 1, 0, 0]),
    )

    assert report["avg_actual_return_when_predicted_up"] == 0.025
    assert report["avg_actual_return_when_predicted_down"] == 0.004999999999999999
    assert report["avg_predicted_return_when_predicted_up"] == 0.045
    assert report["avg_predicted_return_when_predicted_down"] == 0.005000000000000001


def test_build_actual_return_baseline_report_summarizes_full_test_returns():
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


def test_build_probability_summary_reports_distribution():
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


def test_build_probability_tail_report_summarizes_top_and_bottom_tails():
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


def test_build_probability_threshold_report_summarizes_selected_rows():
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


def test_validation_selected_threshold_report_selects_best_validation_threshold():
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


def test_build_predicted_return_quantile_report_summarizes_ranking_buckets():
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


def test_build_return_correlation_report_summarizes_linear_and_rank_signal():
    actual_returns = np.array([10.0, 20.0, 30.0, 40.0])
    predicted_returns = np.array([1.0, 2.0, 4.0, 3.0])

    report = build_return_correlation_report(actual_returns, predicted_returns)

    assert np.isclose(
        report["pearson"], np.corrcoef(actual_returns, predicted_returns)[0, 1]
    )
    assert np.isclose(report["spearman"], 0.8)


def test_build_combined_signal_report_selects_probability_and_return_rank_overlap():
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


def test_predicted_return_reports_reject_empty_and_mismatched_inputs():
    with pytest.raises(ValueError, match="actual_returns"):
        build_predicted_return_quantile_report(np.array([]), np.array([]))

    for report_builder in [
        build_predicted_return_quantile_report,
        build_return_correlation_report,
    ]:
        with pytest.raises(ValueError, match="same length"):
            report_builder(np.array([0.01, -0.02]), np.array([0.01]))
