"""Focused tests for training report assembly, formatting, and logging."""

import logging

import numpy as np
import pandas as pd

import src.train.reporting as reporting

def test_log_feature_importances_summarizes_info_and_keeps_full_debug(caplog):
    feature_importances = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "importance": [0.33333, 0.22222, 0.11111],
        }
    )

    with caplog.at_level(logging.INFO):
        reporting.log_feature_importances("Test Model", feature_importances, top_n=2)

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert info_messages == [
        "Top 2 Feature Importances for Test Model: {'f1': 0.3333, 'f2': 0.2222}"
    ]
    assert all(record.levelno != logging.DEBUG for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        reporting.log_feature_importances("Test Model", feature_importances, top_n=2)

    assert "Feature Importances for Test Model:" in caplog.text
    assert "f1: 0.3333" in caplog.text
    assert "f2: 0.2222" in caplog.text
    assert "f3: 0.1111" in caplog.text


def test_format_top_n_ranked_selection_summary_formats_percentages():
    report = {
        "top_5": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.0123,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "momentum_score_column": "momentum_10d",
                "average_selected_excess_return_vs_benchmark": 0.008,
            },
            "relative_momentum_baseline": {
                "available": True,
                "relative_momentum_score_column": "relative_momentum_10d",
                "average_selected_excess_return_vs_benchmark": 0.009,
            },
            "universe": {
                "average_excess_return_vs_benchmark": 0.0015,
            },
        }
    }

    summary = reporting.format_top_n_ranked_selection_summary(report)

    assert summary == (
        "XGBoost Top-N Ranked Selection Summary:\n"
        "top_5:\n"
        "  model excess=1.23%, random excess=0.40%, "
        "momentum_10d excess=0.80%, relative_momentum_10d excess=0.90%, "
        "universe excess=0.15%\n"
        "  model minus random=0.83%, model minus momentum=0.43%, "
        "model minus relative momentum=0.33%, model beat rate=52.88%"
    )


def test_format_top_n_ranked_selection_summary_handles_unavailable_momentum():
    report = {
        "top_10": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.002,
            },
            "momentum_baseline": {
                "available": False,
                "average_selected_excess_return_vs_benchmark": np.nan,
            },
            "universe": {
                "average_excess_return_vs_benchmark": -0.001,
            },
        }
    }

    summary = reporting.format_top_n_ranked_selection_summary(report)

    assert "momentum excess=unavailable" in summary
    assert "model minus momentum=unavailable" in summary
    assert "relative_momentum excess=n/a" in summary
    assert "model minus relative momentum=n/a" in summary
    assert "universe excess=-0.10%" in summary


def test_format_top_n_ranked_selection_summary_accepts_custom_title():
    report = {
        "top_5": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.002,
            },
            "momentum_baseline": {
                "available": True,
                "average_selected_excess_return_vs_benchmark": 0.004,
            },
            "universe": {
                "average_excess_return_vs_benchmark": 0.001,
            },
        }
    }

    summary = reporting.format_top_n_ranked_selection_summary(
        report,
        title="Custom Ranked Title:",
    )

    assert summary.startswith("Custom Ranked Title:\ntop_5:")
    assert "XGBoost Top-N Ranked Selection Summary:" not in summary


def test_format_top_n_basket_backtest_summary_formats_normal_report():
    report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "momentum_score_column": "momentum_10d",
                "average_basket_excess_return": 0.008,
            },
            "relative_momentum_baseline": {
                "available": True,
                "relative_momentum_score_column": "relative_momentum_10d",
                "average_basket_excess_return": 0.009,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
            },
        }
    }

    summary = reporting.format_top_n_basket_backtest_summary(report)

    assert summary == (
        "XGBoost Top-N Basket Backtest Summary:\n"
        "top_5:\n"
        "  model raw=1.23%, model excess=1.00%, random excess=0.40%, "
        "momentum_10d excess=0.80%, relative_momentum_10d excess=0.90%, "
        "universe excess=0.15%, benchmark raw=0.23%\n"
        "  model minus random=0.60%, model minus momentum=0.20%, "
        "model minus relative momentum=0.10%, beat benchmark rate=52.88%"
    )


def test_format_top_n_basket_backtest_summary_formats_bootstrap_ci_line():
    report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.008,
            },
            "relative_momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.009,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
            },
            "bootstrap_confidence_intervals": {
                "model_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": 0.0052,
                    "ci_upper": 0.0178,
                    "confidence_level": 0.95,
                },
                "model_minus_momentum_baseline_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": -0.0008,
                    "ci_upper": 0.0083,
                    "confidence_level": 0.95,
                },
                "model_minus_relative_momentum_baseline_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": -0.001,
                    "ci_upper": 0.006,
                    "confidence_level": 0.95,
                },
                "model_minus_universe_average_basket_excess_return": {
                    "available": True,
                    "ci_lower": 0.001,
                    "ci_upper": 0.012,
                    "confidence_level": 0.95,
                },
            },
        }
    }

    summary = reporting.format_top_n_basket_backtest_summary(report)

    assert "model excess 95% CI=[0.52%, 1.78%]" in summary
    assert "model minus momentum 95% CI=[-0.08%, 0.83%]" in summary
    assert "model minus relative momentum 95% CI=[-0.10%, 0.60%]" in summary
    assert "model minus universe 95% CI=[0.10%, 1.20%]" in summary


def test_format_top_n_basket_backtest_summary_handles_unavailable_momentum():
    report = {
        "top_10": {
            "model": {
                "average_basket_raw_return": 0.012,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.002,
            },
            "momentum_baseline": {
                "available": False,
                "average_basket_excess_return": np.nan,
            },
            "universe": {
                "average_basket_excess_return": -0.001,
            },
            "benchmark": {
                "average_basket_raw_return": 0.003,
            },
        }
    }

    summary = reporting.format_top_n_basket_backtest_summary(report)

    assert "momentum excess=unavailable" in summary
    assert "model minus momentum=unavailable" in summary
    assert "relative_momentum excess=n/a" in summary
    assert "model minus relative momentum=n/a" in summary
    assert "universe excess=-0.10%" in summary
    assert "benchmark raw=0.30%" in summary


def test_format_top_n_basket_backtest_summary_accepts_custom_title():
    report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.012,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.002,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.004,
            },
            "universe": {
                "average_basket_excess_return": 0.001,
            },
            "benchmark": {
                "average_basket_raw_return": 0.003,
            },
        }
    }

    summary = reporting.format_top_n_basket_backtest_summary(
        report,
        title="Custom Basket Title:",
    )

    assert summary.startswith("Custom Basket Title:\ntop_5:")
    assert "XGBoost Top-N Basket Backtest Summary:" not in summary


def test_format_top_n_basket_backtest_by_year_summary_formats_year_rows():
    report = {
        "top_5": {
            "2022": {
                "model": {
                    "average_basket_excess_return": 0.015,
                    "beat_benchmark_rate": 0.75,
                    "evaluated_dates": 4,
                },
                "random_baseline": {
                    "average_basket_excess_return": 0.004,
                },
                "momentum_baseline": {
                    "available": True,
                    "momentum_score_column": "momentum_10d",
                    "average_basket_excess_return": 0.010,
                },
                "relative_momentum_baseline": {
                    "available": True,
                    "relative_momentum_score_column": "relative_momentum_10d",
                    "average_basket_excess_return": 0.011,
                },
                "universe": {
                    "average_basket_excess_return": 0.002,
                },
            },
            "2023": {
                "model": {
                    "average_basket_excess_return": -0.001,
                    "beat_benchmark_rate": 0.40,
                    "evaluated_dates": 5,
                },
                "random_baseline": {
                    "average_basket_excess_return": 0.003,
                },
                "momentum_baseline": {
                    "available": False,
                    "average_basket_excess_return": np.nan,
                },
                "relative_momentum_baseline": {
                    "available": False,
                    "average_basket_excess_return": np.nan,
                },
                "universe": {
                    "average_basket_excess_return": 0.001,
                },
            },
        }
    }

    summary = reporting.format_top_n_basket_backtest_by_year_summary(report)

    assert summary == (
        "XGBoost Top-N Basket Backtest By-Year Summary:\n"
        "2022 top_5: model excess=1.50%, random excess=0.40%, "
        "momentum_10d excess=1.00%, relative_momentum_10d excess=1.10%, "
        "universe excess=0.20%, model minus random=1.10%, "
        "model minus momentum=0.50%, model minus relative momentum=0.40%, beat benchmark rate=75.00%, "
        "evaluated dates=4\n"
        "2023 top_5: model excess=-0.10%, random excess=0.30%, "
        "momentum excess=unavailable, relative_momentum excess=unavailable, "
        "universe excess=0.10%, model minus random=-0.40%, "
        "model minus momentum=unavailable, model minus relative momentum=unavailable, beat benchmark rate=40.00%, "
        "evaluated dates=5"
    )


def test_format_horizon_comparison_summary_formats_basket_metrics():
    horizon_reports = {
        5: {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.01},
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": True,
                        "momentum_score_column": "momentum_5d",
                        "average_basket_excess_return": 0.003,
                    },
                    "relative_momentum_baseline": {
                        "available": True,
                        "relative_momentum_score_column": "relative_momentum_5d",
                        "average_basket_excess_return": 0.004,
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.004},
                }
            }
        },
        10: {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": -0.01},
                    "random_baseline": {"average_basket_excess_return": 0.001},
                    "momentum_baseline": {
                        "available": False,
                        "momentum_score_column": "momentum_10d",
                        "average_basket_excess_return": np.nan,
                    },
                    "relative_momentum_baseline": {
                        "available": False,
                        "relative_momentum_score_column": "relative_momentum_10d",
                        "average_basket_excess_return": np.nan,
                    },
                    "universe": {"average_basket_excess_return": -0.002},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            }
        },
    }

    summary = reporting.format_horizon_comparison_summary(horizon_reports)

    assert summary == (
        "Horizon Comparison Summary:\n"
        "5d:\n"
        "  top_5 model excess=1.00%, random=0.20%, momentum_5d=0.30%, "
        "relative_momentum_5d=0.40%, "
        "universe=0.10%, benchmark raw=0.40%\n"
        "10d:\n"
        "  top_5 model excess=-1.00%, random=0.10%, momentum_10d=unavailable, "
        "relative_momentum_10d=unavailable, "
        "universe=-0.20%, benchmark raw=0.30%"
    )



def test_format_xgboost_regressor_validation_selection_report_is_readable():
    report = {
        "candidates": [
            {
                "candidate_name": "candidate_0_baseline",
                "params": {"max_depth": 5},
                "validation_top_n_mean_excess_return": 0.01,
                "validation_top_n_basket_excess_returns": {
                    "top_5": 0.02,
                    "top_10": 0.01,
                    "top_20": 0.00,
                },
                "selected": False,
            },
            {
                "candidate_name": "candidate_1_better",
                "params": {"max_depth": 4},
                "validation_top_n_mean_excess_return": 0.03,
                "validation_top_n_basket_excess_returns": {
                    "top_5": 0.04,
                    "top_10": 0.03,
                    "top_20": 0.02,
                },
                "selected": True,
            },
        ],
        "selected_candidate_name": "candidate_1_better",
        "selected_validation_top_n_mean_excess_return": 0.03,
    }

    summary = reporting.format_xgboost_regressor_validation_selection_report(report)

    assert summary.startswith("XGBoost Regressor Validation Top-N Selection Report:")
    assert "candidate_0_baseline: score=1.00%" in summary
    assert "candidate_1_better selected: score=3.00%" in summary
    assert "top_5=4.00%, top_10=3.00%, top_20=2.00%" in summary
    assert "selected=candidate_1_better score=3.00%" in summary



def test_log_xgboost_test_report_uses_explicit_direction_labels(monkeypatch):
    captured = {}

    def fake_build_classification_report(y_true, y_pred):
        captured["y_true"] = np.asarray(y_true)
        captured["y_pred"] = np.asarray(y_pred)
        return {"accuracy": 1.0}

    def fake_build_top_n_selection_reports(
        split_metadata,
        predicted_returns,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(predicted_returns)
        captured["prediction_days"] = prediction_days
        captured["random_trials"] = random_trials
        captured["random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 1}},
            "basket_backtest": {"top_5": {"model": {}}},
            "basket_backtest_by_year": {"top_5": {"2024": {"model": {}}}},
        }

    def fake_build_probability_ranked_top_n_selection_reports(
        split_metadata,
        classifier_probabilities,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["classifier_split_metadata"] = split_metadata
        captured["classifier_probabilities"] = np.asarray(classifier_probabilities)
        captured["classifier_prediction_days"] = prediction_days
        captured["classifier_random_trials"] = random_trials
        captured["classifier_random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 1}},
            "basket_backtest": {"top_5": {"model": {}}},
            "basket_backtest_by_year": {"top_5": {"2024": {"classifier_model": {}}}},
        }

    monkeypatch.setattr(
        reporting,
        "build_classification_report",
        fake_build_classification_report,
    )
    monkeypatch.setattr(
        reporting,
        "build_top_n_selection_reports",
        fake_build_top_n_selection_reports,
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_ranked_top_n_selection_reports",
        fake_build_probability_ranked_top_n_selection_reports,
    )

    y_test = np.array([0.10, -0.20, 0.30])
    direction_y_test = np.array([0, 1, 0])
    classifier_predictions = np.array([0, 1, 1])
    test_split_metadata = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "prediction_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02"]
            ),
            "raw_forward_return": [0.11, -0.18, 0.28],
            "benchmark_forward_return": [0.01, 0.02, -0.02],
            "excess_forward_return": y_test,
            "excess_return_rank_pct_by_date": [0.5, 0.0, 1.0],
            "beat_benchmark_target": direction_y_test,
            "momentum_10d": [0.05, -0.10, 0.20],
            "relative_momentum_10d": [0.04, -0.12, 0.18],
        }
    )
    regressor_predictions = np.array([0.08, -0.15, 0.20])

    report = reporting.log_xgboost_test_report(
        y_train=np.array([0.01, -0.02, 0.03]),
        y_val=np.array([0.02, -0.01, 0.04]),
        y_test=y_test,
        direction_y_test=direction_y_test,
        test_split_metadata=test_split_metadata,
        classifier_predictions=classifier_predictions,
        classifier_validation_probability_up=np.array([0.55, 0.45, 0.65]),
        classifier_probability_up=np.array([0.60, 0.40, 0.70]),
        regressor_predictions=regressor_predictions,
        random_trials=20,
        random_trial_workers=2,
        regressor_validation_selection_report={
            "candidates": [],
            "selected_candidate_name": "candidate_0_baseline",
            "selected_validation_top_n_mean_excess_return": 0.01,
        },
    )

    np.testing.assert_array_equal(captured["y_true"], direction_y_test)
    np.testing.assert_array_equal(captured["y_pred"], classifier_predictions)
    pd.testing.assert_frame_equal(captured["split_metadata"], test_split_metadata)
    pd.testing.assert_frame_equal(
        captured["classifier_split_metadata"], test_split_metadata
    )
    np.testing.assert_array_equal(captured["ranked_predictions"], regressor_predictions)
    np.testing.assert_array_equal(
        captured["classifier_probabilities"],
        np.array([0.60, 0.40, 0.70]),
    )
    assert captured["prediction_days"] == 10
    assert captured["classifier_prediction_days"] == 10
    assert captured["random_trials"] == 20
    assert captured["classifier_random_trials"] == 20
    assert captured["random_trial_workers"] == 2
    assert captured["classifier_random_trial_workers"] == 2
    assert set(report) == {
        "regressor_validation_selection",
        "ranked_selection",
        "basket_backtest",
        "basket_backtest_by_year",
        "same_date_ranking_diagnostics",
        "classifier_probability_ranked_selection",
        "classifier_probability_basket_backtest",
        "classifier_probability_basket_backtest_by_year",
    }
    assert report["regressor_validation_selection"]["selected_candidate_name"] == (
        "candidate_0_baseline"
    )
    assert report["basket_backtest_by_year"] == {"top_5": {"2024": {"model": {}}}}
    assert report["classifier_probability_basket_backtest_by_year"] == {
        "top_5": {"2024": {"classifier_model": {}}}
    }
    assert report["same_date_ranking_diagnostics"]["available"] is True
    assert report["same_date_ranking_diagnostics"]["model"]["score_column"] == (
        "predicted_excess_return"
    )
    assert report["same_date_ranking_diagnostics"]["momentum"]["available"] is True
    assert report["same_date_ranking_diagnostics"]["momentum"]["score_column"] == (
        "momentum_10d"
    )
    assert (
        report["same_date_ranking_diagnostics"]["relative_momentum"]["available"]
        is True
    )
    assert report["same_date_ranking_diagnostics"]["relative_momentum"][
        "score_column"
    ] == "relative_momentum_10d"
    assert not np.array_equal(captured["y_true"], (y_test > 0).astype(int))


def test_log_xgboost_test_report_logs_top_n_summary_at_info_and_full_report_at_debug(
    monkeypatch,
    caplog,
):
    top_n_report = {
        "top_5": {
            "model": {
                "average_selected_excess_return_vs_benchmark": 0.0123,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_selected_excess_return_vs_benchmark": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_selected_excess_return_vs_benchmark": 0.008,
            },
            "universe": {
                "average_excess_return_vs_benchmark": 0.0015,
            },
        }
    }
    basket_report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.008,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
            },
        }
    }
    basket_by_year_report = {
        "top_5": {
            "2023": {
                "model": {
                    "average_basket_excess_return": 0.01,
                    "beat_benchmark_rate": 0.5288,
                    "evaluated_dates": 3,
                },
                "random_baseline": {
                    "average_basket_excess_return": 0.004,
                },
                "momentum_baseline": {
                    "available": True,
                    "average_basket_excess_return": 0.008,
                },
                "relative_momentum_baseline": {
                    "available": True,
                    "average_basket_excess_return": 0.009,
                },
                "universe": {
                    "average_basket_excess_return": 0.0015,
                },
            }
        }
    }

    monkeypatch.setattr(
        reporting,
        "build_classification_report",
        lambda *args, **kwargs: {"accuracy": 1.0},
    )
    monkeypatch.setattr(
        reporting,
        "build_regression_report",
        lambda *args, **kwargs: {"mse": 0.1},
    )
    monkeypatch.setattr(
        reporting,
        "build_actual_return_baseline_report",
        lambda *args, **kwargs: {"count": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_trading_relevance_report",
        lambda *args, **kwargs: {"relevance": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_summary",
        lambda *args, **kwargs: {"count": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_tail_report",
        lambda *args, **kwargs: {"top": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_threshold_report",
        lambda *args, **kwargs: {0.5: {"selected_count": 1}},
    )
    monkeypatch.setattr(
        reporting,
        "build_validation_selected_threshold_report",
        lambda *args, **kwargs: {"selected_threshold": 0.5},
    )
    monkeypatch.setattr(
        reporting,
        "build_predicted_return_quantile_report",
        lambda *args, **kwargs: {"top_10_pct": {}},
    )
    monkeypatch.setattr(
        reporting,
        "build_return_correlation_report",
        lambda *args, **kwargs: {"pearson": 0.0},
    )
    monkeypatch.setattr(
        reporting,
        "build_combined_signal_report",
        lambda *args, **kwargs: {"selected_count": 0},
    )
    monkeypatch.setattr(
        reporting,
        "build_top_n_selection_reports",
        lambda *args, **kwargs: {
            "ranked_selection": top_n_report,
            "basket_backtest": basket_report,
            "basket_backtest_by_year": basket_by_year_report,
        },
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_ranked_top_n_selection_reports",
        lambda *args, **kwargs: {
            "ranked_selection": top_n_report,
            "basket_backtest": basket_report,
            "basket_backtest_by_year": basket_by_year_report,
        },
    )

    with caplog.at_level(logging.INFO):
        reporting.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=pd.DataFrame(),
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=np.array([0.03]),
        )

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert any(
        message.startswith("XGBoost Top-N Ranked Selection Summary:")
        for message in info_messages
    )
    assert any(
        message.startswith("XGBoost Top-N Basket Backtest Summary:")
        for message in info_messages
    )
    assert any(
        message.startswith("XGBoost Top-N Basket Backtest By-Year Summary:")
        for message in info_messages
    )
    assert not any(
        message.startswith(
            "XGBoost Classifier-Probability Top-N Basket Backtest By-Year Summary:"
        )
        for message in info_messages
    )
    assert any(
        message.startswith(
            "XGBoost Classifier-Probability Top-N Ranked Selection Summary:"
        )
        for message in info_messages
    )
    assert any(
        message.startswith(
            "XGBoost Classifier-Probability Top-N Basket Backtest Summary:"
        )
        for message in info_messages
    )
    classifier_ranked_summary = next(
        message
        for message in info_messages
        if message.startswith(
            "XGBoost Classifier-Probability Top-N Ranked Selection Summary:"
        )
    )
    classifier_basket_summary = next(
        message
        for message in info_messages
        if message.startswith(
            "XGBoost Classifier-Probability Top-N Basket Backtest Summary:"
        )
    )
    assert "XGBoost Top-N Ranked Selection Summary:" not in classifier_ranked_summary
    assert "XGBoost Top-N Basket Backtest Summary:" not in classifier_basket_summary
    assert not any(
        message.startswith("XGBoost Top-N Ranked Selection Report:")
        for message in info_messages
    )
    assert not any(
        message.startswith("XGBoost Top-N Basket Backtest Report:")
        for message in info_messages
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        reporting.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=pd.DataFrame(),
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=np.array([0.03]),
        )

    assert "XGBoost Top-N Ranked Selection Report:" in caplog.text
    assert "XGBoost Top-N Basket Backtest Report:" in caplog.text
    assert "XGBoost Top-N Basket Backtest By-Year Report:" in caplog.text
    assert (
        "XGBoost Classifier-Probability Top-N Ranked Selection Report:" in caplog.text
    )
    assert "XGBoost Classifier-Probability Top-N Basket Backtest Report:" in caplog.text
    assert (
        "XGBoost Classifier-Probability Top-N Basket Backtest By-Year Report:"
        in caplog.text
    )


def test_log_xgboost_test_report_logs_basket_summary_at_info_and_full_report_at_debug(
    monkeypatch,
    caplog,
):
    captured = {}
    basket_report = {
        "top_5": {
            "model": {
                "average_basket_raw_return": 0.0123,
                "average_basket_excess_return": 0.01,
                "beat_benchmark_rate": 0.5288,
            },
            "random_baseline": {
                "average_basket_excess_return": 0.004,
            },
            "momentum_baseline": {
                "available": True,
                "average_basket_excess_return": 0.008,
            },
            "universe": {
                "average_basket_excess_return": 0.0015,
            },
            "benchmark": {
                "average_basket_raw_return": 0.0023,
            },
        }
    }

    monkeypatch.setattr(
        reporting,
        "build_classification_report",
        lambda *args, **kwargs: {"accuracy": 1.0},
    )
    monkeypatch.setattr(
        reporting,
        "build_regression_report",
        lambda *args, **kwargs: {"mse": 0.1},
    )
    monkeypatch.setattr(
        reporting,
        "build_actual_return_baseline_report",
        lambda *args, **kwargs: {"count": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_trading_relevance_report",
        lambda *args, **kwargs: {"relevance": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_summary",
        lambda *args, **kwargs: {"count": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_tail_report",
        lambda *args, **kwargs: {"top": 1},
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_threshold_report",
        lambda *args, **kwargs: {0.5: {"selected_count": 1}},
    )
    monkeypatch.setattr(
        reporting,
        "build_validation_selected_threshold_report",
        lambda *args, **kwargs: {"selected_threshold": 0.5},
    )
    monkeypatch.setattr(
        reporting,
        "build_predicted_return_quantile_report",
        lambda *args, **kwargs: {"top_10_pct": {}},
    )
    monkeypatch.setattr(
        reporting,
        "build_return_correlation_report",
        lambda *args, **kwargs: {"pearson": 0.0},
    )
    monkeypatch.setattr(
        reporting,
        "build_combined_signal_report",
        lambda *args, **kwargs: {"selected_count": 0},
    )

    def fake_build_top_n_selection_reports(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["split_metadata"] = split_metadata
        captured["ranked_predictions"] = np.asarray(ranked_predictions)
        captured["prediction_days"] = prediction_days
        captured["random_trials"] = random_trials
        captured["random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"model": {}}},
            "basket_backtest": basket_report,
        }

    monkeypatch.setattr(
        reporting,
        "build_top_n_selection_reports",
        fake_build_top_n_selection_reports,
    )
    monkeypatch.setattr(
        reporting,
        "build_probability_ranked_top_n_selection_reports",
        lambda *args, **kwargs: {
            "ranked_selection": {"top_5": {"model": {}}},
            "basket_backtest": basket_report,
        },
    )

    test_split_metadata = pd.DataFrame({"Ticker": ["AAA"]})
    regressor_predictions = np.array([0.03])

    with caplog.at_level(logging.INFO):
        reporting.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=test_split_metadata,
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=regressor_predictions,
            random_trials=20,
            random_trial_workers=2,
        )

    pd.testing.assert_frame_equal(captured["split_metadata"], test_split_metadata)
    np.testing.assert_array_equal(captured["ranked_predictions"], regressor_predictions)
    assert captured["prediction_days"] == 10
    assert captured["random_trials"] == 20
    assert captured["random_trial_workers"] == 2
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert any(
        message.startswith("XGBoost Top-N Basket Backtest Summary:")
        for message in info_messages
    )
    assert not any(
        message.startswith("XGBoost Top-N Basket Backtest Report:")
        for message in info_messages
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        reporting.log_xgboost_test_report(
            y_train=np.array([0.01]),
            y_val=np.array([0.02]),
            y_test=np.array([0.03]),
            direction_y_test=np.array([1]),
            test_split_metadata=test_split_metadata,
            classifier_predictions=np.array([1]),
            classifier_validation_probability_up=np.array([0.55]),
            classifier_probability_up=np.array([0.60]),
            regressor_predictions=regressor_predictions,
        )

    assert "XGBoost Top-N Basket Backtest Report:" in caplog.text
