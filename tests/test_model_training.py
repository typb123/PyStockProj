"""Tests for model-training pipelines, fitting, and selection helpers."""

import numpy as np
import pandas as pd
import pytest

from src.config import XG_PARAMS_REGRESSOR
from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)
import src.train.model_training as model_training
import src.train.reporting as reporting


def test_xgboost_regressor_candidates_include_default_baseline():
    candidates = model_training.build_xgboost_regressor_candidate_configs()

    assert candidates[0]["candidate_id"] == 0
    assert candidates[0]["candidate_name"] == "candidate_0_baseline"
    assert candidates[0]["params"] == XG_PARAMS_REGRESSOR
    assert len(candidates) >= 2


def test_validation_top_n_selection_chooses_best_candidate_and_preserves_splits(
    monkeypatch,
):
    captured_fit_calls = []

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.array([1.0])

        def fit(self, x, y, eval_set=None, verbose=False):
            captured_fit_calls.append(
                {
                    "x": x.copy(),
                    "y": np.asarray(y).copy(),
                    "eval_x": eval_set[0][0].copy(),
                    "eval_y": np.asarray(eval_set[0][1]).copy(),
                    "verbose": verbose,
                }
            )
            return self

        def predict(self, x):
            return np.full(len(x), self.params["validation_score"])

    def fake_model_only_report(split_metadata, ranked_predictions, top_n_values):
        score = float(np.asarray(ranked_predictions)[0])
        return {
            f"top_{top_n}": {
                "model": {"average_basket_excess_return": score + top_n / 10000}
            }
            for top_n in top_n_values
        }

    monkeypatch.setattr(model_training, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        model_training,
        "build_model_only_top_n_basket_backtest_report",
        fake_model_only_report,
    )

    x_train = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    x_val = pd.DataFrame({"feature_a": [5.0, 6.0], "feature_b": [7.0, 8.0]})
    y_train = np.array([0.01, -0.02])
    y_val = np.array([0.03, 0.04])
    validation_metadata = pd.DataFrame({"Ticker": ["AAA", "BBB"]})
    candidates = [
        {
            "candidate_id": 0,
            "candidate_name": "candidate_0_baseline",
            "params": {"validation_score": 0.01},
        },
        {
            "candidate_id": 1,
            "candidate_name": "candidate_1_better",
            "params": {"validation_score": 0.03},
        },
    ]

    selected_model, report = (
        model_training.select_xgboost_regressor_by_validation_top_n(
            x_train,
            y_train,
            x_val,
            y_val,
            validation_metadata,
            candidate_configs=candidates,
        )
    )

    assert selected_model.params == {"validation_score": 0.03}
    assert report["selection_metric"] == "validation_top_n_mean_excess_return"
    assert report["selected_candidate_id"] == 1
    assert report["candidates"][0]["selected"] is False
    assert report["candidates"][1]["selected"] is True
    assert report["candidates"][1]["available_bucket_count"] == 3
    assert "unavailable or NaN buckets are ignored" in report["selection_bucket_policy"]
    assert len(captured_fit_calls) == 2
    for fit_call in captured_fit_calls:
        pd.testing.assert_frame_equal(fit_call["x"], x_train)
        pd.testing.assert_frame_equal(fit_call["eval_x"], x_val)
        np.testing.assert_array_equal(fit_call["y"], y_train)
        np.testing.assert_array_equal(fit_call["eval_y"], y_val)
        assert fit_call["verbose"] is False


def test_rank_ndcg_training_data_filters_labels_groups_and_sorts_by_date():
    features = pd.DataFrame({"feature": [0, 1, 2, 3, 4, 5]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                ]
            ),
            "excess_return_rank_pct_by_date": [0.90, 0.10, 0.30, 0.70, 0.80, np.nan],
        }
    )

    ranked_features, labels, qid, grouped_metadata = (
        model_training._rank_ndcg_training_data(
            features,
            metadata,
        )
    )

    np.testing.assert_array_equal(ranked_features["feature"].to_numpy(), [1, 3, 0, 2])
    np.testing.assert_array_equal(labels, [0, 3, 4, 1])
    np.testing.assert_array_equal(qid, [0, 0, 1, 1])
    assert grouped_metadata["prediction_date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]


def test_rank_percentile_to_ndcg_relevance_uses_expected_boundaries():
    labels = model_training._rank_percentile_to_ndcg_relevance(
        pd.Series([0.20, 0.21, 0.40, 0.41, 0.60, 0.61, 0.79, 0.80, 1.00])
    )

    np.testing.assert_array_equal(labels.to_numpy(), [0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_rank_ndcg_training_data_rejects_no_usable_rows():
    features = pd.DataFrame({"feature": [0, 1]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "excess_return_rank_pct_by_date": [np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="No usable ranker training rows"):
        model_training._rank_ndcg_training_data(features, metadata)


def test_rank_ndcg_training_data_requires_two_usable_groups():
    features = pd.DataFrame({"feature": [0, 1, 2]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02"]
            ),
            "excess_return_rank_pct_by_date": [0.10, 0.90, 0.50],
        }
    )

    with pytest.raises(ValueError, match="at least 2 usable prediction_date groups"):
        model_training._rank_ndcg_training_data(features, metadata)


def test_train_models_uses_selected_regressor_predictions_for_final_report(
    monkeypatch,
):
    rows = 4
    feature_count = len(MODEL_FEATURE_COLUMNS)
    y_train = np.array([0.01, 0.02, -0.01, 0.03])
    y_val = np.array([0.01, -0.02, 0.02, 0.03])
    captured = {}

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)
            self.scalar = None

        def prepare_for_train(self, data, prediction_days, test_size):
            split_metadata = pd.DataFrame(
                {
                    "Ticker": ["AAA"] * rows,
                    "prediction_date": pd.date_range("2024-01-01", periods=rows),
                    "dailyReturn": [0.01] * rows,
                    "raw_forward_return": [0.02] * rows,
                    "benchmark_forward_return": [0.01] * rows,
                    "excess_forward_return": [0.01] * rows,
                    "beat_benchmark_target": [1] * rows,
                }
            )
            return {
                "x_train": np.ones((rows, feature_count)),
                "x_val": np.ones((rows, feature_count)) * 2,
                "x_test": np.ones((rows, feature_count)) * 3,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": np.array([0.02, -0.01, 0.01, 0.03]),
                "direction_y_train": np.array([1, 1, 0, 1]),
                "direction_y_val": np.array([1, 0, 1, 1]),
                "direction_y_test": np.array([1, 0, 1, 1]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "val": split_metadata.copy(),
                    "test": split_metadata.copy(),
                },
            }

    class FakeLinearRegression:
        def fit(self, x, y):
            self.coef_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.zeros(len(x))

    class FakeClassifier:
        def __init__(self, **params):
            self.feature_importances_ = np.array([])

        def fit(self, x, y, eval_set=None, verbose=False):
            self.feature_importances_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.ones(len(x), dtype=int)

        def predict_proba(self, x):
            return np.column_stack([np.zeros(len(x)), np.ones(len(x))])

        def save_model(self, path):
            pass

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.array([])

        def fit(self, x, y, eval_set=None, verbose=False):
            captured.setdefault("regressor_fit_y", []).append(np.asarray(y).copy())
            self.feature_importances_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.full(len(x), self.params["prediction"])

        def save_model(self, path):
            pass

    candidates = [
        {
            "candidate_id": 0,
            "candidate_name": "candidate_0_baseline",
            "params": {"prediction": 0.01},
        },
        {
            "candidate_id": 1,
            "candidate_name": "candidate_1_selected",
            "params": {"prediction": 0.04},
        },
    ]

    def fake_model_only_report(split_metadata, ranked_predictions, top_n_values):
        score = float(np.asarray(ranked_predictions)[0])
        return {
            f"top_{top_n}": {"model": {"average_basket_excess_return": score}}
            for top_n in top_n_values
        }

    def fake_log_xgboost_test_report(*args, **kwargs):
        captured["final_regressor_predictions"] = np.asarray(args[8])
        captured["selection_report"] = kwargs["regressor_validation_selection_report"]
        return {
            "regressor_validation_selection": captured["selection_report"],
            "basket_backtest": {},
            "same_date_ranking_diagnostics": {"available": True},
        }

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(model_training, "LinearRegression", FakeLinearRegression)
    monkeypatch.setattr(model_training, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(model_training, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        model_training,
        "build_xgboost_regressor_candidate_configs",
        lambda base_params: candidates,
    )
    monkeypatch.setattr(
        model_training,
        "build_model_only_top_n_basket_backtest_report",
        fake_model_only_report,
    )
    monkeypatch.setattr(model_training, "evaluate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_training, "log_xgboost_test_report", fake_log_xgboost_test_report
    )
    monkeypatch.setattr(
        model_training, "log_feature_importances", lambda *args, **kwargs: None
    )

    def fake_save_horizon_model_artifacts(*args, **kwargs):
        captured["saved_model_metadata"] = args[5]
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(
        model_training,
        "save_horizon_model_artifacts",
        fake_save_horizon_model_artifacts,
    )

    report = model_training.train_models(pd.DataFrame({"Close": [1.0]}), prediction_days=10)

    np.testing.assert_array_equal(captured["regressor_fit_y"][0], y_train)
    np.testing.assert_array_equal(
        captured["final_regressor_predictions"], [0.04] * rows
    )
    assert captured["selection_report"]["selected_candidate_id"] == 1
    assert report["regressor_validation_selection"]["selected_candidate_id"] == 1
    assert report["target_mode"] == "excess_return"
    assert report["same_date_ranking_diagnostics"] == {"available": True}
    assert (
        captured["saved_model_metadata"]["regressor_validation_selection"][
            "selected_candidate_id"
        ]
        == 1
    )
    assert (
        captured["saved_model_metadata"]["xgboost_regressor_selected_candidate_name"]
        == "candidate_1_selected"
    )
    assert captured["saved_model_metadata"]["xgboost_regressor_selected_params"] == {
        "prediction": 0.04
    }


def test_train_models_metadata_includes_momentum_features_and_excludes_targets(
    monkeypatch,
):
    captured = {}
    target_columns = {
        "raw_forward_return",
        "benchmark_forward_return",
        "excess_forward_return",
        "targetReturns",
        "beat_benchmark_target",
    }

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)
            self.scalar = None

        def prepare_for_train(self, data, prediction_days, test_size):
            rows = 4
            feature_count = len(MODEL_FEATURE_COLUMNS)
            split_metadata = pd.DataFrame(
                {
                    "Ticker": ["AAA"] * rows,
                    "prediction_date": pd.date_range("2024-01-01", periods=rows),
                    "dailyReturn": [0.01] * rows,
                    "raw_forward_return": [0.02] * rows,
                    "benchmark_forward_return": [0.01] * rows,
                    "excess_forward_return": [0.01] * rows,
                    "beat_benchmark_target": [1] * rows,
                }
            )
            return {
                "x_train": np.ones((rows, feature_count)),
                "x_val": np.ones((rows, feature_count)),
                "x_test": np.ones((rows, feature_count)),
                "y_train": np.array([0.01, 0.02, -0.01, 0.03]),
                "y_val": np.array([0.01, -0.02, 0.02, 0.03]),
                "y_test": np.array([0.02, -0.01, 0.01, 0.03]),
                "direction_y_train": np.array([1, 1, 0, 1]),
                "direction_y_val": np.array([1, 0, 1, 1]),
                "direction_y_test": np.array([1, 0, 1, 1]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "val": split_metadata.copy(),
                    "test": split_metadata,
                },
            }

    class FakeLinearRegression:
        def fit(self, x, y):
            self.coef_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.zeros(len(x))

    class FakeXgbModel:
        def __init__(self, **kwargs):
            self.feature_importances_ = np.array([])

        def fit(self, x, y, eval_set=None, verbose=False):
            self.feature_importances_ = np.ones(x.shape[1])
            return self

        def predict(self, x):
            return np.ones(len(x), dtype=int)

        def predict_proba(self, x):
            return np.column_stack([np.zeros(len(x)), np.ones(len(x))])

        def save_model(self, path):
            pass

    def fake_save_horizon_model_artifacts(
        prediction_days,
        linear_model,
        scaler_lr,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
        regressor,
    ):
        captured["all_features"] = all_features
        captured["model_metadata"] = model_metadata
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(model_training, "LinearRegression", FakeLinearRegression)
    monkeypatch.setattr(model_training, "XGBClassifier", FakeXgbModel)
    monkeypatch.setattr(model_training, "XGBRegressor", FakeXgbModel)
    monkeypatch.setattr(model_training, "evaluate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_training,
        "log_xgboost_test_report",
        lambda *args, **kwargs: {
            "basket_backtest": {},
            "same_date_ranking_diagnostics": {"available": True},
        },
    )
    monkeypatch.setattr(
        model_training, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        model_training,
        "save_horizon_model_artifacts",
        fake_save_horizon_model_artifacts,
    )

    model_training.train_models(pd.DataFrame({"Close": [1.0]}), prediction_days=10)

    metadata = captured["model_metadata"]
    assert metadata["target_mode"] == "excess_return"
    for column in (
        ABSOLUTE_MOMENTUM_FEATURE_COLUMNS + SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS
    ):
        assert column in captured["all_features"]
        assert column in metadata["classifier_features"]
        assert column in metadata["regressor_features"]

    assert metadata["classifier_features"] == MODEL_FEATURE_COLUMNS
    assert metadata["regressor_features"] == MODEL_FEATURE_COLUMNS
    assert metadata["target_type"] == "spy_relative_excess_forward_return"
    assert metadata["benchmark_ticker"] == "SPY"
    assert metadata["regressor_target"] == "targetReturns"
    assert metadata["classifier_target"] == "beat_benchmark_target"
    assert target_columns.isdisjoint(captured["all_features"])
    assert target_columns.isdisjoint(metadata["classifier_features"])
    assert target_columns.isdisjoint(metadata["regressor_features"])


def test_train_models_rejects_unknown_target_mode(monkeypatch):
    monkeypatch.setattr(
        model_training,
        "validate_input_data",
        lambda data: pytest.fail("training path should not run"),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported target_mode='bad_mode'",
    ):
        model_training.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="bad_mode",
        )


def test_train_models_cross_sectional_ranking_trains_on_top_bottom_and_scores_all(
    monkeypatch,
):
    feature_count = len(MODEL_FEATURE_COLUMNS)
    captured = {}

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)
            self.scalar = None

        def prepare_for_train(self, data, prediction_days, test_size):
            train_rows = 5
            val_rows = 3
            test_rows = 4
            return {
                "x_train": np.tile(np.arange(train_rows).reshape(-1, 1), feature_count),
                "x_val": np.ones((val_rows, feature_count)) * 2,
                "x_test": np.ones((test_rows, feature_count)) * 3,
                "y_train": np.array([0.01, 0.02, -0.01, 0.03, 0.04]),
                "y_val": np.array([0.01, -0.02, 0.02]),
                "y_test": np.array([0.02, -0.01, 0.01, 0.03]),
                "direction_y_train": np.array([1, 1, 0, 1, 1]),
                "direction_y_val": np.array([1, 0, 1]),
                "direction_y_test": np.array([1, 0, 1, 1]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "train": pd.DataFrame(
                        {
                            "Ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                            "ranking_train_sample": [
                                True,
                                False,
                                True,
                                False,
                                True,
                            ],
                            "top_quintile_target": [1.0, np.nan, 0.0, np.nan, 1.0],
                        }
                    ),
                    "val": pd.DataFrame(
                        {
                            "Ticker": ["AAA", "BBB", "CCC"],
                            "ranking_train_sample": [True, False, True],
                            "top_quintile_target": [1.0, np.nan, 0.0],
                        }
                    ),
                    "test": pd.DataFrame(
                        {
                            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
                            "prediction_date": pd.date_range(
                                "2024-01-01",
                                periods=test_rows,
                            ),
                            "raw_forward_return": [0.03, 0.01, -0.02, 0.04],
                            "benchmark_forward_return": [0.01] * test_rows,
                            "excess_forward_return": [0.02, 0.00, -0.03, 0.03],
                            "excess_return_rank_pct_by_date": [0.75, 0.50, 0.00, 1.00],
                            "momentum_10d": [0.03, 0.02, -0.01, 0.04],
                            "relative_momentum_10d": [0.02, 0.01, -0.02, 0.03],
                            "ranking_train_sample": [True, False, True, False],
                            "top_quintile_target": [1.0, np.nan, 0.0, np.nan],
                        }
                    ),
                },
            }

    class FakeRankingClassifier:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.ones(feature_count)

        def fit(self, x, y, eval_set=None, verbose=False):
            captured["fit_x_first_feature"] = x.iloc[:, 0].to_numpy()
            captured["fit_y"] = np.asarray(y)
            captured["eval_set_rows"] = len(eval_set[0][0])
            captured["verbose"] = verbose
            return self

        def predict_proba(self, x):
            captured.setdefault("predict_proba_rows", []).append(len(x))
            if len(x) == 3:
                scores = np.array([0.10, 0.20, 0.30])
            else:
                scores = np.array([0.70, 0.40, 0.90, 0.20])
            return np.column_stack([1 - scores, scores])

        def save_model(self, path):
            captured["saved_classifier_path"] = path

    def fake_top_n_reports(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["top_n_split_metadata"] = split_metadata.copy()
        captured["top_n_ranked_predictions"] = np.asarray(ranked_predictions)
        captured["top_n_prediction_days"] = prediction_days
        captured["top_n_random_trials"] = random_trials
        captured["top_n_random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 4}},
            "basket_backtest": {"top_5": {"model": {"average_basket_excess_return": 0.02}}},
            "basket_backtest_by_year": {},
        }

    def fake_save_ranking_model_artifacts(
        prediction_days,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
    ):
        captured["saved_prediction_days"] = prediction_days
        captured["saved_all_features"] = all_features
        captured["saved_model_metadata"] = model_metadata
        classifier.save_model("models/horizon_10/xgboost_classifier.json")
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(model_training, "XGBClassifier", FakeRankingClassifier)
    monkeypatch.setattr(reporting, "build_top_n_selection_reports", fake_top_n_reports)
    monkeypatch.setattr(
        model_training,
        "save_ranking_model_artifacts",
        fake_save_ranking_model_artifacts,
    )
    monkeypatch.setattr(
        model_training, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        model_training,
        "select_xgboost_regressor_by_validation_top_n",
        lambda *args, **kwargs: pytest.fail("regression path should not run"),
    )

    report = model_training.train_models(
        pd.DataFrame({"Close": [1.0]}),
        prediction_days=10,
        target_mode="cross_sectional_top_bottom",
        random_trials=7,
        random_trial_workers=2,
    )

    np.testing.assert_array_equal(captured["fit_x_first_feature"], [0, 2, 4])
    np.testing.assert_array_equal(captured["fit_y"], [1, 0, 1])
    assert captured["eval_set_rows"] == 2
    assert captured["verbose"] is False
    assert captured["predict_proba_rows"] == [3, 4]
    np.testing.assert_allclose(
        captured["top_n_ranked_predictions"],
        [0.70, 0.40, 0.90, 0.20],
    )
    assert len(captured["top_n_split_metadata"]) == 4
    assert captured["top_n_prediction_days"] == 10
    assert captured["top_n_random_trials"] == 7
    assert captured["top_n_random_trial_workers"] == 2
    assert captured["saved_prediction_days"] == 10
    assert captured["saved_all_features"] == MODEL_FEATURE_COLUMNS
    assert captured["saved_classifier_path"] == "models/horizon_10/xgboost_classifier.json"

    metadata = captured["saved_model_metadata"]
    assert metadata["target_mode"] == "cross_sectional_top_bottom"
    assert metadata["target_type"] == "cross_sectional_top_bottom_quintile"
    assert metadata["classifier_target"] == "top_quintile_target"
    assert metadata["ranking_target"] == (
        "top_20_vs_bottom_20_by_excess_forward_return"
    )
    assert metadata["ranking_group_key"] == "prediction_date"
    assert metadata["primary_model_score"] == (
        "probability_of_top_quintile_outperformance"
    )
    assert metadata["model_artifact_type"] == "classifier_only"
    assert metadata["regressor_artifact"] is None
    assert metadata["regressor_features"] == []
    assert report["target_mode"] == "cross_sectional_top_bottom"
    assert report["ranking_score_name"] == (
        "probability_of_top_quintile_outperformance"
    )
    assert report["same_date_ranking_diagnostics"]["available"] is True
    assert report["same_date_ranking_diagnostics"]["score_column"] == (
        "probability_of_top_quintile_outperformance"
    )
    assert report["same_date_ranking_diagnostics"]["model"]["score_column"] == (
        "probability_of_top_quintile_outperformance"
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
    assert report["basket_backtest"]["top_5"]["model"]["average_basket_excess_return"] == 0.02


def test_train_models_cross_sectional_ranking_requires_ranking_metadata(monkeypatch):
    feature_count = len(MODEL_FEATURE_COLUMNS)

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)

        def prepare_for_train(self, data, prediction_days, test_size):
            return {
                "x_train": np.ones((2, feature_count)),
                "x_val": np.ones((2, feature_count)),
                "x_test": np.ones((2, feature_count)),
                "y_train": np.array([0.01, 0.02]),
                "y_val": np.array([0.01, -0.02]),
                "y_test": np.array([0.02, -0.01]),
                "direction_y_train": np.array([1, 1]),
                "direction_y_val": np.array([1, 0]),
                "direction_y_test": np.array([1, 0]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "train": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
                    "val": pd.DataFrame(
                        {
                            "ranking_train_sample": [True, True],
                            "top_quintile_target": [1.0, 0.0],
                        }
                    ),
                    "test": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
                },
            }

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        model_training,
        "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier should not train"),
    )

    with pytest.raises(
        ValueError,
        match=r"requires split_metadata\['train'\] columns",
    ):
        model_training.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_top_bottom",
        )


def test_train_models_cross_sectional_ranking_requires_test_split_metadata(
    monkeypatch,
):
    feature_count = len(MODEL_FEATURE_COLUMNS)

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)

        def prepare_for_train(self, data, prediction_days, test_size):
            ranking_metadata = pd.DataFrame(
                {
                    "ranking_train_sample": [True, True],
                    "top_quintile_target": [1.0, 0.0],
                }
            )
            return {
                "x_train": np.ones((2, feature_count)),
                "x_val": np.ones((2, feature_count)),
                "x_test": np.ones((2, feature_count)),
                "y_train": np.array([0.01, 0.02]),
                "y_val": np.array([0.01, -0.02]),
                "y_test": np.array([0.02, -0.01]),
                "direction_y_train": np.array([1, 1]),
                "direction_y_val": np.array([1, 0]),
                "direction_y_test": np.array([1, 0]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "train": ranking_metadata,
                    "val": ranking_metadata.copy(),
                },
            }

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        model_training,
        "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier should not train"),
    )

    with pytest.raises(
        ValueError,
        match=r"target_mode='cross_sectional_top_bottom' requires split_metadata\['test'\]",
    ):
        model_training.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_top_bottom",
        )


def test_train_models_cross_sectional_ranking_requires_both_classes(monkeypatch):
    feature_count = len(MODEL_FEATURE_COLUMNS)

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)

        def prepare_for_train(self, data, prediction_days, test_size):
            ranking_metadata = pd.DataFrame(
                {
                    "ranking_train_sample": [True, True],
                    "top_quintile_target": [1.0, 1.0],
                }
            )
            return {
                "x_train": np.ones((2, feature_count)),
                "x_val": np.ones((2, feature_count)),
                "x_test": np.ones((2, feature_count)),
                "y_train": np.array([0.01, 0.02]),
                "y_val": np.array([0.01, -0.02]),
                "y_test": np.array([0.02, -0.01]),
                "direction_y_train": np.array([1, 1]),
                "direction_y_val": np.array([1, 0]),
                "direction_y_test": np.array([1, 0]),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "train": ranking_metadata,
                    "val": ranking_metadata.copy(),
                    "test": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
                },
            }

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        model_training,
        "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier should not train"),
    )

    with pytest.raises(
        ValueError,
        match="Ranking training sample must contain both classes",
    ):
        model_training.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_top_bottom",
        )


def test_train_models_cross_sectional_rank_ndcg_trains_grouped_ranker_and_scores_all(
    monkeypatch,
):
    captured = {}
    feature_count = len(MODEL_FEATURE_COLUMNS)
    train_rows = 6
    val_rows = 3
    test_rows = 4

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)

        def prepare_for_train(self, data, prediction_days, test_size):
            return {
                "x_train": np.tile(np.arange(train_rows).reshape(-1, 1), feature_count),
                "x_val": np.tile(
                    np.arange(10, 10 + val_rows).reshape(-1, 1),
                    feature_count,
                ),
                "x_test": np.tile(
                    np.arange(20, 20 + test_rows).reshape(-1, 1),
                    feature_count,
                ),
                "y_train": np.zeros(train_rows),
                "y_val": np.zeros(val_rows),
                "y_test": np.zeros(test_rows),
                "direction_y_train": np.zeros(train_rows),
                "direction_y_val": np.zeros(val_rows),
                "direction_y_test": np.zeros(test_rows),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "train": pd.DataFrame(
                        {
                            "Ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                            "prediction_date": pd.to_datetime(
                                [
                                    "2024-01-02",
                                    "2024-01-01",
                                    "2024-01-02",
                                    "2024-01-01",
                                    "2024-01-03",
                                    "2024-01-02",
                                ]
                            ),
                            "excess_return_rank_pct_by_date": [
                                0.90,
                                0.10,
                                0.30,
                                0.70,
                                0.80,
                                np.nan,
                            ],
                        }
                    ),
                    "val": pd.DataFrame(
                        {
                            "Ticker": ["AAA", "BBB", "CCC"],
                            "prediction_date": pd.to_datetime(
                                ["2024-01-04", "2024-01-04", "2024-01-05"]
                            ),
                            "excess_return_rank_pct_by_date": [0.20, 0.90, 0.50],
                        }
                    ),
                    "test": pd.DataFrame(
                        {
                            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
                            "prediction_date": pd.to_datetime(
                                [
                                    "2024-01-06",
                                    "2024-01-06",
                                    "2024-01-07",
                                    "2024-01-07",
                                ]
                            ),
                            "raw_forward_return": [0.03, 0.01, -0.02, 0.04],
                            "benchmark_forward_return": [0.01] * test_rows,
                            "excess_forward_return": [0.02, 0.00, -0.03, 0.03],
                            "excess_return_rank_pct_by_date": [0.75, 0.50, 0.00, 1.00],
                            "momentum_10d": [0.03, 0.02, -0.01, 0.04],
                            "relative_momentum_10d": [0.02, 0.01, -0.02, 0.03],
                        }
                    ),
                },
            }

    class FakeRanker:
        def __init__(self, **params):
            captured["ranker_params"] = params
            self.feature_importances_ = np.ones(feature_count)

        def fit(self, x, y, *, qid=None, eval_set=None, eval_qid=None, verbose=True):
            captured["fit_first_feature"] = x.iloc[:, 0].to_numpy()
            captured["fit_y"] = np.asarray(y)
            captured["fit_qid"] = np.asarray(qid)
            captured["eval_first_feature"] = eval_set[0][0].iloc[:, 0].to_numpy()
            captured["eval_y"] = np.asarray(eval_set[0][1])
            captured["eval_qid"] = np.asarray(eval_qid[0])
            captured["fit_verbose"] = verbose
            return self

        def predict(self, x):
            captured.setdefault("predict_lengths", []).append(len(x))
            if len(x) == val_rows:
                return np.array([0.10, 0.20, 0.30])
            return np.array([0.70, 0.40, 0.90, 0.20])

    def fake_top_n_reports(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["top_n_split_metadata"] = split_metadata.copy()
        captured["top_n_ranked_predictions"] = np.asarray(ranked_predictions)
        captured["top_n_prediction_days"] = prediction_days
        captured["top_n_random_trials"] = random_trials
        captured["top_n_random_trial_workers"] = random_trial_workers
        return {
            "ranked_selection": {"top_5": {"selected_row_count": 4}},
            "basket_backtest": {"top_5": {"model": {"average_basket_excess_return": 0.02}}},
            "basket_backtest_by_year": {},
        }

    def fake_save_rank_ndcg_model_artifacts(
        prediction_days,
        data_preparator,
        all_features,
        model_metadata,
        ranker,
    ):
        captured["saved_prediction_days"] = prediction_days
        captured["saved_all_features"] = all_features
        captured["saved_model_metadata"] = model_metadata
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(model_training, "XGBRanker", FakeRanker)
    monkeypatch.setattr(reporting, "build_top_n_selection_reports", fake_top_n_reports)
    monkeypatch.setattr(
        model_training,
        "save_rank_ndcg_model_artifacts",
        fake_save_rank_ndcg_model_artifacts,
    )
    monkeypatch.setattr(
        model_training, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        model_training,
        "select_xgboost_regressor_by_validation_top_n",
        lambda *args, **kwargs: pytest.fail("regression path should not run"),
    )

    report = model_training.train_models(
        pd.DataFrame({"Close": [1.0]}),
        prediction_days=10,
        target_mode="cross_sectional_rank_ndcg",
        random_trials=7,
        random_trial_workers=2,
        ranker_params={"objective": "rank:ndcg", "random_state": 42},
    )

    np.testing.assert_array_equal(captured["fit_first_feature"], [1, 3, 0, 2])
    np.testing.assert_array_equal(captured["fit_y"], [0, 3, 4, 1])
    np.testing.assert_array_equal(captured["fit_qid"], [0, 0, 1, 1])
    np.testing.assert_array_equal(captured["eval_first_feature"], [10, 11])
    np.testing.assert_array_equal(captured["eval_y"], [0, 4])
    np.testing.assert_array_equal(captured["eval_qid"], [0, 0])
    assert captured["fit_verbose"] is False
    assert captured["predict_lengths"] == [val_rows, test_rows]
    np.testing.assert_allclose(
        captured["top_n_ranked_predictions"],
        [0.70, 0.40, 0.90, 0.20],
    )
    assert len(captured["top_n_split_metadata"]) == test_rows
    assert captured["top_n_prediction_days"] == 10
    assert captured["top_n_random_trials"] == 7
    assert captured["top_n_random_trial_workers"] == 2
    assert captured["saved_prediction_days"] == 10
    assert captured["saved_all_features"] == MODEL_FEATURE_COLUMNS

    metadata = captured["saved_model_metadata"]
    assert metadata["target_mode"] == "cross_sectional_rank_ndcg"
    assert metadata["target_type"] == "grouped_learning_to_rank_ndcg"
    assert metadata["ranking_group_key"] == "prediction_date"
    assert metadata["ranking_label_source"] == "excess_return_rank_pct_by_date"
    assert metadata["ranking_label_type"] == "graded_0_to_4_from_rank_percentile"
    assert metadata["primary_model_score"] == "xgboost_rank_ndcg_score"
    assert metadata["model_artifact_type"] == "ranker"
    assert metadata["classifier_artifact"] is None
    assert metadata["regressor_artifact"] is None
    assert metadata["ranker_training_row_count"] == 4
    assert metadata["ranker_training_candidate_count"] == train_rows
    assert metadata["ranker_training_group_count"] == 2
    assert metadata["ranker_training_dropped_row_count"] == 2
    assert metadata["ranker_validation_scored_row_count"] == val_rows
    assert metadata["ranker_test_scored_row_count"] == test_rows
    assert report["target_mode"] == "cross_sectional_rank_ndcg"
    assert report["ranking_score_name"] == "xgboost_rank_ndcg_score"
    assert report["same_date_ranking_diagnostics"]["model"]["score_column"] == (
        "xgboost_rank_ndcg_score"
    )
    assert report["same_date_ranking_diagnostics"]["momentum"]["score_column"] == (
        "momentum_10d"
    )
    assert report["same_date_ranking_diagnostics"]["relative_momentum"][
        "score_column"
    ] == "relative_momentum_10d"
    assert report["basket_backtest"]["top_5"]["model"]["average_basket_excess_return"] == 0.02


def test_train_models_rejects_unknown_feature_columns_override(monkeypatch):
    monkeypatch.setattr(
        model_training,
        "DataPreparator",
        lambda *args, **kwargs: pytest.fail("data prep should not run"),
    )

    with pytest.raises(ValueError, match="feature_columns_override"):
        model_training.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_rank_ndcg",
            feature_columns_override=["momentum_10d", "targetReturns"],
        )


def test_train_models_feature_columns_override_reaches_rank_ndcg_branch(monkeypatch):
    captured = []
    feature_count = len(MODEL_FEATURE_COLUMNS)
    override_features = ["momentum_10d", "volatility"]

    class FakePreparator:
        def prepare_for_train(self, data, prediction_days, test_size):
            rows = 2
            return {
                "x_train": np.ones((rows, feature_count)),
                "x_val": np.ones((rows, feature_count)) * 2,
                "x_test": np.ones((rows, feature_count)) * 3,
                "y_train": np.zeros(rows),
                "y_val": np.zeros(rows),
                "y_test": np.zeros(rows),
                "direction_y_train": np.zeros(rows),
                "direction_y_val": np.zeros(rows),
                "direction_y_test": np.zeros(rows),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {},
            }

    def fake_train_cross_sectional_rank_ndcg_model(
        data_preparator,
        prepared_data,
        all_features,
        linear_features,
        classifier_features,
        x_train_ranker,
        x_val_ranker,
        x_test_ranker,
        prediction_days,
        random_trials,
        random_trial_workers,
        ranker_params=None,
    ):
        captured.append(
            {
                "all_features": all_features,
                "classifier_features": classifier_features,
                "x_train_columns": list(x_train_ranker.columns),
                "x_val_columns": list(x_val_ranker.columns),
                "x_test_columns": list(x_test_ranker.columns),
            }
        )
        return {"target_mode": "cross_sectional_rank_ndcg"}

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        model_training,
        "train_cross_sectional_rank_ndcg_model",
        fake_train_cross_sectional_rank_ndcg_model,
    )

    model_training.train_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
    )
    model_training.train_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
        feature_columns_override=override_features,
    )

    assert captured[0]["all_features"] == MODEL_FEATURE_COLUMNS
    assert captured[0]["classifier_features"] == MODEL_FEATURE_COLUMNS
    assert captured[0]["x_train_columns"] == MODEL_FEATURE_COLUMNS
    assert captured[1]["all_features"] == MODEL_FEATURE_COLUMNS
    assert captured[1]["classifier_features"] == override_features
    assert captured[1]["x_train_columns"] == override_features
    assert captured[1]["x_val_columns"] == override_features
    assert captured[1]["x_test_columns"] == override_features


@pytest.mark.parametrize(
    "missing_column, expected_message",
    [
        ("prediction_date", "prediction_date"),
        ("excess_return_rank_pct_by_date", "excess_return_rank_pct_by_date"),
    ],
)
def test_train_models_cross_sectional_rank_ndcg_requires_rank_metadata(
    monkeypatch,
    missing_column,
    expected_message,
):
    feature_count = len(MODEL_FEATURE_COLUMNS)

    class FakePreparator:
        def __init__(self):
            self.feature_columns = list(MODEL_FEATURE_COLUMNS)

        def prepare_for_train(self, data, prediction_days, test_size):
            rank_metadata = pd.DataFrame(
                {
                    "prediction_date": pd.to_datetime(
                        ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
                    ),
                    "excess_return_rank_pct_by_date": [0.1, 0.9, 0.2, 0.8],
                }
            ).drop(columns=[missing_column])
            return {
                "x_train": np.ones((4, feature_count)),
                "x_val": np.ones((2, feature_count)),
                "x_test": np.ones((2, feature_count)),
                "y_train": np.zeros(4),
                "y_val": np.zeros(2),
                "y_test": np.zeros(2),
                "direction_y_train": np.zeros(4),
                "direction_y_val": np.zeros(2),
                "direction_y_test": np.zeros(2),
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "split_metadata": {
                    "train": rank_metadata,
                    "val": rank_metadata.copy(),
                    "test": rank_metadata.copy(),
                },
            }

    monkeypatch.setattr(model_training, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        model_training,
        "XGBRanker",
        lambda *args, **kwargs: pytest.fail("ranker should not train"),
    )

    with pytest.raises(ValueError, match=expected_message):
        model_training.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_rank_ndcg",
        )
