"""Tests for walk-forward model execution and target-mode dispatch."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_contract import MODEL_FEATURE_COLUMNS
import src.train.walk_forward_runner as walk_forward_runner


def test_prepare_walk_forward_model_frame_uses_public_data_preparator_api():
    captured = {}
    data = pd.DataFrame({"Close": [1.0]})
    prepared_frame = pd.DataFrame(
        {"prediction_date": pd.to_datetime(["2020-01-01"])}
    )

    class FakePreparator:
        def prepare_model_frame(self, received_data, prediction_days):
            captured["data"] = received_data
            captured["prediction_days"] = prediction_days
            return prepared_frame, ["feature_a"]

    result = walk_forward_runner.prepare_walk_forward_model_frame(
        data,
        prediction_days=10,
        data_preparator=FakePreparator(),
    )

    assert captured["data"] is data
    assert captured["prediction_days"] == 10
    assert result[0] is prepared_frame
    assert result[1] == ["feature_a"]


def test_run_walk_forward_models_records_selected_candidate_and_test_metrics(
    monkeypatch,
):
    captured = {}
    prepared_frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2020-01-01"])})
    fold = {
        "fold_index": 0,
        "train_years": [2015, 2016, 2017, 2018, 2019],
        "validation_years": [2020],
        "test_years": [2021],
        "train_date_range": {"start": "2015-01-01", "end": "2019-12-31"},
        "validation_date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "test_date_range": {"start": "2021-01-01", "end": "2021-12-31"},
    }

    class FakeRegressor:
        def predict(self, x):
            captured["predict_x"] = x.copy()
            return np.array([0.20, 0.10])

    def fake_select(
        x_train,
        y_train,
        x_val,
        y_val,
        validation_split_metadata,
        candidate_configs=None,
    ):
        captured["candidate_configs"] = candidate_configs
        pd.testing.assert_frame_equal(
            validation_split_metadata,
            pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
        )
        return FakeRegressor(), {
            "selected_candidate_id": 1,
            "selected_candidate_name": "candidate_1",
            "selected_validation_top_n_mean_excess_return": 0.03,
        }

    def fake_top_n_reports(
        split_metadata,
        ranked_predictions,
        prediction_days=None,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["top_n_predictions"] = np.asarray(ranked_predictions)
        captured["top_n_prediction_days"] = prediction_days
        captured["top_n_random_trials"] = random_trials
        captured["top_n_random_trial_workers"] = random_trial_workers
        return {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.04},
                    "momentum_baseline": {"average_basket_excess_return": 0.01},
                    "universe": {"average_basket_excess_return": 0.02},
                }
            }
        }

    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (prepared_frame, ["feature_a"]),
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_walk_forward_split",
        lambda *args, **kwargs: {
            "x_train": np.array([[1.0], [2.0]]),
            "x_val": np.array([[3.0], [4.0]]),
            "x_test": np.array([[5.0], [6.0]]),
            "y_train": np.array([0.01, 0.02]),
            "y_val": np.array([0.03, 0.04]),
            "split_metadata": {
                "val": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
                "test": pd.DataFrame({"Ticker": ["AAA", "BBB"]}),
            },
            "split_date_ranges": {
                "train": {"start": "2015-01-01", "end": "2019-12-15"},
                "validation": {"start": "2020-01-01", "end": "2020-12-15"},
                "test": {"start": "2021-01-01", "end": "2021-12-31"},
            },
        },
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_xgboost_regressor_candidate_configs",
        lambda params: [
            {"candidate_id": 0, "candidate_name": "candidate_0", "params": {}}
        ],
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "select_xgboost_regressor_by_validation_top_n",
        fake_select,
    )
    monkeypatch.setattr(walk_forward_runner, "build_top_n_selection_reports", fake_top_n_reports)
    monkeypatch.setattr(
        walk_forward_runner,
        "XGBRanker",
        lambda *args, **kwargs: pytest.fail("ranker path should not run"),
    )

    report = walk_forward_runner.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        prediction_days=10,
        random_trials=7,
        random_trial_workers=2,
    )

    assert report["prediction_days"] == 10
    assert report["target_mode"] == "excess_return"
    assert report["folds"][0]["selected_candidate_name"] == "candidate_1"
    assert report["folds"][0]["validation_selection_score"] == 0.03
    assert report["folds"][0]["train_date_range"] == {
        "start": "2015-01-01",
        "end": "2019-12-15",
    }
    assert report["folds"][0]["validation_date_range"] == {
        "start": "2020-01-01",
        "end": "2020-12-15",
    }
    assert report["folds"][0]["test_date_range"] == {
        "start": "2021-01-01",
        "end": "2021-12-31",
    }
    assert np.isclose(report["folds"][0]["top_n"]["top_5"]["model_excess"], 0.04)
    assert np.isclose(
        report["folds"][0]["top_n"]["top_5"]["model_minus_momentum"],
        0.03,
    )
    np.testing.assert_array_equal(captured["top_n_predictions"], [0.20, 0.10])
    assert captured["top_n_prediction_days"] == 10
    assert captured["top_n_random_trials"] == 7
    assert captured["top_n_random_trial_workers"] == 2
    assert report["aggregate"]["selected_candidate_counts"] == {"candidate_1": 1}


def test_run_walk_forward_models_rank_ndcg_trains_grouped_ranker_and_evaluates_top_n(
    monkeypatch,
):
    captured = {}
    prepared_frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2020-01-01"])})
    fold = {
        "fold_index": 0,
        "train_years": [2015, 2016, 2017, 2018, 2019],
        "validation_years": [2020],
        "test_years": [2021],
        "train_date_range": {"start": "2015-01-01", "end": "2019-12-31"},
        "validation_date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "test_date_range": {"start": "2021-01-01", "end": "2021-12-31"},
    }

    class FakeRanker:
        def __init__(self, **params):
            captured["ranker_params"] = params

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
            captured["predict_x"] = x.copy()
            return np.array([0.70, 0.40, 0.90, 0.20])

    def fake_rank_ndcg_report(
        test_split_metadata,
        ranking_scores,
        prediction_days=10,
        random_trials=100,
        random_trial_workers=4,
    ):
        captured["rank_ndcg_metadata"] = test_split_metadata.copy()
        captured["rank_ndcg_scores"] = np.asarray(ranking_scores)
        captured["rank_ndcg_prediction_days"] = prediction_days
        captured["rank_ndcg_random_trials"] = random_trials
        captured["rank_ndcg_random_trial_workers"] = random_trial_workers
        return {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.04},
                    "momentum_baseline": {"average_basket_excess_return": 0.01},
                    "universe": {"average_basket_excess_return": 0.02},
                }
            }
        }

    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (prepared_frame, ["feature_a"]),
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_walk_forward_split",
        lambda *args, **kwargs: {
            "x_train": np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            "x_val": np.array([[10.0], [11.0]]),
            "x_test": np.array([[20.0], [21.0], [22.0], [23.0]]),
            "y_train": np.zeros(5),
            "y_val": np.zeros(2),
            "split_metadata": {
                "train": pd.DataFrame(
                    {
                        "Ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                        "prediction_date": pd.to_datetime(
                            [
                                "2019-01-02",
                                "2019-01-01",
                                "2019-01-02",
                                "2019-01-01",
                                "2019-01-03",
                            ]
                        ),
                        "excess_return_rank_pct_by_date": [
                            0.90,
                            0.10,
                            0.30,
                            0.70,
                            0.80,
                        ],
                    }
                ),
                "val": pd.DataFrame(
                    {
                        "Ticker": ["AAA", "BBB"],
                        "prediction_date": pd.to_datetime(
                            ["2020-01-02", "2020-01-02"]
                        ),
                        "excess_return_rank_pct_by_date": [0.20, 0.90],
                    }
                ),
                "test": pd.DataFrame(
                    {
                        "Ticker": ["AAA", "BBB", "CCC", "DDD"],
                        "prediction_date": pd.to_datetime(
                            [
                                "2021-01-02",
                                "2021-01-02",
                                "2021-01-03",
                                "2021-01-03",
                            ]
                        ),
                        "excess_return_rank_pct_by_date": [0.75, 0.50, 0.00, 1.00],
                    }
                ),
            },
            "split_date_ranges": {
                "train": {"start": "2015-01-01", "end": "2019-12-15"},
                "validation": {"start": "2020-01-01", "end": "2020-12-15"},
                "test": {"start": "2021-01-01", "end": "2021-12-31"},
            },
        },
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "select_xgboost_regressor_by_validation_top_n",
        lambda *args, **kwargs: pytest.fail("regression path should not run"),
    )
    monkeypatch.setattr(walk_forward_runner, "XGBRanker", FakeRanker)
    monkeypatch.setattr(walk_forward_runner, "log_rank_ndcg_test_report", fake_rank_ndcg_report)

    report = walk_forward_runner.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        prediction_days=10,
        random_trials=7,
        random_trial_workers=2,
        target_mode="cross_sectional_rank_ndcg",
    )

    np.testing.assert_array_equal(captured["fit_first_feature"], [1.0, 3.0, 0.0, 2.0])
    np.testing.assert_array_equal(captured["fit_y"], [0, 3, 4, 1])
    np.testing.assert_array_equal(captured["fit_qid"], [0, 0, 1, 1])
    np.testing.assert_array_equal(captured["eval_first_feature"], [10.0, 11.0])
    np.testing.assert_array_equal(captured["eval_y"], [0, 4])
    np.testing.assert_array_equal(captured["eval_qid"], [0, 0])
    assert captured["fit_verbose"] is False
    assert captured["ranker_params"] == walk_forward_runner.XG_PARAMS_RANKER
    np.testing.assert_array_equal(
        captured["rank_ndcg_scores"],
        [0.70, 0.40, 0.90, 0.20],
    )
    assert len(captured["rank_ndcg_metadata"]) == 4
    assert captured["rank_ndcg_prediction_days"] == 10
    assert captured["rank_ndcg_random_trials"] == 7
    assert captured["rank_ndcg_random_trial_workers"] == 2
    assert report["target_mode"] == "cross_sectional_rank_ndcg"
    assert report["folds"][0]["selected_candidate_name"] == "rank_ndcg"
    assert report["folds"][0]["validation_selection_score"] is None
    assert np.isclose(report["folds"][0]["top_n"]["top_5"]["model_excess"], 0.04)
    assert np.isclose(
        report["folds"][0]["top_n"]["top_5"]["model_minus_momentum"],
        0.03,
    )


def test_run_walk_forward_models_rank_ndcg_without_validation_eval_set_removes_early_stopping(
    monkeypatch,
):
    captured = {}
    fold = {
        "fold_index": 0,
        "train_years": [2019],
        "validation_years": [2020],
        "test_years": [2021],
        "train_date_range": {"start": "2019-01-01", "end": "2019-12-31"},
        "validation_date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        "test_date_range": {"start": "2021-01-01", "end": "2021-12-31"},
    }

    class FakeRanker:
        def __init__(self, **params):
            captured["ranker_params"] = params

        def fit(self, x, y, *, qid=None, eval_set=None, eval_qid=None, verbose=True):
            captured["eval_set"] = eval_set
            captured["eval_qid"] = eval_qid
            return self

        def predict(self, x):
            return np.array([0.70, 0.40])

    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (
            pd.DataFrame({"prediction_date": pd.to_datetime(["2020-01-01"])}),
            ["feature_a"],
        ),
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_walk_forward_split",
        lambda *args, **kwargs: {
            "x_train": np.array([[0.0], [1.0], [2.0], [3.0]]),
            "x_val": np.array([[10.0]]),
            "x_test": np.array([[20.0], [21.0]]),
            "y_train": np.zeros(4),
            "y_val": np.zeros(1),
            "split_metadata": {
                "train": pd.DataFrame(
                    {
                        "prediction_date": pd.to_datetime(
                            [
                                "2019-01-01",
                                "2019-01-01",
                                "2019-01-02",
                                "2019-01-02",
                            ]
                        ),
                        "excess_return_rank_pct_by_date": [0.10, 0.90, 0.30, 0.70],
                    }
                ),
                "val": pd.DataFrame(
                    {
                        "prediction_date": pd.to_datetime(["2020-01-01"]),
                        "excess_return_rank_pct_by_date": [0.90],
                    }
                ),
                "test": pd.DataFrame(
                    {
                        "prediction_date": pd.to_datetime(
                            ["2021-01-01", "2021-01-01"]
                        ),
                        "excess_return_rank_pct_by_date": [0.10, 0.90],
                    }
                ),
            },
        },
    )
    monkeypatch.setattr(walk_forward_runner, "XGBRanker", FakeRanker)
    monkeypatch.setattr(
        walk_forward_runner,
        "log_rank_ndcg_test_report",
        lambda *args, **kwargs: {
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.04},
                    "momentum_baseline": {"average_basket_excess_return": 0.01},
                    "universe": {"average_basket_excess_return": 0.02},
                }
            }
        },
    )

    walk_forward_runner.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
    )

    assert captured["eval_set"] is None
    assert captured["eval_qid"] is None
    assert "early_stopping_rounds" not in captured["ranker_params"]


def test_run_walk_forward_models_rejects_unsupported_target_mode(monkeypatch):
    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: pytest.fail("walk-forward prep should not run"),
    )

    with pytest.raises(ValueError, match="Unsupported walk-forward target_mode"):
        walk_forward_runner.run_walk_forward_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_top_bottom",
        )


def test_run_walk_forward_models_without_override_uses_prepared_full_feature_contract(
    monkeypatch,
):
    captured = {}
    prepared_frame = pd.DataFrame(
        {"prediction_date": pd.to_datetime(["2020-01-01"])}
    )
    fold = {
        "fold_index": 0,
        "train_years": [2019],
        "validation_years": [2020],
        "test_years": [2021],
        "train_date_range": {"start": None, "end": None},
        "validation_date_range": {"start": None, "end": None},
        "test_date_range": {"start": None, "end": None},
    }

    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: (prepared_frame, list(MODEL_FEATURE_COLUMNS)),
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )

    def fake_build_split(frame, current_fold, feature_columns, **kwargs):
        captured["split_features"] = list(feature_columns)
        return {"split_date_ranges": {}}

    def fake_run_regressor_fold(
        split,
        feature_columns,
        regressor_candidate_configs,
        **kwargs,
    ):
        captured["fit_features"] = list(feature_columns)
        return {
            "selected_candidate_id": 0,
            "selected_candidate_name": "candidate_0",
            "selected_validation_top_n_mean_excess_return": 0.0,
        }, {}

    monkeypatch.setattr(walk_forward_runner, "build_walk_forward_split", fake_build_split)
    monkeypatch.setattr(
        walk_forward_runner,
        "_run_excess_return_walk_forward_fold",
        fake_run_regressor_fold,
    )

    walk_forward_runner.run_walk_forward_models(pd.DataFrame({"Close": [1.0]}))

    assert captured["split_features"] == MODEL_FEATURE_COLUMNS
    assert captured["fit_features"] == MODEL_FEATURE_COLUMNS


def test_run_walk_forward_models_feature_override_preserves_order_for_all_splits(
    monkeypatch,
):
    captured = {}
    override = [MODEL_FEATURE_COLUMNS[5], MODEL_FEATURE_COLUMNS[0]]
    prepared_frame = pd.DataFrame(
        {"prediction_date": pd.to_datetime(["2020-01-01"])}
    )
    fold = {
        "fold_index": 0,
        "train_years": [2019],
        "validation_years": [2020],
        "test_years": [2021],
        "train_date_range": {"start": None, "end": None},
        "validation_date_range": {"start": None, "end": None},
        "test_date_range": {"start": None, "end": None},
    }

    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: (prepared_frame, list(MODEL_FEATURE_COLUMNS)),
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )

    def fake_build_split(frame, current_fold, feature_columns, **kwargs):
        captured["split_features"] = list(feature_columns)
        return {
            "x_train": np.zeros((3, len(feature_columns))),
            "x_val": np.zeros((2, len(feature_columns))),
            "x_test": np.zeros((4, len(feature_columns))),
            "split_date_ranges": {},
        }

    def fake_run_ranker_fold(split, feature_columns, **kwargs):
        captured["fit_features"] = list(feature_columns)
        captured["split_widths"] = [
            split[key].shape[1] for key in ("x_train", "x_val", "x_test")
        ]
        return {
            "selected_candidate_id": None,
            "selected_candidate_name": "rank_ndcg",
            "selected_validation_top_n_mean_excess_return": None,
        }, {}

    monkeypatch.setattr(walk_forward_runner, "build_walk_forward_split", fake_build_split)
    monkeypatch.setattr(
        walk_forward_runner,
        "_run_rank_ndcg_walk_forward_fold",
        fake_run_ranker_fold,
    )

    walk_forward_runner.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
        feature_columns_override=override,
    )

    assert captured["split_features"] == override
    assert captured["fit_features"] == override
    assert captured["split_widths"] == [2, 2, 2]


@pytest.mark.parametrize(
    "feature_columns_override",
    [[], ["targetReturns"]],
)
def test_run_walk_forward_models_rejects_invalid_feature_overrides_before_preparation(
    monkeypatch,
    feature_columns_override,
):
    monkeypatch.setattr(
        walk_forward_runner,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: pytest.fail("walk-forward prep should not run"),
    )

    with pytest.raises(ValueError, match="feature_columns_override"):
        walk_forward_runner.run_walk_forward_models(
            pd.DataFrame({"Close": [1.0]}),
            feature_columns_override=feature_columns_override,
        )

