"""Trainer tests for caching, metadata, CLI validation, and report plumbing."""

import importlib.util
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.config import (
    DEFAULT_TRAINING_UNIVERSE,
    LARGE_MEGA_CAP_STOCKS,
    TRAINING_TICKERS,
    TRAINING_UNIVERSES,
    XG_PARAMS_REGRESSOR,
    get_training_tickers,
)
from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)
from src.train.rank_ndcg_feature_ablations import (
    FeatureAblationSpec,
    build_rank_ndcg_feature_ablation_specs,
    extract_ablation_result_row,
    format_ablation_table,
)
import src.data.training_data as training_data
import src.train.artifacts as artifacts
import src.train.training_contract as training_contract
import src.train.reporting as reporting
import src.train.trainer as trainer


ABLATION_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_rank_ndcg_feature_ablations.py"
)
ABLATION_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_rank_ndcg_feature_ablations",
    ABLATION_SCRIPT_PATH,
)
ablation_script = importlib.util.module_from_spec(ABLATION_SCRIPT_SPEC)
ABLATION_SCRIPT_SPEC.loader.exec_module(ablation_script)


def test_trainer_reexports_training_data_public_api():
    public_names = (
        "PROJECT_ROOT",
        "YFINANCE_CACHE_DIR",
        "YFINANCE_CACHE_FORMAT",
        "get_yfinance_cache_path",
        "normalize_raw_ohlcv_index",
        "load_cached_yfinance_data",
        "write_yfinance_cache",
        "fetch_raw_ticker_data",
        "fetch_tickers_data",
        "prepare_data_parallel",
        "validate_input_data",
    )

    for name in public_names:
        assert getattr(trainer, name) is getattr(training_data, name)


def test_trainer_reexports_training_contract_public_api():
    public_names = (
        "TARGET_MODE_EXCESS_RETURN",
        "TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM",
        "TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG",
        "TARGET_MODES",
        "WALK_FORWARD_TARGET_MODES",
        "validate_target_mode",
        "validate_walk_forward_target_mode",
        "resolve_model_feature_columns",
    )

    for name in public_names:
        assert getattr(trainer, name) is getattr(training_contract, name)


def test_trainer_reexports_reporting_public_api():
    public_names = (
        "log_feature_importances",
        "format_top_n_ranked_selection_summary",
        "format_top_n_basket_backtest_summary",
        "format_top_n_basket_backtest_by_year_summary",
        "format_horizon_comparison_summary",
        "format_same_date_ranking_diagnostics_summary",
        "format_xgboost_regressor_validation_selection_report",
        "evaluate_model",
        "log_xgboost_test_report",
        "log_rank_ndcg_test_report",
        "log_ranking_classifier_test_report",
        "format_walk_forward_fold_summary",
        "format_walk_forward_summary",
    )

    for name in public_names:
        assert getattr(trainer, name) is getattr(reporting, name)


def test_trainer_reexports_artifact_public_api():
    public_names = (
        "build_model_metadata",
        "build_horizon_model_paths",
        "build_rank_ndcg_model_paths",
        "save_model_artifacts",
        "save_horizon_model_artifacts",
        "save_ranking_model_artifacts",
        "save_rank_ndcg_model_artifacts",
    )

    for name in public_names:
        assert getattr(trainer, name) is getattr(artifacts, name)


def test_default_training_universe_is_large_mega_cap_stocks():
    assert DEFAULT_TRAINING_UNIVERSE == "large_mega_cap_stocks"


def test_training_tickers_match_default_large_mega_cap_universe():
    assert TRAINING_TICKERS == LARGE_MEGA_CAP_STOCKS
    assert TRAINING_TICKERS == get_training_tickers()


def test_named_training_universes_have_no_duplicates():
    for universe_name, tickers in TRAINING_UNIVERSES.items():
        assert len(tickers) == len(set(tickers)), universe_name


def test_named_training_universes_include_spy_benchmark():
    for universe_name, tickers in TRAINING_UNIVERSES.items():
        assert "SPY" in tickers, universe_name


def test_get_training_tickers_returns_copy():
    tickers = get_training_tickers("large_mega_cap_stocks")
    tickers.append("SHOULD_NOT_MUTATE")

    assert "SHOULD_NOT_MUTATE" not in get_training_tickers("large_mega_cap_stocks")
    assert "SHOULD_NOT_MUTATE" not in TRAINING_UNIVERSES["large_mega_cap_stocks"]


def test_get_training_tickers_rejects_unknown_universe_with_valid_choices():
    with pytest.raises(ValueError) as exc_info:
        get_training_tickers("bad_name")

    message = str(exc_info.value)
    assert "bad_name" in message
    for universe_name in TRAINING_UNIVERSES:
        assert universe_name in message


def test_xgboost_regressor_candidates_include_default_baseline():
    candidates = trainer.build_xgboost_regressor_candidate_configs()

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

    monkeypatch.setattr(trainer, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        trainer,
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

    selected_model, report = trainer.select_xgboost_regressor_by_validation_top_n(
        x_train,
        y_train,
        x_val,
        y_val,
        validation_metadata,
        candidate_configs=candidates,
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(trainer, "LinearRegression", FakeLinearRegression)
    monkeypatch.setattr(trainer, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(trainer, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        trainer,
        "build_xgboost_regressor_candidate_configs",
        lambda base_params: candidates,
    )
    monkeypatch.setattr(
        trainer,
        "build_model_only_top_n_basket_backtest_report",
        fake_model_only_report,
    )
    monkeypatch.setattr(trainer, "evaluate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer, "log_xgboost_test_report", fake_log_xgboost_test_report
    )
    monkeypatch.setattr(
        trainer, "log_feature_importances", lambda *args, **kwargs: None
    )

    def fake_save_horizon_model_artifacts(*args, **kwargs):
        captured["saved_model_metadata"] = args[5]
        return {"model_metadata": "models/horizon_10/model_metadata.pkl"}

    monkeypatch.setattr(
        trainer,
        "save_horizon_model_artifacts",
        fake_save_horizon_model_artifacts,
    )

    report = trainer.train_models(pd.DataFrame({"Close": [1.0]}), prediction_days=10)

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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(trainer, "LinearRegression", FakeLinearRegression)
    monkeypatch.setattr(trainer, "XGBClassifier", FakeXgbModel)
    monkeypatch.setattr(trainer, "XGBRegressor", FakeXgbModel)
    monkeypatch.setattr(trainer, "evaluate_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer,
        "log_xgboost_test_report",
        lambda *args, **kwargs: {
            "basket_backtest": {},
            "same_date_ranking_diagnostics": {"available": True},
        },
    )
    monkeypatch.setattr(
        trainer, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        trainer,
        "save_horizon_model_artifacts",
        fake_save_horizon_model_artifacts,
    )

    trainer.train_models(pd.DataFrame({"Close": [1.0]}), prediction_days=10)

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
        trainer,
        "validate_input_data",
        lambda data: pytest.fail("training path should not run"),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported target_mode='bad_mode'",
    ):
        trainer.train_models(
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(trainer, "XGBClassifier", FakeRankingClassifier)
    monkeypatch.setattr(reporting, "build_top_n_selection_reports", fake_top_n_reports)
    monkeypatch.setattr(
        trainer,
        "save_ranking_model_artifacts",
        fake_save_ranking_model_artifacts,
    )
    monkeypatch.setattr(
        trainer, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        trainer,
        "select_xgboost_regressor_by_validation_top_n",
        lambda *args, **kwargs: pytest.fail("regression path should not run"),
    )

    report = trainer.train_models(
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        trainer,
        "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier should not train"),
    )

    with pytest.raises(
        ValueError,
        match=r"requires split_metadata\['train'\] columns",
    ):
        trainer.train_models(
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        trainer,
        "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier should not train"),
    )

    with pytest.raises(
        ValueError,
        match=r"target_mode='cross_sectional_top_bottom' requires split_metadata\['test'\]",
    ):
        trainer.train_models(
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        trainer,
        "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier should not train"),
    )

    with pytest.raises(
        ValueError,
        match="Ranking training sample must contain both classes",
    ):
        trainer.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_top_bottom",
        )


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

    ranked_features, labels, qid, grouped_metadata = trainer._rank_ndcg_training_data(
        features,
        metadata,
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
    labels = trainer._rank_percentile_to_ndcg_relevance(
        pd.Series([0.20, 0.21, 0.40, 0.41, 0.60, 0.61, 0.79, 0.80, 1.00])
    )

    np.testing.assert_array_equal(labels.to_numpy(), [0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_rank_ndcg_feature_ablation_specs_include_expected_subsets():
    specs = {
        spec.name: spec
        for spec in build_rank_ndcg_feature_ablation_specs()
    }

    assert list(specs) == [
        "all_features",
        "drop_raw_ohlc",
        "drop_price_level_trend",
        "drop_volume_scale",
        "drop_ichimoku",
        "drop_relative_momentum",
        "drop_macd_redundant",
        "minimal_momentum_risk_oscillator",
        "momentum_only",
        "momentum_plus_volatility",
    ]
    assert specs["all_features"].feature_columns == MODEL_FEATURE_COLUMNS
    assert set(["Open", "High", "Low", "Close"]).isdisjoint(
        specs["drop_raw_ohlc"].feature_columns
    )
    assert "Volume" in specs["drop_raw_ohlc"].feature_columns
    assert set(["BB_Middle", "BB_Upper", "BB_Lower"]).isdisjoint(
        specs["drop_price_level_trend"].feature_columns
    )
    assert "BB_Std" in specs["drop_price_level_trend"].feature_columns
    assert specs["momentum_only"].feature_columns == ABSOLUTE_MOMENTUM_FEATURE_COLUMNS
    minimal = specs["minimal_momentum_risk_oscillator"].feature_columns
    assert "macdHistogram" in minimal
    assert "macd" not in minimal
    assert "signalLine" not in minimal


def test_rank_ndcg_ablation_result_extraction_handles_missing_metrics():
    spec = FeatureAblationSpec(
        name="tiny",
        feature_columns=["momentum_10d"],
        included_groups="absolute_momentum",
    )
    report = {
        "aggregate": {
            "fold_count": 5,
            "top_n": {
                "top_5": {
                    "average_model_excess": 0.03,
                    "average_model_minus_momentum": 0.01,
                    "fold_win_rate_vs_momentum": 0.6,
                    "average_model_minus_universe": 0.02,
                    "fold_win_rate_vs_universe": 0.8,
                }
            },
        },
    }

    row = extract_ablation_result_row(
        report,
        spec,
        prediction_days=10,
        period="10y",
        universe="large_mega_cap_stocks",
    )

    assert row["ablation_name"] == "tiny"
    assert row["feature_count"] == 1
    assert row["evaluated_fold_count"] == 5
    assert row["top_5_average_model_excess"] == 0.03
    assert row["top_5_average_model_minus_momentum"] == 0.01
    assert row["top_5_fold_win_rate_vs_momentum"] == 0.6
    assert row["top_5_average_model_minus_universe"] == 0.02
    assert row["top_5_fold_win_rate_vs_universe"] == 0.8
    assert np.isnan(row["top_10_average_model_excess"])
    assert np.isnan(row["top_20_fold_win_rate_vs_universe"])

    missing_row = extract_ablation_result_row(
        {},
        spec,
        prediction_days=10,
        period="10y",
        universe="large_mega_cap_stocks",
    )
    assert missing_row["evaluated_fold_count"] == 0
    assert np.isnan(missing_row["top_5_average_model_excess"])


def test_rank_ndcg_ablation_table_includes_required_stdout_columns():
    row = {
        "ablation_name": "all_features",
        "feature_count": 39,
        "top_5_average_model_excess": 0.01,
        "top_10_average_model_excess": 0.02,
        "top_5_average_model_minus_momentum": 0.001,
        "top_10_average_model_minus_momentum": 0.002,
        "top_5_fold_win_rate_vs_momentum": 0.60,
        "top_10_fold_win_rate_vs_momentum": 0.40,
    }

    table = format_ablation_table([row])

    for column in [
        "features",
        "top5_avg_excess",
        "top10_avg_excess",
        "top5_avg_minus_mom",
        "top10_avg_minus_mom",
        "top5_win_vs_mom",
        "top10_win_vs_mom",
    ]:
        assert column in table


def test_rank_ndcg_ablation_runner_uses_walk_forward_for_every_feature_spec(
    monkeypatch,
    tmp_path,
):
    calls = []
    specs = build_rank_ndcg_feature_ablation_specs()

    monkeypatch.setattr(
        ablation_script,
        "prepare_data_parallel",
        lambda *args, **kwargs: pd.DataFrame({"Close": [1.0]}),
    )

    def fake_run_walk_forward_models(data, **kwargs):
        calls.append(kwargs)
        return {
            "aggregate": {
                "fold_count": 5,
                "top_n": {
                    f"top_{top_n}": {
                        "average_model_excess": top_n / 1000,
                        "average_model_minus_momentum": top_n / 10000,
                        "fold_win_rate_vs_momentum": 0.6,
                        "average_model_minus_universe": top_n / 2000,
                        "fold_win_rate_vs_universe": 0.8,
                    }
                    for top_n in (5, 10, 20)
                },
            }
        }

    monkeypatch.setattr(
        ablation_script,
        "run_walk_forward_models",
        fake_run_walk_forward_models,
    )
    output_path = tmp_path / "ablation.csv"

    results = ablation_script.main(
        [
            "--prediction-days",
            "20",
            "--period",
            "10y",
            "--random-trials",
            "7",
            "--output",
            str(output_path),
        ]
    )

    assert len(calls) == len(specs)
    assert [call["feature_columns_override"] for call in calls] == [
        spec.feature_columns for spec in specs
    ]
    assert all(
        call["target_mode"]
        == training_contract.TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG
        for call in calls
    )
    assert all(call["random_trial_workers"] == 8 for call in calls)
    assert results[0]["evaluated_fold_count"] == 5
    assert results[0]["top_10_average_model_minus_momentum"] == 0.001
    csv_result = pd.read_csv(output_path)
    assert "top_5_fold_win_rate_vs_momentum" in csv_result.columns
    assert "top_20_average_model_minus_universe" in csv_result.columns


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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(trainer, "XGBRanker", FakeRanker)
    monkeypatch.setattr(reporting, "build_top_n_selection_reports", fake_top_n_reports)
    monkeypatch.setattr(
        trainer,
        "save_rank_ndcg_model_artifacts",
        fake_save_rank_ndcg_model_artifacts,
    )
    monkeypatch.setattr(
        trainer, "log_feature_importances", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        trainer,
        "select_xgboost_regressor_by_validation_top_n",
        lambda *args, **kwargs: pytest.fail("regression path should not run"),
    )

    report = trainer.train_models(
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
        trainer,
        "DataPreparator",
        lambda *args, **kwargs: pytest.fail("data prep should not run"),
    )

    with pytest.raises(ValueError, match="feature_columns_override"):
        trainer.train_models(
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        trainer,
        "train_cross_sectional_rank_ndcg_model",
        fake_train_cross_sectional_rank_ndcg_model,
    )

    trainer.train_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
    )
    trainer.train_models(
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

    monkeypatch.setattr(trainer, "DataPreparator", FakePreparator)
    monkeypatch.setattr(
        trainer,
        "XGBRanker",
        lambda *args, **kwargs: pytest.fail("ranker should not train"),
    )

    with pytest.raises(ValueError, match=expected_message):
        trainer.train_models(
            pd.DataFrame({"Close": [1.0]}),
            target_mode="cross_sectional_rank_ndcg",
        )


def test_rank_ndcg_training_data_rejects_no_usable_rows():
    features = pd.DataFrame({"feature": [0, 1]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "excess_return_rank_pct_by_date": [np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="No usable ranker training rows"):
        trainer._rank_ndcg_training_data(features, metadata)


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
        trainer._rank_ndcg_training_data(features, metadata)


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
        trainer,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (prepared_frame, ["feature_a"]),
    )
    monkeypatch.setattr(
        trainer,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        trainer,
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
        trainer,
        "build_xgboost_regressor_candidate_configs",
        lambda params: [
            {"candidate_id": 0, "candidate_name": "candidate_0", "params": {}}
        ],
    )
    monkeypatch.setattr(
        trainer,
        "select_xgboost_regressor_by_validation_top_n",
        fake_select,
    )
    monkeypatch.setattr(trainer, "build_top_n_selection_reports", fake_top_n_reports)
    monkeypatch.setattr(
        trainer,
        "XGBRanker",
        lambda *args, **kwargs: pytest.fail("ranker path should not run"),
    )

    report = trainer.run_walk_forward_models(
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
        trainer,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (prepared_frame, ["feature_a"]),
    )
    monkeypatch.setattr(
        trainer,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        trainer,
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
        trainer,
        "select_xgboost_regressor_by_validation_top_n",
        lambda *args, **kwargs: pytest.fail("regression path should not run"),
    )
    monkeypatch.setattr(trainer, "XGBRanker", FakeRanker)
    monkeypatch.setattr(trainer, "log_rank_ndcg_test_report", fake_rank_ndcg_report)

    report = trainer.run_walk_forward_models(
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
    assert captured["ranker_params"] == trainer.XG_PARAMS_RANKER
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
        trainer,
        "prepare_walk_forward_model_frame",
        lambda data, prediction_days: (
            pd.DataFrame({"prediction_date": pd.to_datetime(["2020-01-01"])}),
            ["feature_a"],
        ),
    )
    monkeypatch.setattr(
        trainer,
        "build_expanding_yearly_walk_forward_folds",
        lambda *args, **kwargs: [fold],
    )
    monkeypatch.setattr(
        trainer,
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
    monkeypatch.setattr(trainer, "XGBRanker", FakeRanker)
    monkeypatch.setattr(
        trainer,
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

    trainer.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
    )

    assert captured["eval_set"] is None
    assert captured["eval_qid"] is None
    assert "early_stopping_rounds" not in captured["ranker_params"]


def test_run_walk_forward_models_rejects_unsupported_target_mode(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: pytest.fail("walk-forward prep should not run"),
    )

    with pytest.raises(ValueError, match="Unsupported walk-forward target_mode"):
        trainer.run_walk_forward_models(
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
        trainer,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: (prepared_frame, list(MODEL_FEATURE_COLUMNS)),
    )
    monkeypatch.setattr(
        trainer,
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

    monkeypatch.setattr(trainer, "build_walk_forward_split", fake_build_split)
    monkeypatch.setattr(
        trainer,
        "_run_excess_return_walk_forward_fold",
        fake_run_regressor_fold,
    )

    trainer.run_walk_forward_models(pd.DataFrame({"Close": [1.0]}))

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
        trainer,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: (prepared_frame, list(MODEL_FEATURE_COLUMNS)),
    )
    monkeypatch.setattr(
        trainer,
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

    monkeypatch.setattr(trainer, "build_walk_forward_split", fake_build_split)
    monkeypatch.setattr(
        trainer,
        "_run_rank_ndcg_walk_forward_fold",
        fake_run_ranker_fold,
    )

    trainer.run_walk_forward_models(
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
        trainer,
        "prepare_walk_forward_model_frame",
        lambda *args, **kwargs: pytest.fail("walk-forward prep should not run"),
    )

    with pytest.raises(ValueError, match="feature_columns_override"):
        trainer.run_walk_forward_models(
            pd.DataFrame({"Close": [1.0]}),
            feature_columns_override=feature_columns_override,
        )


def test_parse_args_defaults_to_ten_prediction_days():
    args = trainer.parse_args([])

    assert args.prediction_days == 10
    assert args.horizons == [10]
    assert args.all_horizons is False
    assert args.period == "5y"
    assert args.universe == DEFAULT_TRAINING_UNIVERSE
    assert args.no_cache is False
    assert args.random_trials == 100
    assert args.random_trial_workers == 8
    assert args.walk_forward is False
    assert args.target_mode == "excess_return"


def test_parse_args_accepts_period():
    args = trainer.parse_args(["--period", "10y"])

    assert args.period == "10y"


def test_parse_args_accepts_training_universe():
    args = trainer.parse_args(["--universe", "broad_sector_etfs"])

    assert args.universe == "broad_sector_etfs"


def test_parse_args_accepts_excess_return_target_mode():
    args = trainer.parse_args(["--target-mode", "excess_return"])

    assert args.target_mode == "excess_return"


def test_parse_args_accepts_cross_sectional_top_bottom_target_mode():
    args = trainer.parse_args(["--target-mode", "cross_sectional_top_bottom"])

    assert args.target_mode == "cross_sectional_top_bottom"


def test_parse_args_accepts_cross_sectional_rank_ndcg_target_mode():
    args = trainer.parse_args(["--target-mode", "cross_sectional_rank_ndcg"])

    assert args.target_mode == "cross_sectional_rank_ndcg"


def test_parse_args_rejects_unknown_target_mode():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--target-mode", "bad_mode"])


def test_parse_args_accepts_random_trials():
    args = trainer.parse_args(["--random-trials", "20"])

    assert args.random_trials == 20


def test_parse_args_accepts_random_trial_workers():
    args = trainer.parse_args(["--random-trial-workers", "2"])

    assert args.random_trial_workers == 2


def test_parse_args_accepts_walk_forward_options():
    args = trainer.parse_args(
        [
            "--walk-forward",
            "--prediction-days",
            "20",
            "--walk-forward-min-train-years",
            "4",
            "--walk-forward-validation-years",
            "2",
            "--walk-forward-test-years",
            "1",
        ]
    )

    assert args.walk_forward is True
    assert args.horizons == [20]
    assert args.walk_forward_min_train_years == 4
    assert args.walk_forward_validation_years == 2
    assert args.walk_forward_test_years == 1


def test_parse_args_rejects_invalid_random_trials_values():
    for argv in [
        ["--random-trials", "0"],
        ["--random-trials", "-1"],
    ]:
        with pytest.raises(SystemExit):
            trainer.parse_args(argv)


def test_parse_args_rejects_removed_parallel_random_trials_flag():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--parallel-random-trials"])


def test_parse_args_rejects_invalid_random_trial_workers_values():
    for argv in [
        ["--random-trial-workers", "0"],
        ["--random-trial-workers", "-1"],
    ]:
        with pytest.raises(SystemExit):
            trainer.parse_args(argv)


def test_parse_args_accepts_prediction_days_long_and_short_flags():
    long_args = trainer.parse_args(["--prediction-days", "5"])
    short_args = trainer.parse_args(["-d", "5"])

    assert long_args.prediction_days == 5
    assert long_args.horizons == [5]
    assert short_args.prediction_days == 5
    assert short_args.horizons == [5]


def test_parse_args_rejects_invalid_prediction_days_values():
    for argv in [
        ["--prediction-days", "0"],
        ["--prediction-days", "-1"],
        ["--prediction-days", "abc"],
    ]:
        with pytest.raises(SystemExit):
            trainer.parse_args(argv)


def test_parse_args_all_horizons_resolves_research_horizons():
    args = trainer.parse_args(["--all-horizons"])

    assert args.all_horizons is True
    assert args.horizons == [5, 10, 20, 50]


def test_parse_args_rejects_all_horizons_with_prediction_days():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--all-horizons", "--prediction-days", "5"])


def test_parse_args_rejects_walk_forward_with_all_horizons():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--walk-forward", "--all-horizons"])


def test_parse_args_rejects_walk_forward_cross_sectional_top_bottom():
    with pytest.raises(SystemExit):
        trainer.parse_args(
            [
                "--walk-forward",
                "--target-mode",
                "cross_sectional_top_bottom",
            ]
        )


def test_parse_args_rejects_unknown_training_universe():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--universe", "bad_name"])


def test_main_trains_all_horizons_and_logs_comparison(monkeypatch, caplog, capsys):
    trained_horizons = []
    trained_target_modes = []
    trained_random_trials = []
    trained_random_trial_workers = []
    prepare_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        target_mode="excess_return",
    ):
        trained_horizons.append(prediction_days)
        trained_target_modes.append(target_mode)
        trained_random_trials.append(random_trials)
        trained_random_trial_workers.append(random_trial_workers)
        return {
            "prediction_days": prediction_days,
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": prediction_days / 1000},
                    "random_baseline": {"average_basket_excess_return": 0.001},
                    "momentum_baseline": {
                        "available": True,
                        "momentum_score_column": f"momentum_{prediction_days}d",
                        "average_basket_excess_return": 0.002,
                    },
                    "relative_momentum_baseline": {
                        "available": True,
                        "relative_momentum_score_column": (
                            f"relative_momentum_{prediction_days}d"
                        ),
                        "average_basket_excess_return": 0.003,
                    },
                    "universe": {"average_basket_excess_return": 0.0},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    with caplog.at_level(logging.INFO):
        trainer.main(
            [
                "--all-horizons",
                "--period",
                "10y",
                "--random-trials",
                "20",
                "--random-trial-workers",
                "2",
            ]
        )

    assert prepare_calls == [
        (get_training_tickers(DEFAULT_TRAINING_UNIVERSE), "10y", True)
    ]
    assert trained_horizons == [5, 10, 20, 50]
    assert trained_target_modes == ["excess_return"] * 4
    assert trained_random_trials == [20, 20, 20, 20]
    assert trained_random_trial_workers == [2, 2, 2, 2]
    assert "Selected YFinance period: 10y" in caplog.text
    assert "Selected target mode: excess_return" in caplog.text
    assert "Raw YFinance OHLCV cache enabled: True" in caplog.text
    assert "Horizon Comparison Summary:" in caplog.text
    assert "50d:" in caplog.text
    output = capsys.readouterr().out
    assert "Horizon Comparison Summary:" in output
    assert "50d:" in output
    assert "XGBoost Beat-Benchmark Classification Report" not in output


def test_main_passes_selected_broad_sector_etf_universe_to_prepare_data(
    monkeypatch,
):
    prepare_calls = []
    trained_horizons = []
    trained_target_modes = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        target_mode="excess_return",
    ):
        trained_horizons.append(prediction_days)
        trained_target_modes.append(target_mode)
        return {
            "prediction_days": prediction_days,
            "basket_backtest": {
                "top_5": {
                    "model": {"average_basket_excess_return": 0.01},
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": False,
                        "momentum_score_column": "momentum_10d",
                    },
                    "relative_momentum_baseline": {
                        "available": False,
                        "relative_momentum_score_column": "relative_momentum_10d",
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)
    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    trainer.main(
        [
            "--prediction-days",
            "10",
            "--period",
            "10y",
            "--universe",
            "broad_sector_etfs",
        ]
    )

    assert prepare_calls == [(get_training_tickers("broad_sector_etfs"), "10y", True)]
    assert trained_horizons == [10]
    assert trained_target_modes == ["excess_return"]


def test_main_single_horizon_prints_top_n_basket_summary(monkeypatch, capsys):
    trained_horizons = []
    trained_target_modes = []
    trained_random_trials = []
    trained_random_trial_workers = []
    prepare_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        target_mode="excess_return",
    ):
        trained_horizons.append(prediction_days)
        trained_target_modes.append(target_mode)
        trained_random_trials.append(random_trials)
        trained_random_trial_workers.append(random_trial_workers)
        return {
            "prediction_days": prediction_days,
            "basket_backtest": {
                "top_5": {
                    "model": {
                        "average_basket_raw_return": 0.02,
                        "average_basket_excess_return": 0.01,
                        "beat_benchmark_rate": 0.55,
                    },
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": True,
                        "momentum_score_column": "momentum_10d",
                        "average_basket_excess_return": 0.004,
                    },
                    "relative_momentum_baseline": {
                        "available": True,
                        "relative_momentum_score_column": "relative_momentum_10d",
                        "average_basket_excess_return": 0.005,
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    trainer.main(
        [
            "--prediction-days",
            "10",
            "--no-cache",
            "--random-trials",
            "20",
            "--random-trial-workers",
            "2",
        ]
    )

    assert prepare_calls == [
        (get_training_tickers(DEFAULT_TRAINING_UNIVERSE), "5y", False)
    ]
    assert trained_horizons == [10]
    assert trained_target_modes == ["excess_return"]
    assert trained_random_trials == [20]
    assert trained_random_trial_workers == [2]
    output = capsys.readouterr().out
    assert "XGBoost Top-N Basket Backtest Summary:" in output
    assert "top_5:" in output
    assert "Horizon Comparison Summary:" not in output
    assert "XGBoost Beat-Benchmark Classification Report" not in output


def test_main_passes_cross_sectional_target_mode_to_train_models(monkeypatch):
    trained_target_modes = []

    monkeypatch.setattr(
        trainer,
        "prepare_data_parallel",
        lambda *args, **kwargs: pd.DataFrame({"Close": [1.0]}),
    )

    def fake_train_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        target_mode="excess_return",
    ):
        trained_target_modes.append(target_mode)
        return {
            "prediction_days": prediction_days,
            "target_mode": target_mode,
            "basket_backtest": {
                "top_5": {
                    "model": {
                        "average_basket_raw_return": 0.02,
                        "average_basket_excess_return": 0.01,
                        "beat_benchmark_rate": 0.55,
                    },
                    "random_baseline": {"average_basket_excess_return": 0.002},
                    "momentum_baseline": {
                        "available": False,
                        "momentum_score_column": "momentum_10d",
                    },
                    "relative_momentum_baseline": {
                        "available": False,
                        "relative_momentum_score_column": "relative_momentum_10d",
                    },
                    "universe": {"average_basket_excess_return": 0.001},
                    "benchmark": {"average_basket_raw_return": 0.003},
                }
            },
        }

    monkeypatch.setattr(trainer, "train_models", fake_train_models)

    trainer.main(["--target-mode", "cross_sectional_top_bottom"])

    assert trained_target_modes == ["cross_sectional_top_bottom"]


def test_main_walk_forward_runs_walk_forward_path(monkeypatch, capsys):
    prepare_calls = []
    walk_forward_calls = []

    def fake_prepare_data_parallel(tickers, period="5y", use_cache=True):
        prepare_calls.append((list(tickers), period, use_cache))
        return pd.DataFrame({"Close": [1.0]})

    def fake_run_walk_forward_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        min_train_years=5,
        validation_years=1,
        test_years=1,
        target_mode="excess_return",
    ):
        walk_forward_calls.append(
            {
                "prediction_days": prediction_days,
                "random_trials": random_trials,
                "random_trial_workers": random_trial_workers,
                "min_train_years": min_train_years,
                "validation_years": validation_years,
                "test_years": test_years,
                "target_mode": target_mode,
            }
        )
        return {
            "prediction_days": prediction_days,
            "folds": [
                {
                    "fold_index": 0,
                    "train_date_range": {
                        "start": "2015-01-01",
                        "end": "2019-12-31",
                    },
                    "validation_date_range": {
                        "start": "2020-01-01",
                        "end": "2020-12-31",
                    },
                    "test_date_range": {
                        "start": "2021-01-01",
                        "end": "2021-12-31",
                    },
                    "selected_candidate_name": "candidate_0",
                    "validation_selection_score": 0.01,
                    "top_n": {
                        "top_5": {
                            "model_excess": 0.02,
                            "model_minus_momentum": 0.01,
                            "model_minus_universe": 0.03,
                        }
                    },
                }
            ],
            "aggregate": {
                "fold_count": 1,
                "selected_candidate_counts": {"candidate_0": 1},
                "top_n": {
                    "top_5": {
                        "average_model_excess": 0.02,
                        "average_model_minus_momentum": 0.01,
                        "average_model_minus_universe": 0.03,
                        "fold_win_rate_vs_momentum": 1.0,
                        "fold_win_rate_vs_universe": 1.0,
                    }
                },
            },
        }

    monkeypatch.setattr(trainer, "prepare_data_parallel", fake_prepare_data_parallel)
    monkeypatch.setattr(
        trainer,
        "run_walk_forward_models",
        fake_run_walk_forward_models,
    )
    monkeypatch.setattr(
        trainer,
        "train_models",
        lambda *args, **kwargs: pytest.fail("normal training path should not run"),
    )

    report = trainer.main(
        [
            "--walk-forward",
            "--prediction-days",
            "10",
            "--period",
            "10y",
            "--random-trials",
            "20",
            "--random-trial-workers",
            "2",
            "--walk-forward-min-train-years",
            "4",
        ]
    )

    assert prepare_calls == [
        (get_training_tickers(DEFAULT_TRAINING_UNIVERSE), "10y", True)
    ]
    assert walk_forward_calls == [
        {
            "prediction_days": 10,
            "random_trials": 20,
            "random_trial_workers": 2,
            "min_train_years": 4,
            "validation_years": 1,
            "test_years": 1,
            "target_mode": "excess_return",
        }
    ]
    assert report["aggregate"]["fold_count"] == 1
    output = capsys.readouterr().out
    assert "Walk-Forward Top-N Summary:" in output
    assert "top_5:" in output
    assert "XGBoost Top-N Basket Backtest Summary:" not in output


def test_main_walk_forward_forwards_rank_ndcg_target_mode(monkeypatch):
    walk_forward_target_modes = []

    monkeypatch.setattr(
        trainer,
        "prepare_data_parallel",
        lambda *args, **kwargs: pd.DataFrame({"Close": [1.0]}),
    )

    def fake_run_walk_forward_models(
        data,
        prediction_days,
        random_trials=100,
        random_trial_workers=4,
        min_train_years=5,
        validation_years=1,
        test_years=1,
        target_mode="excess_return",
    ):
        walk_forward_target_modes.append(target_mode)
        return {
            "prediction_days": prediction_days,
            "target_mode": target_mode,
            "folds": [],
            "aggregate": {"fold_count": 0, "top_n": {}},
        }

    monkeypatch.setattr(
        trainer,
        "run_walk_forward_models",
        fake_run_walk_forward_models,
    )
    monkeypatch.setattr(
        trainer,
        "train_models",
        lambda *args, **kwargs: pytest.fail("normal training path should not run"),
    )

    report = trainer.main(
        [
            "--walk-forward",
            "--target-mode",
            "cross_sectional_rank_ndcg",
            "--prediction-days",
            "10",
        ]
    )

    assert walk_forward_target_modes == ["cross_sectional_rank_ndcg"]
    assert report["target_mode"] == "cross_sectional_rank_ndcg"
