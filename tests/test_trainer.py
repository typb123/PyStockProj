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
    get_training_tickers,
)
from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from src.train.rank_ndcg_feature_ablations import (
    FeatureAblationSpec,
    build_rank_ndcg_feature_ablation_specs,
    extract_ablation_result_row,
    format_ablation_table,
)
import src.data.training_data as training_data
import src.train.artifacts as artifacts
import src.train.model_training as model_training
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


def test_trainer_reexports_model_training_public_api():
    public_names = (
        "VALIDATION_TOP_N_SELECTION_VALUES",
        "XG_PARAMS_RANKER",
        "build_xgboost_regressor_candidate_configs",
        "select_xgboost_regressor_by_validation_top_n",
        "cross_validate_model",
        "train_cross_sectional_rank_ndcg_model",
        "train_cross_sectional_ranking_model",
        "train_models",
    )

    for name in public_names:
        assert getattr(trainer, name) is getattr(model_training, name)


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
