"""Focused tests for Rank-NDCG feature-ablation comparison semantics."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.data_prep import DataPreparator
from src.features.feature_contract import MODEL_FEATURE_COLUMNS
from src.train.evaluation.walk_forward import build_walk_forward_split
from src.train.rank_ndcg_feature_ablations import (
    FeatureAblationSpec,
    assert_matching_ablation_population,
    build_benchmark_parallelism_configs,
    build_paired_ablation_report,
    build_output_path,
    build_rank_ndcg_feature_ablation_specs,
    run_rank_ndcg_ablation_specs,
    validate_ablation_parallelism,
)
import src.train.model_training as model_training
import src.train.rank_ndcg_feature_ablations as rank_ndcg_feature_ablations
import src.train.walk_forward_runner as walk_forward_runner


def _raw_feature_frame():
    """Build enough raw rows to exercise full-contract filtering and folds."""
    dates = [
        pd.Timestamp(f"{year}-01-{day:02d}")
        for year in range(2019, 2024)
        for day in (1, 2)
    ]
    rows = []
    for ticker_position, ticker in enumerate(("SPY", "AAA", "BBB")):
        for date_position, prediction_date in enumerate(dates):
            row = {
                "Ticker": ticker,
                "prediction_date": prediction_date,
                "Close": 100.0 + ticker_position + date_position,
            }
            row.update(
                {
                    feature: 1.0 + ticker_position + date_position / 100.0
                    for feature in MODEL_FEATURE_COLUMNS
                }
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    # This row would re-enter if a dropped feature changed eligibility filtering.
    frame.loc[
        (frame["Ticker"] == "AAA")
        & (frame["prediction_date"] == pd.Timestamp("2019-01-01")),
        "vma_20",
    ] = np.nan
    return frame


def _fold(years=(2019, 2020), validation_year=2021, test_year=2022):
    return {
        "fold_index": 0,
        "train_years": list(years),
        "validation_years": [validation_year],
        "test_years": [test_year],
        "train_date_range": {"start": None, "end": None},
        "validation_date_range": {"start": None, "end": None},
        "test_date_range": {"start": None, "end": None},
    }


def _top_n_bucket(model_excess):
    return {
        "model_excess": model_excess,
        "model_minus_momentum": model_excess - 0.01,
        "model_minus_universe": model_excess - 0.02,
    }


def _paired_report_fixture(model_excess_by_fold):
    folds = [
        {
            "fold_index": fold_index,
            "top_n": {
                f"top_{top_n}": _top_n_bucket(model_excess)
                for top_n in (5, 10, 20)
            },
        }
        for fold_index, model_excess in enumerate(model_excess_by_fold)
    ]
    average = float(np.mean(model_excess_by_fold))
    aggregate_bucket = {
        "average_model_excess": average,
        "average_model_minus_momentum": average - 0.01,
        "fold_win_rate_vs_momentum": 0.5,
        "average_model_minus_universe": average - 0.02,
        "fold_win_rate_vs_universe": 0.5,
    }
    return {
        "folds": folds,
        "aggregate": {
            "fold_count": len(folds),
            "top_n": {
                f"top_{top_n}": dict(aggregate_bucket) for top_n in (5, 10, 20)
            },
        },
        "population_identity": {
            "eligibility_feature_columns": list(MODEL_FEATURE_COLUMNS),
            "embargo_prediction_days": 10,
            "prepared": {"row_count": 12, "row_digest": "prepared"},
            "folds": [{"fold_index": index} for index in range(len(folds))],
        },
    }


def test_leave_one_out_specs_have_one_baseline_and_one_ordered_removal_per_feature():
    specs = build_rank_ndcg_feature_ablation_specs(mode="leave-one-out")

    assert [spec.name for spec in specs].count("all_features") == 1
    assert len(specs) == len(MODEL_FEATURE_COLUMNS) + 1
    assert specs[0].feature_columns == MODEL_FEATURE_COLUMNS
    for feature, spec in zip(MODEL_FEATURE_COLUMNS, specs[1:]):
        assert spec.name == f"drop__{feature}"
        assert spec.removed_features == (feature,)
        assert feature not in spec.feature_columns
        assert spec.feature_columns == [
            candidate for candidate in MODEL_FEATURE_COLUMNS if candidate != feature
        ]


def test_leave_one_out_selection_only_runs_requested_feature_in_contract_order():
    specs = build_rank_ndcg_feature_ablation_specs(
        mode="leave-one-out",
        drop_features=["vma_20"],
    )

    assert [spec.name for spec in specs] == [
        "all_features",
        "drop__vma_20",
    ]
    assert specs[1].feature_columns == [
        feature
        for feature in MODEL_FEATURE_COLUMNS
        if feature != "vma_20"
    ]


def test_leave_one_out_selection_is_deterministic_even_when_requested_out_of_order():
    requested = ["vma_10", "vma_20"]
    specs = build_rank_ndcg_feature_ablation_specs(
        mode="leave-one-out",
        drop_features=requested,
    )

    expected = [
        feature for feature in MODEL_FEATURE_COLUMNS if feature in set(requested)
    ]
    assert [spec.removed_features[0] for spec in specs[1:]] == expected


def test_group_ablation_specs_remain_available_and_start_with_baseline():
    specs = build_rank_ndcg_feature_ablation_specs(mode="groups")

    assert specs[0].name == "all_features"
    assert any(spec.name == "drop_volume_scale" for spec in specs)


def test_cpu_parallelism_validation_accepts_budgeted_configs_and_rejects_oversubscription():
    assert validate_ablation_parallelism(3, 8, 24).outer_workers == 3
    assert [
        (config.outer_workers, config.xgb_threads)
        for config in build_benchmark_parallelism_configs(24)
    ] == [(1, 24), (2, 12), (3, 8), (4, 6), (6, 4)]

    with pytest.raises(ValueError, match="must not exceed cpu_budget"):
        validate_ablation_parallelism(3, 9, 24)


def test_outer_ablation_execution_is_ordered_and_disables_nested_random_workers(
    monkeypatch,
):
    calls = []
    blas_limit_calls = []
    pool_contexts = []

    class ReversedSynchronousProcessPool:
        def __init__(
            self,
            max_workers,
            mp_context=None,
            initializer=None,
            initargs=(),
        ):
            assert mp_context.get_start_method() == "spawn"
            pool_contexts.append(mp_context.get_start_method())
            self.initializer = initializer
            self.initargs = initargs

        def __enter__(self):
            self.initializer(*self.initargs)
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, worker, tasks):
            return [worker(task) for task in reversed(list(tasks))]

    class RecordingBlasLimit:
        def __init__(self, *, limits, user_api):
            blas_limit_calls.append((limits, user_api))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_walk_forward(data, **kwargs):
        calls.append(kwargs)
        return {"folds": [{"fold_index": 0}], "marker": kwargs["model_seed"]}

    monkeypatch.setattr(
        rank_ndcg_feature_ablations,
        "ProcessPoolExecutor",
        ReversedSynchronousProcessPool,
    )
    monkeypatch.setattr(
        rank_ndcg_feature_ablations,
        "run_walk_forward_models",
        fake_walk_forward,
    )
    monkeypatch.setattr(
        rank_ndcg_feature_ablations,
        "threadpool_limits",
        RecordingBlasLimit,
    )
    specs = [
        FeatureAblationSpec("all_features", ["momentum_5d"]),
        FeatureAblationSpec("drop__momentum_5d", ["momentum_10d"]),
    ]

    runs = run_rank_ndcg_ablation_specs(
        pd.DataFrame({"Close": [1.0]}),
        specs,
        outer_workers=2,
        xgb_threads=4,
        cpu_budget=8,
        walk_forward_kwargs={
            "target_mode": "cross_sectional_rank_ndcg",
            "random_trial_workers": 8,
            "model_seed": 137,
        },
    )

    assert [run.spec.name for run in runs] == [spec.name for spec in specs]
    assert [call["feature_columns_override"] for call in calls] == [
        specs[1].feature_columns,
        specs[0].feature_columns,
    ]
    assert all(call["random_trial_workers"] == 1 for call in calls)
    assert all(call["xgb_threads"] == 4 for call in calls)
    assert all(call["model_seed"] == 137 for call in calls)
    assert blas_limit_calls == [(1, "blas"), (1, "blas")]
    assert pool_contexts == ["spawn"]


def test_serial_ablation_execution_preserves_random_worker_setting(monkeypatch):
    calls = []
    specs = [FeatureAblationSpec("all_features", ["momentum_5d"])]

    def fake_walk_forward(data, **kwargs):
        calls.append(kwargs)
        return {"folds": []}

    monkeypatch.setattr(
        rank_ndcg_feature_ablations,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: pytest.fail("serial execution must not create a pool"),
    )

    runs = run_rank_ndcg_ablation_specs(
        pd.DataFrame({"Close": [1.0]}),
        specs,
        outer_workers=1,
        xgb_threads=8,
        cpu_budget=8,
        walk_forward_kwargs={"random_trial_workers": 3},
        serial_runner=fake_walk_forward,
    )

    assert [run.spec.name for run in runs] == ["all_features"]
    assert calls[0]["random_trial_workers"] == 3
    assert calls[0]["xgb_threads"] == 8


def test_outer_worker_initializer_inputs_and_tasks_are_spawn_pickleable():
    data = pd.DataFrame({"Close": [1.0]})
    spec = FeatureAblationSpec("all_features", ["momentum_5d"])
    task = (
        0,
        spec,
        {"model_seed": 137, "random_trial_workers": 1, "xgb_threads": 4},
        1,
        True,
    )

    assert pickle.loads(pickle.dumps(data)).equals(data)
    assert pickle.loads(pickle.dumps(task)) == task


def test_default_ablation_output_paths_distinguish_mode_seed_and_lofo_selection():
    common = {"prediction_days": 10, "period": "10y", "universe": "tiny"}
    group_path = build_output_path(**common, mode="groups")
    lofo_path = build_output_path(
        **common,
        mode="leave-one-out",
        drop_features=["vma_20"],
    )
    seeded_path = build_output_path(**common, mode="groups", model_seed=137)

    assert group_path != lofo_path
    assert group_path != seeded_path
    assert "leave-one-out_drop_vma_20" in lofo_path.name
    assert "seed_137" in seeded_path.name


def test_removed_feature_missing_rows_do_not_reenter_ablation_population():
    raw_data = _raw_feature_frame()
    prepared_frame, full_features = DataPreparator().prepare_model_frame(
        raw_data,
        prediction_days=1,
    )
    feature_without_vma = [
        feature
        for feature in full_features
        if feature != "vma_20"
    ]

    full_split = build_walk_forward_split(
        prepared_frame,
        _fold(),
        full_features,
        prediction_days=1,
    )
    ablation_split = build_walk_forward_split(
        prepared_frame,
        _fold(),
        feature_without_vma,
        prediction_days=1,
    )

    for split_name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(
            full_split["split_metadata"][split_name],
            ablation_split["split_metadata"][split_name],
        )
    assert not (
        (prepared_frame["Ticker"] == "AAA")
        & (prepared_frame["prediction_date"] == pd.Timestamp("2019-01-01"))
    ).any()


def test_rank_ndcg_effective_train_and_validation_populations_match_after_filtering():
    raw_data = _raw_feature_frame()
    prepared_frame, full_features = DataPreparator().prepare_model_frame(
        raw_data,
        prediction_days=1,
    )
    ablation_features = [
        feature
        for feature in full_features
        if feature != "vma_20"
    ]
    full_split = build_walk_forward_split(
        prepared_frame,
        _fold(),
        full_features,
        prediction_days=1,
    )
    ablation_split = build_walk_forward_split(
        prepared_frame,
        _fold(),
        ablation_features,
        prediction_days=1,
    )

    for split_name, min_group_count in (("train", 2), ("val", 1)):
        full_features_after_filter, full_labels, full_qid, full_metadata = (
            model_training._rank_ndcg_training_data(
                pd.DataFrame(
                    full_split[f"x_{split_name}"],
                    columns=full_features,
                ),
                full_split["split_metadata"][split_name],
                min_group_count=min_group_count,
            )
        )
        (
            ablation_features_after_filter,
            ablation_labels,
            ablation_qid,
            ablation_metadata,
        ) = model_training._rank_ndcg_training_data(
            pd.DataFrame(
                ablation_split[f"x_{split_name}"],
                columns=ablation_features,
            ),
            ablation_split["split_metadata"][split_name],
            min_group_count=min_group_count,
        )

        pd.testing.assert_frame_equal(full_metadata, ablation_metadata)
        np.testing.assert_array_equal(full_labels, ablation_labels)
        np.testing.assert_array_equal(full_qid, ablation_qid)
        assert len(full_features_after_filter) == len(ablation_features_after_filter)
        assert full_features_after_filter.shape[1] == len(full_features)
        assert ablation_features_after_filter.shape[1] == len(ablation_features)


def test_baseline_and_ablation_reports_have_identical_population_identity(
    monkeypatch,
):
    raw_data = _raw_feature_frame()
    full_features = list(MODEL_FEATURE_COLUMNS)
    feature_without_vma = [
        feature
        for feature in full_features
        if feature != "vma_20"
    ]

    def fake_ranker_fold(*args, **kwargs):
        basket_backtest = {
            f"top_{top_n}": {
                "model": {"average_basket_excess_return": 0.02},
                "momentum_baseline": {"average_basket_excess_return": 0.01},
                "universe": {"average_basket_excess_return": 0.00},
            }
            for top_n in (5, 10, 20)
        }
        return {
            "selected_candidate_id": None,
            "selected_candidate_name": "rank_ndcg",
            "selected_validation_top_n_mean_excess_return": None,
        }, basket_backtest

    monkeypatch.setattr(
        walk_forward_runner,
        "_run_rank_ndcg_walk_forward_fold",
        fake_ranker_fold,
    )

    report_kwargs = {
        "prediction_days": 1,
        "min_train_years": 2,
        "validation_years": 1,
        "test_years": 1,
        "random_trials": 1,
        "random_trial_workers": 1,
        "target_mode": "cross_sectional_rank_ndcg",
    }
    baseline_report = walk_forward_runner.run_walk_forward_models(
        raw_data,
        feature_columns_override=full_features,
        **report_kwargs,
    )
    ablation_report = walk_forward_runner.run_walk_forward_models(
        raw_data,
        feature_columns_override=feature_without_vma,
        **report_kwargs,
    )

    assert baseline_report["population_identity"] == ablation_report[
        "population_identity"
    ]
    assert_matching_ablation_population(baseline_report, ablation_report)


def test_walk_forward_forwards_explicit_model_seed_to_ranker_fold(monkeypatch):
    captured = {}
    fold = _fold()
    prepared_frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2021-01-01"])})

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
    monkeypatch.setattr(
        walk_forward_runner,
        "build_walk_forward_split",
        lambda *args, **kwargs: {"split_date_ranges": {}},
    )

    def fake_ranker_fold(split, feature_columns, *, model_seed=None, **kwargs):
        captured["model_seed"] = model_seed
        return {
            "selected_candidate_id": None,
            "selected_candidate_name": "rank_ndcg",
            "selected_validation_top_n_mean_excess_return": None,
        }, {}

    monkeypatch.setattr(
        walk_forward_runner,
        "_run_rank_ndcg_walk_forward_fold",
        fake_ranker_fold,
    )

    report = walk_forward_runner.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        target_mode="cross_sectional_rank_ndcg",
        model_seed=137,
    )

    assert captured["model_seed"] == 137
    assert report["model_seed"] == 137


def test_model_seed_does_not_override_regression_candidate_configuration(monkeypatch):
    captured = {}
    fold = _fold()
    prepared_frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2021-01-01"])})

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
    monkeypatch.setattr(
        walk_forward_runner,
        "build_walk_forward_split",
        lambda *args, **kwargs: {"split_date_ranges": {}},
    )

    def fake_build_configs(base_params):
        captured["regressor_params"] = dict(base_params)
        return []

    monkeypatch.setattr(
        walk_forward_runner,
        "build_xgboost_regressor_candidate_configs",
        fake_build_configs,
    )
    monkeypatch.setattr(
        walk_forward_runner,
        "_run_excess_return_walk_forward_fold",
        lambda *args, **kwargs: (
            {
                "selected_candidate_id": 0,
                "selected_candidate_name": "candidate_0",
                "selected_validation_top_n_mean_excess_return": 0.0,
            },
            {},
        ),
    )

    walk_forward_runner.run_walk_forward_models(
        pd.DataFrame({"Close": [1.0]}),
        model_seed=137,
    )

    assert captured["regressor_params"] == walk_forward_runner.XG_PARAMS_REGRESSOR


def test_paired_delta_uses_ablated_minus_baseline_with_fold_win_loss_tie_counts():
    baseline_report = _paired_report_fixture([0.10, 0.20, 0.30])
    ablation_report = _paired_report_fixture([0.20, 0.10, 0.30])

    paired = build_paired_ablation_report(
        baseline_report,
        ablation_report,
        ablation_name="drop__vma_20",
    )

    top_5 = paired["top_n"]["top_5"]
    assert paired["delta_convention"] == "ablated_minus_baseline"
    assert top_5["folds"][0]["delta"]["model_excess"] == 0.10
    assert top_5["folds"][1]["delta"]["model_excess"] == -0.10
    assert top_5["folds"][2]["delta"]["model_excess"] == 0.0
    assert top_5["fold_win_loss_tie"]["model_excess"] == {
        "ablation_win_count": 1,
        "baseline_win_count": 1,
        "tie_count": 1,
        "evaluated_fold_count": 3,
    }
