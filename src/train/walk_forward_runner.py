"""Walk-forward model execution and target-mode dispatch."""

import hashlib
import logging
import time

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.config import PREDICTION_DAYS, XG_PARAMS_REGRESSOR
from src.data.data_prep import DataPreparator
from src.data.training_data import validate_input_data
from src.train.evaluation import build_top_n_selection_reports
from src.train.evaluation.walk_forward import (
    DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
    DEFAULT_WALK_FORWARD_TEST_YEARS,
    DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
    build_expanding_yearly_walk_forward_folds,
    build_model_population_identity,
    build_walk_forward_aggregate_summary,
    build_walk_forward_fold_population_identity,
    build_walk_forward_split,
)
from src.train.model_training import (
    XG_PARAMS_RANKER,
    _rank_ndcg_training_data,
    _require_rank_ndcg_metadata,
    build_xgboost_regressor_candidate_configs,
    select_xgboost_regressor_by_validation_top_n,
)
from src.train.reporting import (
    build_walk_forward_report_for_split,
    format_walk_forward_fold_summary,
    log_rank_ndcg_test_report,
)
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_EXCESS_RETURN,
    resolve_model_feature_columns,
    validate_walk_forward_target_mode,
)


def prepare_walk_forward_model_frame(
    data,
    prediction_days=PREDICTION_DAYS,
    data_preparator=None,
):
    """Create the SPY-relative modeling frame used by yearly walk-forward folds."""
    data_preparator = data_preparator or DataPreparator()
    return data_preparator.prepare_model_frame(
        data,
        prediction_days=prediction_days,
    )


def _rank_population_identity(metadata, labels, qid):
    """Identify the ordered, effective population supplied to XGBRanker."""
    identity_columns = [
        column
        for column in ("_source_index", "Ticker", "prediction_date")
        if column in metadata.columns
    ]
    identity_frame = metadata[identity_columns].copy()
    row_hashes = pd.util.hash_pandas_object(
        identity_frame, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(row_hashes.tobytes())
    digest.update(np.asarray(labels).tobytes())
    digest.update(np.asarray(qid).tobytes())
    return {
        "row_count": int(len(metadata)),
        "query_count": int(len(np.unique(qid))),
        "identity_columns": identity_columns,
        "ordered_row_digest": digest.hexdigest(),
    }


def _assert_rank_context_fold_is_valid(split, fold):
    """Fail early if a frozen split cannot safely feed canonical Rank-NDCG."""
    for split_name, metadata_name, fold_range_name in (
        ("train", "train", "train_date_range"),
        ("validation", "val", "validation_date_range"),
        ("test", "test", None),
    ):
        metadata = _require_rank_ndcg_metadata(
            split["split_metadata"], metadata_name
        )
        if {"prediction_date", "Ticker"}.issubset(metadata.columns) and metadata.duplicated(
            ["prediction_date", "Ticker"]
        ).any():
            raise ValueError(
                f"Rank-NDCG context fold {fold['fold_index']} has duplicate "
                f"{split_name} prediction_date/Ticker candidates."
            )
        for endpoint_column in (
            "forward_end_date",
            "benchmark_forward_end_date",
        ):
            if endpoint_column not in metadata.columns:
                continue
            endpoint = pd.to_datetime(metadata[endpoint_column])
            if endpoint.isna().any():
                raise ValueError(
                    f"Rank-NDCG context fold {fold['fold_index']} has missing "
                    f"{endpoint_column} in {split_name}."
                )
            if fold_range_name is None:
                continue
            # The embargo removes tail *prediction* dates, while endpoints may
            # legitimately extend into that removed tail.  They may not cross
            # the original chronological partition boundary.
            split_end = pd.Timestamp(fold[fold_range_name]["end"])
            if (endpoint > split_end).any():
                raise ValueError(
                    f"Rank-NDCG context fold {fold['fold_index']} {split_name} "
                    f"{endpoint_column} crosses its embargoed chronological boundary."
                )

    for array_name in ("x_train", "x_val", "x_test"):
        if not np.isfinite(split[array_name]).all():
            raise ValueError(
                f"Rank-NDCG context fold {fold['fold_index']} has non-finite "
                f"values in {array_name}."
            )


def prepare_rank_ndcg_walk_forward_context(
    data: pd.DataFrame,
    prediction_days: int = PREDICTION_DAYS,
    min_train_years: int = DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
    validation_years: int = DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
    test_years: int = DEFAULT_WALK_FORWARD_TEST_YEARS,
    max_folds: int | None = None,
) -> dict:
    """Prepare immutable-in-practice Rank-NDCG state shared by feature variants.

    ``prepare_model_frame`` deliberately computes cross-sectional rank labels
    before full-contract eligibility filtering.  The context preserves that
    existing target policy by reusing its resulting full-contract frame.
    """
    started_at = time.perf_counter()
    data = validate_input_data(data)
    prepared_frame, feature_columns = prepare_walk_forward_model_frame(
        data, prediction_days=prediction_days
    )
    folds = build_expanding_yearly_walk_forward_folds(
        prepared_frame,
        min_train_years=min_train_years,
        validation_years=validation_years,
        test_years=test_years,
    )
    if not folds:
        raise ValueError(
            "Not enough prediction years to create walk-forward folds with the "
            "requested windows."
        )
    if max_folds is not None:
        max_folds = int(max_folds)
        if max_folds < 1:
            raise ValueError("max_folds must be at least 1 when supplied.")
        folds = folds[-max_folds:]

    prepared_folds = []
    fold_population_identities = []
    for fold in folds:
        # StandardScaler is feature-wise: fitting this full-contract scaler on
        # training rows is exactly equivalent to fitting it again on any subset.
        split = build_walk_forward_split(
            prepared_frame, fold, feature_columns, prediction_days=prediction_days
        )
        _assert_rank_context_fold_is_valid(split, fold)
        rank_populations = {}
        for split_name, metadata_name, min_group_count in (
            ("train", "train", 2),
            ("validation", "val", 1),
        ):
            features = pd.DataFrame(
                split["x_train" if split_name == "train" else "x_val"],
                columns=feature_columns,
            )
            grouped, labels, qid, metadata = _rank_ndcg_training_data(
                features,
                _require_rank_ndcg_metadata(
                    split["split_metadata"], metadata_name
                ),
                min_group_count=min_group_count,
            )
            if not np.isfinite(np.asarray(labels)).all():
                raise ValueError(
                    f"Rank-NDCG context fold {fold['fold_index']} has non-finite "
                    f"effective {split_name} labels."
                )
            if len(grouped) != len(labels) or len(labels) != len(qid):
                raise ValueError(
                    f"Rank-NDCG context fold {fold['fold_index']} has inconsistent "
                    f"effective {split_name} row/qid counts."
                )
            rank_populations[split_name] = {
                "features": grouped,
                "labels": np.asarray(labels),
                "qid": np.asarray(qid),
                "metadata": metadata,
                "identity": _rank_population_identity(metadata, labels, qid),
            }
        prepared_folds.append(
            {"fold": fold, "split": split, "rank_populations": rank_populations}
        )
        fold_population_identities.append(
            build_walk_forward_fold_population_identity(fold, split)
        )

    return {
        "context_kind": "rank_ndcg_walk_forward",
        "prediction_days": int(prediction_days),
        "feature_columns": list(feature_columns),
        "folds": prepared_folds,
        "max_folds": max_folds,
        "population_identity": {
            "eligibility_feature_columns": list(feature_columns),
            "embargo_prediction_days": int(prediction_days),
            "prepared": build_model_population_identity(prepared_frame),
            "folds": fold_population_identities,
        },
        "context_preparation_seconds": time.perf_counter() - started_at,
    }


def run_rank_ndcg_walk_forward_from_context(
    context: dict,
    *,
    feature_columns_override: list[str] | None = None,
    prediction_days: int | None = None,
    random_trials: int = 100,
    random_trial_workers: int = 8,
    model_seed: int | None = None,
    xgb_threads: int | None = None,
    target_mode: str = TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    max_folds: int | None = None,
) -> dict:
    """Fit and evaluate one Rank-NDCG feature subset from frozen fold state."""
    if context.get("context_kind") != "rank_ndcg_walk_forward":
        raise ValueError("Invalid Rank-NDCG walk-forward context.")
    if target_mode != TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG:
        raise ValueError("Rank-NDCG context only supports cross_sectional_rank_ndcg.")
    if prediction_days is not None and int(prediction_days) != context["prediction_days"]:
        raise ValueError("Rank-NDCG context cannot be reused for another horizon.")
    if max_folds is not None and int(max_folds) != context.get("max_folds"):
        raise ValueError(
            "Rank-NDCG context does not match requested max_folds."
        )
    feature_columns = (
        resolve_model_feature_columns(feature_columns_override)
        if feature_columns_override is not None
        else list(context["feature_columns"])
    )
    full_columns = list(context["feature_columns"])
    missing_columns = [column for column in feature_columns if column not in full_columns]
    if missing_columns:
        raise ValueError(
            "Rank-NDCG context does not contain requested feature columns: "
            f"{missing_columns}"
        )
    feature_positions = [full_columns.index(column) for column in feature_columns]
    fold_reports = []
    phase_timings = {"model_fit_seconds": 0.0, "prediction_seconds": 0.0, "evaluation_seconds": 0.0}
    for prepared_fold in context["folds"]:
        fold = prepared_fold["fold"]
        full_split = prepared_fold["split"]
        split = dict(full_split)
        split["x_train"] = full_split["x_train"][:, feature_positions]
        split["x_val"] = full_split["x_val"][:, feature_positions]
        split["x_test"] = full_split["x_test"][:, feature_positions]
        selection_report, basket_backtest_report = _run_rank_ndcg_walk_forward_fold(
            split,
            feature_columns,
            prediction_days=context["prediction_days"],
            random_trials=random_trials,
            random_trial_workers=random_trial_workers,
            model_seed=model_seed,
            xgb_threads=xgb_threads,
            prepared_rank_populations=prepared_fold["rank_populations"],
        )
        for name, value in selection_report.pop("phase_timings", {}).items():
            phase_timings[name] += value
        fold_reports.append(
            build_walk_forward_report_for_split(
                fold, split, selection_report, basket_backtest_report
            )
        )
    return {
        "prediction_days": int(context["prediction_days"]),
        "target_mode": TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
        "model_seed": None if model_seed is None else int(model_seed),
        "folds": fold_reports,
        "aggregate": build_walk_forward_aggregate_summary(fold_reports),
        "population_identity": context["population_identity"],
        "rank_population_identity": [
            {
                "fold_index": item["fold"]["fold_index"],
                "train": item["rank_populations"]["train"]["identity"],
                "validation": item["rank_populations"]["validation"]["identity"],
            }
            for item in context["folds"]
        ],
        "phase_timings": {
            "context_preparation_seconds": context["context_preparation_seconds"],
            **phase_timings,
        },
    }


def run_walk_forward_models(
    data: pd.DataFrame,
    prediction_days: int = PREDICTION_DAYS,
    random_trials: int = 100,
    random_trial_workers: int = 8,
    min_train_years: int = DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
    validation_years: int = DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
    test_years: int = DEFAULT_WALK_FORWARD_TEST_YEARS,
    regressor_params: dict | None = None,
    target_mode: str = TARGET_MODE_EXCESS_RETURN,
    feature_columns_override: list[str] | None = None,
    model_seed: int | None = None,
    xgb_threads: int | None = None,
    max_folds: int | None = None,
    prepared_context: dict | None = None,
) -> dict:
    """Run expanding-window walk-forward Top-N diagnostics for one horizon.

    Model feature overrides are applied only after ``prepare_model_frame`` has
    filtered rows with the full current feature contract.  This keeps all
    feature-ablation variants on the same eligible population.
    """
    validate_walk_forward_target_mode(target_mode)
    requested_feature_columns = (
        resolve_model_feature_columns(feature_columns_override)
        if feature_columns_override is not None
        else None
    )
    if prepared_context is not None:
        if target_mode != TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG:
            raise ValueError(
                "prepared_context is only supported for cross_sectional_rank_ndcg."
            )
        return run_rank_ndcg_walk_forward_from_context(
            prepared_context,
            feature_columns_override=requested_feature_columns,
            prediction_days=prediction_days,
            random_trials=random_trials,
            random_trial_workers=random_trial_workers,
            model_seed=model_seed,
            xgb_threads=xgb_threads,
            max_folds=max_folds,
        )
    logging.info(
        "Running walk-forward evaluation "
        f"for prediction_days={prediction_days}, "
        f"target_mode={target_mode}, "
        f"min_train_years={min_train_years}, validation_years={validation_years}, "
        f"test_years={test_years}."
    )
    data = validate_input_data(data)
    prepared_frame, prepared_feature_columns = prepare_walk_forward_model_frame(
        data,
        prediction_days=prediction_days,
    )
    feature_columns = (
        requested_feature_columns
        if requested_feature_columns is not None
        else prepared_feature_columns
    )
    folds = build_expanding_yearly_walk_forward_folds(
        prepared_frame,
        min_train_years=min_train_years,
        validation_years=validation_years,
        test_years=test_years,
    )
    if not folds:
        raise ValueError(
            "Not enough prediction years to create walk-forward folds with the "
            "requested windows."
        )
    if max_folds is not None:
        max_folds = int(max_folds)
        if max_folds < 1:
            raise ValueError("max_folds must be at least 1 when supplied.")
        # Later folds have the largest expanding training window, making them a
        # useful representative subset for benchmark-only executions.
        folds = folds[-max_folds:]

    fold_reports = []
    fold_population_identities = []
    regressor_candidate_configs = (
        build_xgboost_regressor_candidate_configs(
            regressor_params or XG_PARAMS_REGRESSOR
        )
        if target_mode == TARGET_MODE_EXCESS_RETURN
        else None
    )
    for fold in folds:
        logging.info(
            "Starting walk-forward fold "
            f"{fold['fold_index']}: train={fold['train_date_range']}, "
            f"validation={fold['validation_date_range']}, "
            f"test={fold['test_date_range']}."
        )
        split = build_walk_forward_split(
            prepared_frame,
            fold,
            feature_columns,
            prediction_days=prediction_days,
        )
        if target_mode == TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG:
            selection_report, basket_backtest_report = _run_rank_ndcg_walk_forward_fold(
                split,
                feature_columns,
                prediction_days=prediction_days,
                random_trials=random_trials,
                random_trial_workers=random_trial_workers,
                model_seed=model_seed,
                xgb_threads=xgb_threads,
            )
        else:
            selection_report, basket_backtest_report = (
                _run_excess_return_walk_forward_fold(
                    split,
                    feature_columns,
                    regressor_candidate_configs,
                    prediction_days=prediction_days,
                    random_trials=random_trials,
                    random_trial_workers=random_trial_workers,
                )
            )
        fold_report = build_walk_forward_report_for_split(
            fold,
            split,
            selection_report,
            basket_backtest_report,
        )
        fold_reports.append(fold_report)
        fold_population_identities.append(
            build_walk_forward_fold_population_identity(fold, split)
        )
        logging.info(format_walk_forward_fold_summary(fold_report))

    aggregate_summary = build_walk_forward_aggregate_summary(fold_reports)
    return {
        "prediction_days": int(prediction_days),
        "target_mode": target_mode,
        "model_seed": None if model_seed is None else int(model_seed),
        "folds": fold_reports,
        "aggregate": aggregate_summary,
        "population_identity": {
            "eligibility_feature_columns": list(prepared_feature_columns),
            "embargo_prediction_days": int(prediction_days),
            "prepared": build_model_population_identity(prepared_frame),
            "folds": fold_population_identities,
        },
    }


def _run_excess_return_walk_forward_fold(
    split,
    feature_columns,
    regressor_candidate_configs,
    prediction_days=PREDICTION_DAYS,
    random_trials=100,
    random_trial_workers=8,
):
    """Train and evaluate the existing excess-return regressor walk-forward fold."""
    x_train = pd.DataFrame(split["x_train"], columns=feature_columns)
    x_val = pd.DataFrame(split["x_val"], columns=feature_columns)
    x_test = pd.DataFrame(split["x_test"], columns=feature_columns)
    selected_regressor, selection_report = select_xgboost_regressor_by_validation_top_n(
        x_train,
        split["y_train"],
        x_val,
        split["y_val"],
        split["split_metadata"]["val"],
        candidate_configs=regressor_candidate_configs,
    )
    test_predictions = selected_regressor.predict(x_test)
    top_n_reports = build_top_n_selection_reports(
        split["split_metadata"]["test"],
        test_predictions,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    return selection_report, top_n_reports["basket_backtest"]


def _run_rank_ndcg_walk_forward_fold(
    split,
    feature_columns,
    prediction_days=PREDICTION_DAYS,
    random_trials=100,
    random_trial_workers=8,
    model_seed=None,
    xgb_threads=None,
    prepared_rank_populations=None,
):
    """Train and evaluate one grouped Rank-NDCG walk-forward fold."""
    train_metadata = _require_rank_ndcg_metadata(split["split_metadata"], "train")
    val_metadata = _require_rank_ndcg_metadata(split["split_metadata"], "val")
    test_metadata = _require_rank_ndcg_metadata(split["split_metadata"], "test")
    x_train = pd.DataFrame(split["x_train"], columns=feature_columns)
    x_val = pd.DataFrame(split["x_val"], columns=feature_columns)
    x_test = pd.DataFrame(split["x_test"], columns=feature_columns)

    if prepared_rank_populations is None:
        x_train_grouped, y_train_ranker, train_qid, _ = _rank_ndcg_training_data(
            x_train,
            train_metadata,
        )
    else:
        train_population = prepared_rank_populations["train"]
        x_train_grouped = train_population["features"].loc[:, feature_columns]
        y_train_ranker = train_population["labels"]
        train_qid = train_population["qid"]
    fit_kwargs = {"qid": train_qid, "verbose": False}
    if prepared_rank_populations is not None:
        val_population = prepared_rank_populations["validation"]
        x_val_grouped = val_population["features"].loc[:, feature_columns]
        y_val_ranker = val_population["labels"]
        val_qid = val_population["qid"]
        fit_kwargs["eval_set"] = [(x_val_grouped, y_val_ranker)]
        fit_kwargs["eval_qid"] = [val_qid]
    else:
        try:
            x_val_grouped, y_val_ranker, val_qid, _ = _rank_ndcg_training_data(
                x_val,
                val_metadata,
                min_group_count=1,
            )
            fit_kwargs["eval_set"] = [(x_val_grouped, y_val_ranker)]
            fit_kwargs["eval_qid"] = [val_qid]
        except ValueError as error:
            logging.info(f"Rank-NDCG validation eval_set unavailable: {error}")

    ranker_params = dict(XG_PARAMS_RANKER)
    if model_seed is not None:
        ranker_params["random_state"] = int(model_seed)
    if xgb_threads is not None:
        xgb_threads = int(xgb_threads)
        if xgb_threads < 1:
            raise ValueError("xgb_threads must be at least 1 when supplied.")
        ranker_params["n_jobs"] = xgb_threads
    if "eval_set" not in fit_kwargs:
        ranker_params.pop("early_stopping_rounds", None)
    ranker = XGBRanker(**ranker_params)
    fit_started_at = time.perf_counter()
    ranker.fit(x_train_grouped, y_train_ranker, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_started_at
    prediction_started_at = time.perf_counter()
    ranking_test_scores = ranker.predict(x_test)
    prediction_seconds = time.perf_counter() - prediction_started_at
    evaluation_started_at = time.perf_counter()
    rank_ndcg_reports = log_rank_ndcg_test_report(
        test_metadata,
        ranking_test_scores,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started_at
    selection_report = {
        "selected_candidate_id": None,
        "selected_candidate_name": "rank_ndcg",
        "selected_validation_top_n_mean_excess_return": None,
        "ranker_n_jobs": ranker_params["n_jobs"],
        "phase_timings": {
            "model_fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
    }
    return selection_report, rank_ndcg_reports["basket_backtest"]
