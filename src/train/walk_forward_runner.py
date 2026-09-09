"""Walk-forward model execution and target-mode dispatch."""

import logging

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
):
    """Train and evaluate one grouped Rank-NDCG walk-forward fold."""
    train_metadata = _require_rank_ndcg_metadata(split["split_metadata"], "train")
    val_metadata = _require_rank_ndcg_metadata(split["split_metadata"], "val")
    test_metadata = _require_rank_ndcg_metadata(split["split_metadata"], "test")
    x_train = pd.DataFrame(split["x_train"], columns=feature_columns)
    x_val = pd.DataFrame(split["x_val"], columns=feature_columns)
    x_test = pd.DataFrame(split["x_test"], columns=feature_columns)

    x_train_grouped, y_train_ranker, train_qid, _ = _rank_ndcg_training_data(
        x_train,
        train_metadata,
    )
    fit_kwargs = {"qid": train_qid, "verbose": False}
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
    ranker.fit(x_train_grouped, y_train_ranker, **fit_kwargs)
    ranking_test_scores = ranker.predict(x_test)
    rank_ndcg_reports = log_rank_ndcg_test_report(
        test_metadata,
        ranking_test_scores,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    selection_report = {
        "selected_candidate_id": None,
        "selected_candidate_name": "rank_ndcg",
        "selected_validation_top_n_mean_excess_return": None,
    }
    return selection_report, rank_ndcg_reports["basket_backtest"]
