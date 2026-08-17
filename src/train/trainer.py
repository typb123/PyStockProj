"""Training entry point for SPY-relative stock prediction models.

The pipeline keeps Linear Regression as a separate baseline artifact, while
XGBoost trains a beat-benchmark classifier and an excess-return regressor.
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.data.data_prep import DataPreparator
from src.data.training_data import (
    PROJECT_ROOT,
    YFINANCE_CACHE_DIR,
    YFINANCE_CACHE_FORMAT,
    fetch_raw_ticker_data,
    fetch_tickers_data,
    get_yfinance_cache_path,
    load_cached_yfinance_data,
    normalize_raw_ohlcv_index,
    prepare_data_parallel,
    validate_input_data,
    write_yfinance_cache,
)
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    TARGET_MODE_EXCESS_RETURN,
    TARGET_MODES,
    WALK_FORWARD_TARGET_MODES,
    resolve_model_feature_columns,
    validate_target_mode,
    validate_walk_forward_target_mode,
)
from src.train.artifacts import (
    build_horizon_model_paths,
    build_model_metadata,
    build_rank_ndcg_model_paths,
    save_horizon_model_artifacts,
    save_model_artifacts,
    save_rank_ndcg_model_artifacts,
    save_ranking_model_artifacts,
)
from src.train.reporting import (
    build_walk_forward_report_for_split,
    evaluate_model,
    format_horizon_comparison_summary,
    format_same_date_ranking_diagnostics_summary,
    format_top_n_basket_backtest_by_year_summary,
    format_top_n_basket_backtest_summary,
    format_top_n_ranked_selection_summary,
    format_walk_forward_fold_summary,
    format_walk_forward_summary,
    format_xgboost_regressor_validation_selection_report,
    log_feature_importances,
    log_rank_ndcg_test_report,
    log_ranking_classifier_test_report,
    log_xgboost_test_report,
)
from src.train.model_training import (
    VALIDATION_TOP_N_SELECTION_VALUES,
    XG_PARAMS_RANKER,
    _rank_ndcg_training_data,
    _ranking_labeled_rows,
    _require_rank_ndcg_metadata,
    _require_ranking_metadata,
    _require_split_metadata,
    _validate_ranking_training_labels,
    build_xgboost_regressor_candidate_configs,
    cross_validate_model,
    select_xgboost_regressor_by_validation_top_n,
)
from src.train.evaluation import (
    build_top_n_selection_reports,
)
from xgboost import XGBClassifier, XGBRanker
from src.config import (
    DEFAULT_TRAINING_UNIVERSE,
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
    TRAINING_UNIVERSES,
    get_training_tickers,
    PREDICTION_DAYS,
    TEST_SIZE,
)
from src.train.evaluation.walk_forward import (
    DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
    DEFAULT_WALK_FORWARD_TEST_YEARS,
    DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
    build_expanding_yearly_walk_forward_folds,
    build_walk_forward_aggregate_summary,
    build_walk_forward_split,
    prepare_walk_forward_model_frame,
)


logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s: - %(levelname)s -%(message)s",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ALL_HORIZONS = [5, 10, 20, 50]


def train_cross_sectional_rank_ndcg_model(
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
    """Train grouped learning-to-rank model on same-date rank percentile labels."""
    split_metadata = prepared_data["split_metadata"]
    train_metadata = _require_rank_ndcg_metadata(split_metadata, "train")
    val_metadata = _require_rank_ndcg_metadata(split_metadata, "val")
    test_metadata = _require_rank_ndcg_metadata(split_metadata, "test")

    x_train_grouped, y_train_ranker, train_qid, train_grouped_metadata = (
        _rank_ndcg_training_data(x_train_ranker, train_metadata)
    )
    fit_kwargs = {"qid": train_qid, "verbose": False}
    try:
        x_val_grouped, y_val_ranker, val_qid, _ = _rank_ndcg_training_data(
            x_val_ranker,
            val_metadata,
            min_group_count=1,
        )
        fit_kwargs["eval_set"] = [(x_val_grouped, y_val_ranker)]
        fit_kwargs["eval_qid"] = [val_qid]
    except ValueError as error:
        logging.info(f"Rank-NDCG validation eval_set unavailable: {error}")

    logging.info(
        "Training XGBoost Cross-Sectional Rank-NDCG model on "
        f"{len(x_train_grouped)} rows across {len(np.unique(train_qid))} "
        f"prediction_date groups out of {len(x_train_ranker)} training candidates."
    )
    ranker_fit_params = dict(ranker_params or XG_PARAMS_RANKER)
    if "eval_set" not in fit_kwargs:
        ranker_fit_params.pop("early_stopping_rounds", None)
    ranker = XGBRanker(**ranker_fit_params)
    ranker.fit(x_train_grouped, y_train_ranker, **fit_kwargs)

    ranking_validation_scores = ranker.predict(x_val_ranker)
    ranking_test_scores = ranker.predict(x_test_ranker)

    xgboost_test_reports = log_rank_ndcg_test_report(
        test_metadata,
        ranking_test_scores,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )

    importances_ranker = ranker.feature_importances_
    feature_importance_ranker = pd.DataFrame(
        {"feature": classifier_features, "importance": importances_ranker}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances(
        "XGBoost Cross-Sectional Rank-NDCG",
        feature_importance_ranker,
    )

    model_metadata = build_model_metadata(
        linear_features,
        classifier_features,
        regressor_features=[],
        prediction_days=prediction_days,
        target_mode=TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    )
    model_metadata["ranker_training_row_count"] = int(len(x_train_grouped))
    model_metadata["ranker_training_candidate_count"] = int(len(x_train_ranker))
    model_metadata["ranker_training_group_count"] = int(len(np.unique(train_qid)))
    model_metadata["ranker_training_dropped_row_count"] = int(
        len(x_train_ranker) - len(x_train_grouped)
    )
    model_metadata["ranker_validation_scored_row_count"] = int(
        len(ranking_validation_scores)
    )
    model_metadata["ranker_test_scored_row_count"] = int(len(ranking_test_scores))
    model_metadata["ranker_training_prediction_dates"] = int(
        train_grouped_metadata["prediction_date"].nunique()
    )
    model_metadata["classifier_artifact"] = None
    model_metadata["regressor_artifact"] = None

    saved_paths = save_rank_ndcg_model_artifacts(
        prediction_days,
        data_preparator,
        all_features,
        model_metadata,
        ranker,
    )
    logging.info(
        f"Rank-NDCG training completed for prediction_days={prediction_days}. "
        f"Models saved to {Path(saved_paths['model_metadata']).parent}."
    )

    return {
        "prediction_days": prediction_days,
        "target_mode": TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
        "ranking_score_name": "xgboost_rank_ndcg_score",
        "model_metadata": model_metadata,
        "artifact_paths": saved_paths,
        "ranked_selection": xgboost_test_reports["ranked_selection"],
        "basket_backtest": xgboost_test_reports["basket_backtest"],
        "basket_backtest_by_year": xgboost_test_reports["basket_backtest_by_year"],
        "same_date_ranking_diagnostics": xgboost_test_reports[
            "same_date_ranking_diagnostics"
        ],
    }


def train_cross_sectional_ranking_model(
    data_preparator,
    prepared_data,
    all_features,
    linear_features,
    classifier_features,
    x_train_classifier,
    x_val_classifier,
    x_test_classifier,
    prediction_days,
    random_trials,
    random_trial_workers,
    classifier_params=None,
):
    """Train ranking-mode classifier on labeled top/bottom rows and score all rows."""
    split_metadata = prepared_data["split_metadata"]
    train_metadata = _require_ranking_metadata(split_metadata, "train")
    val_metadata = _require_ranking_metadata(split_metadata, "val")
    test_metadata = _require_split_metadata(
        split_metadata,
        "test",
        TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    )

    x_train_ranking, ranking_y_train = _ranking_labeled_rows(
        x_train_classifier,
        train_metadata,
    )
    _validate_ranking_training_labels(ranking_y_train)
    x_val_labeled, ranking_y_val = _ranking_labeled_rows(
        x_val_classifier,
        val_metadata,
    )

    logging.info(
        "Training XGBoost Cross-Sectional Ranking Classifier on "
        f"{len(x_train_ranking)} top/bottom rows out of {len(x_train_classifier)} "
        "training candidates."
    )
    classifier = XGBClassifier(**(classifier_params or XG_PARAMS_CLASSIFIER))
    fit_kwargs = {"verbose": False}
    if len(x_val_labeled) > 0:
        fit_kwargs["eval_set"] = [(x_val_labeled, ranking_y_val)]
    classifier.fit(x_train_ranking, ranking_y_train, **fit_kwargs)

    ranking_validation_scores = classifier.predict_proba(x_val_classifier)[:, 1]
    ranking_test_scores = classifier.predict_proba(x_test_classifier)[:, 1]

    xgboost_test_reports = log_ranking_classifier_test_report(
        test_metadata,
        ranking_test_scores,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )

    importances_clf = classifier.feature_importances_
    feature_importance_clf = pd.DataFrame(
        {"feature": classifier_features, "importance": importances_clf}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances(
        "XGBoost Cross-Sectional Ranking Classifier",
        feature_importance_clf,
    )

    model_metadata = build_model_metadata(
        linear_features,
        classifier_features,
        regressor_features=[],
        prediction_days=prediction_days,
        target_mode=TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    )
    model_metadata["ranking_training_row_count"] = int(len(x_train_ranking))
    model_metadata["ranking_training_candidate_count"] = int(len(x_train_classifier))
    model_metadata["ranking_validation_scored_row_count"] = int(
        len(ranking_validation_scores)
    )
    model_metadata["ranking_test_scored_row_count"] = int(len(ranking_test_scores))
    model_metadata["model_artifact_type"] = "classifier_only"
    model_metadata["regressor_artifact"] = None

    saved_paths = save_ranking_model_artifacts(
        prediction_days,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
    )
    logging.info(
        f"Ranking-mode training completed for prediction_days={prediction_days}. "
        f"Models saved to {Path(saved_paths['model_metadata']).parent}."
    )

    return {
        "prediction_days": prediction_days,
        "target_mode": TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
        "ranking_score_name": "probability_of_top_quintile_outperformance",
        "model_metadata": model_metadata,
        "artifact_paths": saved_paths,
        "ranked_selection": xgboost_test_reports["ranked_selection"],
        "basket_backtest": xgboost_test_reports["basket_backtest"],
        "basket_backtest_by_year": xgboost_test_reports["basket_backtest_by_year"],
        "same_date_ranking_diagnostics": xgboost_test_reports[
            "same_date_ranking_diagnostics"
        ],
    }


def train_models(
    data: pd.DataFrame,
    prediction_days: int = PREDICTION_DAYS,
    random_trials: int = 100,
    random_trial_workers: int = 4,
    target_mode: str = TARGET_MODE_EXCESS_RETURN,
    classifier_params: dict | None = None,
    regressor_params: dict | None = None,
    ranker_params: dict | None = None,
    feature_columns_override: list[str] | None = None,
) -> dict:
    """
    Train the linear baseline, beat-benchmark classifier, and excess-return regressor.

    Validation data is used for XGBoost eval_set and threshold research; test data
    is reserved for final diagnostics.
    """
    validate_target_mode(target_mode)
    model_feature_columns = resolve_model_feature_columns(feature_columns_override)
    logging.info(
        "Training models with explicitly defined feature arrays "
        f"for prediction_days={prediction_days}, target_mode={target_mode}."
    )

    # DataPreparator owns target creation, chronological splits, and shared scaling.
    data = validate_input_data(data)
    data_preparator = DataPreparator()
    prepared_data = data_preparator.prepare_for_train(
        data, prediction_days=prediction_days, test_size=TEST_SIZE
    )

    # Rebuild DataFrames so explicit feature lists preserve their trained column order.
    all_features = prepared_data["feature_names"]
    x_train_full = pd.DataFrame(prepared_data["x_train"], columns=all_features)
    x_val_full = pd.DataFrame(prepared_data["x_val"], columns=all_features)
    x_test_full = pd.DataFrame(prepared_data["x_test"], columns=all_features)
    y_train = prepared_data["y_train"]
    y_val = prepared_data["y_val"]
    y_test = prepared_data["y_test"]
    logging.info(
        "Prepared data summary: "
        f"x_train={len(x_train_full)}, x_val={len(x_val_full)}, "
        f"x_test={len(x_test_full)}, features={len(all_features)}"
    )

    # Feature lists are intentionally inline for now; prediction metadata mirrors them.
    linear_features = [
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_lag_close_26",
        "chikou_return_26",
        "chikou_above_lag_26",
        "Open",
        "Close",
        "rsi",
        "signalLine",
        "ATR",
        "20_day_avg",
        "macd",
        "BB_Std",
        "obv",
        "dailyReturn",
        "macdHistogram",
        "vma_20",
        "High",
        "Low",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "stoch_k",
        "stoch_d",
        "Volume",
        "10_day_avg",
        "volatility",
        "vma_10",
        "5_day_avg",
    ]
    classifier_features = list(model_feature_columns)
    regressor_features = list(model_feature_columns)

    classifier_feature_names = classifier_features
    regressor_feature_names = regressor_features

    x_train_classifier = x_train_full[classifier_features]
    x_val_classifier = x_val_full[classifier_features]
    x_test_classifier = x_test_full[classifier_features]

    if target_mode == TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG:
        return train_cross_sectional_rank_ndcg_model(
            data_preparator,
            prepared_data,
            all_features,
            linear_features,
            classifier_features,
            x_train_classifier,
            x_val_classifier,
            x_test_classifier,
            prediction_days,
            random_trials,
            random_trial_workers,
            ranker_params=ranker_params,
        )

    if target_mode == TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM:
        return train_cross_sectional_ranking_model(
            data_preparator,
            prepared_data,
            all_features,
            linear_features,
            classifier_features,
            x_train_classifier,
            x_val_classifier,
            x_test_classifier,
            prediction_days,
            random_trials,
            random_trial_workers,
            classifier_params=classifier_params,
        )

    # Linear Regression is a separate scaled baseline, not a stacked XGBoost feature.
    scaler_lr = StandardScaler()
    x_train_lr_scaled = scaler_lr.fit_transform(x_train_full[linear_features])
    x_test_lr_scaled = scaler_lr.transform(x_test_full[linear_features])

    logging.info("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(x_train_lr_scaled, y_train)
    evaluate_model(linear_model, x_test_lr_scaled, y_test, model_type="regression")

    importances_lr = np.abs(linear_model.coef_)
    feature_importance_lr = pd.DataFrame(
        {"feature": linear_features, "importance": importances_lr}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances("Linear Regression", feature_importance_lr)

    x_train_regressor = x_train_full[regressor_features]
    x_val_regressor = x_val_full[regressor_features]
    x_test_regressor = x_test_full[regressor_features]

    # The classifier predicts beat-SPY labels; the regressor predicts excess return.
    direction_y_train = prepared_data["direction_y_train"]
    direction_y_val = prepared_data["direction_y_val"]
    direction_y_test = prepared_data["direction_y_test"]

    logging.info("Training XGBoost Classifier...")
    classifier = XGBClassifier(**(classifier_params or XG_PARAMS_CLASSIFIER))
    classifier.fit(
        x_train_classifier,
        direction_y_train,
        eval_set=[(x_val_classifier, direction_y_val)],
        verbose=False,
    )
    evaluate_model(
        classifier, x_test_classifier, direction_y_test, model_type="classification"
    )

    importances_clf = classifier.feature_importances_
    feature_importance_clf = pd.DataFrame(
        {"feature": classifier_feature_names, "importance": importances_clf}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances("XGBoost Classifier", feature_importance_clf)

    logging.info("Training XGBoost Regressor...")
    regressor_candidate_configs = build_xgboost_regressor_candidate_configs(
        regressor_params or XG_PARAMS_REGRESSOR
    )
    (
        regressor,
        regressor_validation_selection_report,
    ) = select_xgboost_regressor_by_validation_top_n(
        x_train_regressor,
        y_train,
        x_val_regressor,
        y_val,
        prepared_data["split_metadata"]["val"],
        candidate_configs=regressor_candidate_configs,
    )
    evaluate_model(regressor, x_test_regressor, y_test, model_type="regression")
    # Validation probabilities choose the beat-benchmark threshold; test evaluates it once.
    xgboost_test_reports = log_xgboost_test_report(
        y_train,
        y_val,
        y_test,
        direction_y_test,
        prepared_data["split_metadata"]["test"],
        classifier.predict(x_test_classifier),
        classifier.predict_proba(x_val_classifier)[:, 1],
        classifier.predict_proba(x_test_classifier)[:, 1],
        regressor.predict(x_test_regressor),
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
        regressor_validation_selection_report=regressor_validation_selection_report,
    )

    importances_reg = regressor.feature_importances_
    feature_importance_reg = pd.DataFrame(
        {"feature": regressor_feature_names, "importance": importances_reg}
    ).sort_values(by="importance", ascending=False)
    log_feature_importances("XGBoost Regressor", feature_importance_reg)

    # Save all preprocessing and feature metadata needed to reproduce training inputs.
    model_metadata = build_model_metadata(
        linear_features,
        classifier_features,
        regressor_features,
        prediction_days,
        target_mode=target_mode,
    )
    model_metadata["regressor_validation_selection"] = (
        regressor_validation_selection_report
    )
    model_metadata["xgboost_regressor_selected_candidate_name"] = (
        regressor_validation_selection_report.get("selected_candidate_name")
    )
    model_metadata["xgboost_regressor_selected_params"] = (
        regressor_validation_selection_report.get("selected_params")
    )
    saved_paths = save_horizon_model_artifacts(
        prediction_days,
        linear_model,
        scaler_lr,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
        regressor,
    )
    logging.info(
        f"Training completed for prediction_days={prediction_days}. "
        f"Models saved to {Path(saved_paths['model_metadata']).parent}."
    )
    return {
        "prediction_days": prediction_days,
        "target_mode": target_mode,
        "model_metadata": model_metadata,
        "artifact_paths": saved_paths,
        "regressor_validation_selection": xgboost_test_reports.get(
            "regressor_validation_selection",
            {},
        ),
        "basket_backtest": xgboost_test_reports["basket_backtest"],
        "same_date_ranking_diagnostics": xgboost_test_reports[
            "same_date_ranking_diagnostics"
        ],
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
) -> dict:
    """Run expanding-window walk-forward Top-N diagnostics for one horizon."""
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

    fold_reports = []
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
        logging.info(format_walk_forward_fold_summary(fold_report))

    aggregate_summary = build_walk_forward_aggregate_summary(fold_reports)
    return {
        "prediction_days": int(prediction_days),
        "target_mode": target_mode,
        "folds": fold_reports,
        "aggregate": aggregate_summary,
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


def _positive_int(value, argument_name="value"):
    """Parse a command-line value as a positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must be a positive integer"
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{argument_name} must be a positive integer")
    return parsed


def _positive_prediction_days(value):
    """Parse the prediction horizon CLI argument."""
    return _positive_int(value, "prediction_days")


def _positive_random_trials(value):
    """Parse the random baseline trial-count CLI argument."""
    return _positive_int(value, "random_trials")


def _positive_random_trial_workers(value):
    """Parse the random baseline worker-count CLI argument."""
    return _positive_int(value, "random_trial_workers")


def _positive_year_count(value, argument_name="years"):
    """Parse a positive walk-forward year-count CLI argument."""
    return _positive_int(value, argument_name)


def parse_args(argv=None):
    """Parse trainer CLI options and resolve the requested prediction horizons."""
    parser = argparse.ArgumentParser(
        description="Train SPY-relative stock prediction models."
    )
    horizon_group = parser.add_mutually_exclusive_group()
    horizon_group.add_argument(
        "-d",
        "--prediction-days",
        type=_positive_prediction_days,
        default=None,
        help=f"Prediction horizon in trading days. Defaults to {PREDICTION_DAYS}.",
    )
    horizon_group.add_argument(
        "--all-horizons",
        action="store_true",
        help=f"Train research horizons {ALL_HORIZONS}.",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help='YFinance history period for raw OHLCV fetches. Defaults to "5y".',
    )
    parser.add_argument(
        "--universe",
        choices=sorted(TRAINING_UNIVERSES),
        default=DEFAULT_TRAINING_UNIVERSE,
        help=f"Training universe to fetch. Defaults to {DEFAULT_TRAINING_UNIVERSE}.",
    )
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default=TARGET_MODE_EXCESS_RETURN,
        help=(
            "Training target mode. Cross-sectional modes are accepted for "
            "experimental ranking research."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Bypass the raw YFinance OHLCV cache entirely: always fetch from "
            "YFinance and do not read or write cache files."
        ),
    )
    parser.add_argument(
        "--random-trials",
        type=_positive_random_trials,
        default=100,
        help=(
            "Controls random baseline trials for Top-N ranked-selection and "
            "basket-backtest reports."
        ),
    )
    parser.add_argument(
        "--random-trial-workers",
        type=_positive_random_trial_workers,
        default=8,
        help=(
            "Worker count for Top-N random baseline trials. Use 1 for sequential "
            "execution."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Run expanding-window yearly walk-forward evaluation for one horizon "
            "instead of normal training/artifact saving."
        ),
    )
    parser.add_argument(
        "--walk-forward-min-train-years",
        type=lambda value: _positive_year_count(value, "walk_forward_min_train_years"),
        default=DEFAULT_WALK_FORWARD_MIN_TRAIN_YEARS,
        help=("Minimum training-history window in years for walk-forward evaluation."),
    )
    parser.add_argument(
        "--walk-forward-validation-years",
        type=lambda value: _positive_year_count(value, "walk_forward_validation_years"),
        default=DEFAULT_WALK_FORWARD_VALIDATION_YEARS,
        help="Validation window in years for walk-forward evaluation.",
    )
    parser.add_argument(
        "--walk-forward-test-years",
        type=lambda value: _positive_year_count(value, "walk_forward_test_years"),
        default=DEFAULT_WALK_FORWARD_TEST_YEARS,
        help="Test window in years for walk-forward evaluation.",
    )

    args = parser.parse_args(argv)
    if args.walk_forward and args.all_horizons:
        parser.error("--walk-forward supports one prediction horizon at a time.")
    if args.walk_forward and args.target_mode not in WALK_FORWARD_TARGET_MODES:
        parser.error(
            "--walk-forward supports target modes: "
            f"{', '.join(WALK_FORWARD_TARGET_MODES)}."
        )

    if args.all_horizons:
        args.horizons = list(ALL_HORIZONS)
    else:
        args.prediction_days = args.prediction_days or PREDICTION_DAYS
        args.horizons = [args.prediction_days]

    return args


def main(argv=None):
    """Run the end-to-end training workflow from CLI arguments."""
    args = parse_args(argv)
    validate_target_mode(args.target_mode)
    use_cache = not args.no_cache
    training_tickers = get_training_tickers(args.universe)
    logging.info(f"Selected YFinance period: {args.period}")
    logging.info(f"Selected target mode: {args.target_mode}")
    logging.info(f"Raw YFinance OHLCV cache enabled: {use_cache}")
    logging.info(
        f"Selected training universe: {args.universe} "
        f"({len(training_tickers)} tickers): {training_tickers}"
    )
    data = prepare_data_parallel(
        training_tickers,
        period=args.period,
        use_cache=use_cache,
    )
    if data.empty:
        logging.error("No data fetched for training. Exiting...")
        return

    if args.walk_forward:
        prediction_days = args.horizons[0]
        walk_forward_report = run_walk_forward_models(
            data.copy(),
            prediction_days=prediction_days,
            random_trials=args.random_trials,
            random_trial_workers=args.random_trial_workers,
            min_train_years=args.walk_forward_min_train_years,
            validation_years=args.walk_forward_validation_years,
            test_years=args.walk_forward_test_years,
            target_mode=args.target_mode,
        )
        summary = format_walk_forward_summary(walk_forward_report)
        logging.info(summary)
        print(summary)
        return walk_forward_report

    horizon_reports = {}
    for prediction_days in args.horizons:
        logging.info(f"Starting model training for prediction_days={prediction_days}.")
        horizon_reports[prediction_days] = train_models(
            data.copy(),
            prediction_days=prediction_days,
            target_mode=args.target_mode,
            random_trials=args.random_trials,
            random_trial_workers=args.random_trial_workers,
        )
        logging.info(f"Model training completed for prediction_days={prediction_days}.")

    if args.all_horizons:
        summary = format_horizon_comparison_summary(horizon_reports)
        logging.info(summary)
        print(summary)
    else:
        prediction_days = args.horizons[0]
        summary = format_top_n_basket_backtest_summary(
            horizon_reports[prediction_days]["basket_backtest"]
        )
        print(summary)

    print("Model training completed. Check training.log for details.")


if __name__ == "__main__":
    main()
