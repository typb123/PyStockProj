"""Model-training pipelines, fitting, selection, and input preparation."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

from src.config import (
    PREDICTION_DAYS,
    TEST_SIZE,
    TRAINING_UNIVERSES,
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
)
from src.data.data_prep import DataPreparator
from src.data.data_fetch import get_yfinance_data_provenance
from src.data.training_data import validate_input_data
from src.train.artifacts import (
    build_model_metadata,
    save_cross_sectional_rank_ndcg_artifacts,
    save_cross_sectional_top_bottom_artifacts,
    save_excess_return_model_artifacts,
)
from src.train.evaluation import build_model_only_top_n_basket_backtest_report
from src.train.reporting import (
    evaluate_model,
    log_feature_importances,
    log_rank_ndcg_test_report,
    log_ranking_classifier_test_report,
    log_xgboost_test_report,
)
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    TARGET_MODE_EXCESS_RETURN,
    resolve_model_feature_columns,
    validate_target_mode,
)


VALIDATION_TOP_N_SELECTION_VALUES = (5, 10, 20)
XG_PARAMS_RANKER = {
    "objective": "rank:ndcg",
    "eval_metric": ["ndcg@5", "ndcg@10"],
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": 42,
}


def _build_training_population(
    data: pd.DataFrame,
    *,
    universe_name: str | None,
    benchmark_ticker: str = "SPY",
) -> dict:
    """Record the ordered tickers actually present in one training input frame."""
    if "Ticker" not in data.columns:
        raise ValueError("Training data must include Ticker for artifact provenance.")
    if universe_name is not None and (
        not isinstance(universe_name, str) or not universe_name
    ):
        raise ValueError("training_universe_name must be a non-empty string or None.")
    if universe_name is not None and universe_name not in TRAINING_UNIVERSES:
        raise ValueError(
            "training_universe_name must identify a configured training universe."
        )

    ordered_tickers = []
    seen = set()
    for raw_ticker in data["Ticker"]:
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        ordered_tickers.append(ticker)

    if benchmark_ticker not in seen:
        raise ValueError(
            f"Training data must include benchmark ticker {benchmark_ticker}."
        )
    candidate_tickers = [
        ticker for ticker in ordered_tickers if ticker != benchmark_ticker
    ]
    if not candidate_tickers:
        raise ValueError("Training data must include at least one candidate ticker.")

    return {
        "universe_name": universe_name,
        "candidate_tickers": candidate_tickers,
        "candidate_ticker_count": len(candidate_tickers),
        "benchmark_ticker": benchmark_ticker,
    }


def _finite_report_float(value):
    """Return a finite float for scalar report values, otherwise None."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _candidate_params_with_overrides(base_params, overrides):
    """Return XGBoost params with conservative candidate overrides applied."""
    candidate_params = dict(base_params)
    candidate_params.update(overrides)
    return candidate_params


def build_xgboost_regressor_candidate_configs(base_params=None):
    """Build the small validation-selection candidate set for the regressor."""
    base_params = dict(XG_PARAMS_REGRESSOR if base_params is None else base_params)
    base_max_depth = int(base_params.get("max_depth", 5))
    base_min_child_weight = float(base_params.get("min_child_weight", 5))
    base_gamma = float(base_params.get("gamma", 0.05))
    base_reg_lambda = float(base_params.get("reg_lambda", 5.0))

    candidate_specs = [
        ("candidate_0_baseline", {}),
        (
            "candidate_1_shallower_more_regularized",
            {
                "max_depth": max(2, base_max_depth - 1),
                "min_child_weight": base_min_child_weight + 2,
                "gamma": base_gamma * 1.5,
                "reg_lambda": base_reg_lambda * 1.5,
            },
        ),
        (
            "candidate_2_slightly_deeper_less_regularized",
            {
                "max_depth": base_max_depth + 1,
                "min_child_weight": max(1, base_min_child_weight - 2),
                "gamma": base_gamma * 0.7,
                "reg_lambda": base_reg_lambda * 0.7,
            },
        ),
    ]

    candidates = []
    seen_param_sets = set()
    for candidate_id, (candidate_name, overrides) in enumerate(candidate_specs):
        params = _candidate_params_with_overrides(base_params, overrides)
        param_key = tuple(sorted(params.items()))
        if param_key in seen_param_sets:
            continue
        seen_param_sets.add(param_key)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "params": params,
            }
        )

    return candidates


def _validation_top_n_model_excess_returns(
    validation_basket_report,
    top_n_values=VALIDATION_TOP_N_SELECTION_VALUES,
):
    """Extract bucket-level validation model basket excess returns."""
    bucket_excess_returns = {}
    usable_values = []

    for top_n in top_n_values:
        bucket_name = f"top_{top_n}"
        excess_return = _finite_report_float(
            validation_basket_report.get(bucket_name, {})
            .get("model", {})
            .get("average_basket_excess_return")
        )
        bucket_excess_returns[bucket_name] = excess_return
        if excess_return is not None:
            usable_values.append(excess_return)

    return bucket_excess_returns, usable_values


def _validation_top_n_mean_excess_return(
    validation_basket_report,
    top_n_values=VALIDATION_TOP_N_SELECTION_VALUES,
):
    """Mean available validation model basket excess across Top-N buckets."""
    bucket_excess_returns, usable_values = _validation_top_n_model_excess_returns(
        validation_basket_report,
        top_n_values=top_n_values,
    )
    if not usable_values:
        return np.nan, bucket_excess_returns, 0

    return float(np.mean(usable_values)), bucket_excess_returns, len(usable_values)


def _is_better_validation_candidate(candidate_score, best_score):
    """Return whether candidate_score should replace best_score."""
    if candidate_score is None or pd.isna(candidate_score):
        return False
    if best_score is None or pd.isna(best_score):
        return True
    return float(candidate_score) > float(best_score)


def select_xgboost_regressor_by_validation_top_n(
    x_train,
    y_train,
    x_val,
    y_val,
    validation_split_metadata,
    candidate_configs=None,
    top_n_values=VALIDATION_TOP_N_SELECTION_VALUES,
):
    """Train candidate regressors and select by validation Top-N basket excess."""
    candidate_configs = (
        build_xgboost_regressor_candidate_configs()
        if candidate_configs is None
        else list(candidate_configs)
    )
    if not candidate_configs:
        raise ValueError("At least one XGBoost regressor candidate is required.")

    candidate_reports = []
    selected_model = None
    selected_candidate_report = None
    best_score = None

    for candidate_order, candidate_config in enumerate(candidate_configs):
        params = dict(candidate_config["params"])
        candidate_id = candidate_config.get("candidate_id", candidate_order)
        candidate_name = candidate_config.get(
            "candidate_name",
            f"candidate_{candidate_id}",
        )

        candidate_model = XGBRegressor(**params)
        candidate_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        validation_predictions = candidate_model.predict(x_val)
        validation_basket_report = build_model_only_top_n_basket_backtest_report(
            validation_split_metadata,
            validation_predictions,
            top_n_values=top_n_values,
        )
        (
            validation_score,
            bucket_excess_returns,
            available_bucket_count,
        ) = _validation_top_n_mean_excess_return(
            validation_basket_report,
            top_n_values=top_n_values,
        )
        candidate_report = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "params": params,
            "validation_top_n_mean_excess_return": validation_score,
            "validation_top_n_basket_excess_returns": bucket_excess_returns,
            "available_bucket_count": available_bucket_count,
        }
        candidate_reports.append(candidate_report)

        if selected_model is None or _is_better_validation_candidate(
            validation_score,
            best_score,
        ):
            selected_model = candidate_model
            selected_candidate_report = candidate_report
            best_score = validation_score

    if selected_candidate_report is None:
        selected_candidate_report = candidate_reports[0]

    for candidate_report in candidate_reports:
        candidate_report["selected"] = candidate_report is selected_candidate_report

    selection_report = {
        "selection_metric": "validation_top_n_mean_excess_return",
        "selection_bucket_policy": (
            "Mean of available top_5/top_10/top_20 model "
            "average_basket_excess_return values; unavailable or NaN buckets "
            "are ignored."
        ),
        "top_n_values": list(top_n_values),
        "candidates": candidate_reports,
        "selected_candidate_id": selected_candidate_report["candidate_id"],
        "selected_candidate_name": selected_candidate_report["candidate_name"],
        "selected_params": selected_candidate_report["params"],
        "selected_validation_top_n_mean_excess_return": (
            selected_candidate_report["validation_top_n_mean_excess_return"]
        ),
    }

    return selected_model, selection_report


def cross_validate_model(x, y, model=None, cv=5):
    """
    Perform cross-validation on the given model and data.

    Parameters:
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Target variable.
        model: Scikit-learn model (default: LinearRegression).
        cv (int): Number of cross-validation folds.

    Returns:
        float: Average cross-validated MSE.
    """
    if model is None:
        model = LinearRegression()

    # Define custom scoring for negative MSE
    scores = cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=cv)
    if scores is None:
        raise ValueError(
            "Cross-validation scoring failed. Please check your model and data."
        )
    avg_mse = -np.mean(scores)
    logging.info(f"Cross-Validated MSE (cv={cv}): {avg_mse:.4f}")
    return avg_mse


def _require_split_metadata(split_metadata, split_name, target_mode):
    """Return split metadata with a clear error if the split is unavailable."""
    if split_name not in split_metadata:
        raise ValueError(
            f"target_mode={target_mode!r} requires "
            f"split_metadata[{split_name!r}]."
        )
    return split_metadata[split_name]


def _require_ranking_metadata(split_metadata, split_name):
    """Return split metadata after checking ranking-label columns are present."""
    required_columns = ["ranking_train_sample", "top_quintile_target"]
    metadata = _require_split_metadata(
        split_metadata,
        split_name,
        TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    )

    missing_columns = [
        column for column in required_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            "target_mode='cross_sectional_top_bottom' requires "
            f"split_metadata[{split_name!r}] columns: {missing_columns}"
        )
    return metadata


def _ranking_labeled_rows(features, metadata):
    """Select top/bottom ranking-label rows from a feature split."""
    ranking_sample_mask = (
        metadata["ranking_train_sample"].fillna(False).astype(bool).to_numpy()
    )
    target_values = metadata["top_quintile_target"]
    labeled_mask = ranking_sample_mask & target_values.notna().to_numpy()
    labels = target_values.loc[labeled_mask].astype(int).to_numpy()
    return features.iloc[labeled_mask], labels


def _validate_ranking_training_labels(labels):
    """Ensure the ranking classifier has a usable binary training target."""
    if len(labels) == 0:
        raise ValueError(
            "No valid ranking training rows found for "
            "target_mode='cross_sectional_top_bottom'."
        )

    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(
            "Ranking training sample must contain both classes for "
            "target_mode='cross_sectional_top_bottom'."
        )


def _require_rank_ndcg_metadata(split_metadata, split_name):
    """Return split metadata after checking rank-NDCG metadata columns."""
    required_columns = ["prediction_date", "excess_return_rank_pct_by_date"]
    metadata = _require_split_metadata(
        split_metadata,
        split_name,
        TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    )

    missing_columns = [
        column for column in required_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            "target_mode='cross_sectional_rank_ndcg' requires "
            f"split_metadata[{split_name!r}] columns: {missing_columns}"
        )
    return metadata


def _rank_percentile_to_ndcg_relevance(rank_pct):
    """Convert same-date realized rank percentile into graded relevance 0..4."""
    rank_pct = pd.Series(rank_pct, dtype=float)
    relevance = np.select(
        [
            rank_pct <= 0.20,
            rank_pct <= 0.40,
            rank_pct <= 0.60,
            rank_pct < 0.80,
            rank_pct >= 0.80,
        ],
        [0, 1, 2, 3, 4],
        default=np.nan,
    )
    return pd.Series(relevance, index=rank_pct.index).astype(int)


def _rank_ndcg_training_data(features, metadata, min_group_count=2):
    """Return date-sorted features, relevance labels, and qid for XGBRanker."""
    if len(features) != len(metadata):
        raise ValueError(
            "Rank-NDCG feature rows must align with split metadata rows: "
            f"features={len(features)}, metadata={len(metadata)}."
        )

    working_metadata = metadata.reset_index(drop=True).copy()
    working_metadata["_row_position"] = np.arange(len(working_metadata))
    rank_pct = working_metadata["excess_return_rank_pct_by_date"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    usable_mask = rank_pct.notna() & working_metadata["prediction_date"].notna()
    usable_metadata = working_metadata.loc[usable_mask].copy()
    if usable_metadata.empty:
        raise ValueError(
            "No usable ranker training rows found for "
            "target_mode='cross_sectional_rank_ndcg'."
        )

    usable_metadata["rank_ndcg_relevance"] = _rank_percentile_to_ndcg_relevance(
        usable_metadata["excess_return_rank_pct_by_date"]
    ).to_numpy()
    group_sizes = usable_metadata.groupby("prediction_date", sort=False).size()
    eligible_dates = group_sizes[group_sizes >= 2].index
    usable_metadata = usable_metadata[
        usable_metadata["prediction_date"].isin(eligible_dates)
    ].copy()
    if usable_metadata.empty:
        raise ValueError(
            "No usable ranker training rows remain after dropping prediction_date "
            "groups with fewer than 2 rows for target_mode='cross_sectional_rank_ndcg'."
        )

    usable_group_count = int(usable_metadata["prediction_date"].nunique())
    if usable_group_count < min_group_count:
        raise ValueError(
            f"Rank-NDCG training requires at least {min_group_count} usable "
            "prediction_date groups "
            "after filtering."
        )

    usable_metadata = usable_metadata.sort_values(
        ["prediction_date", "_row_position"],
        kind="mergesort",
    )
    ranked_features = features.iloc[
        usable_metadata["_row_position"].to_numpy()
    ].reset_index(drop=True)
    labels = usable_metadata["rank_ndcg_relevance"].astype(int).to_numpy()
    qid = pd.factorize(usable_metadata["prediction_date"], sort=False)[0]
    return ranked_features, labels, qid, usable_metadata.drop(columns=["_row_position"])


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
    training_population,
    data_provenance=None,
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
        training_population=training_population,
        data_provenance=data_provenance,
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

    saved_paths = save_cross_sectional_rank_ndcg_artifacts(
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
    training_population,
    data_provenance=None,
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
        training_population=training_population,
        data_provenance=data_provenance,
    )
    model_metadata["ranking_training_row_count"] = int(len(x_train_ranking))
    model_metadata["ranking_training_candidate_count"] = int(len(x_train_classifier))
    model_metadata["ranking_validation_scored_row_count"] = int(
        len(ranking_validation_scores)
    )
    model_metadata["ranking_test_scored_row_count"] = int(len(ranking_test_scores))
    model_metadata["model_artifact_type"] = "classifier_only"
    model_metadata["regressor_artifact"] = None

    saved_paths = save_cross_sectional_top_bottom_artifacts(
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
    training_universe_name: str | None = None,
    training_period: str | None = None,
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
    training_population = _build_training_population(
        data,
        universe_name=training_universe_name,
    )
    data_provenance = get_yfinance_data_provenance(training_period)
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
        "tenkan_sen_to_close",
        "kijun_sen_to_close",
        "senkou_span_a_to_close",
        "senkou_span_b_to_close",
        "chikou_lag_close_26_to_close",
        "chikou_return_26",
        "chikou_above_lag_26",
        "open_to_close",
        "rsi",
        "signal_line_to_close",
        "atr_to_close",
        "sma_20_to_close",
        "macd_to_close",
        "bb_std_to_close",
        "rolling_signed_volume_20d",
        "dailyReturn",
        "macd_histogram_to_close",
        "vma_20",
        "high_to_close",
        "low_to_close",
        "bb_middle_to_close",
        "bb_upper_to_close",
        "bb_lower_to_close",
        "stoch_k",
        "stoch_d",
        "Volume",
        "sma_10_to_close",
        "volatility",
        "vma_10",
        "sma_5_to_close",
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
            training_population,
            data_provenance=data_provenance,
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
            training_population,
            data_provenance=data_provenance,
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
        training_population=training_population,
        data_provenance=data_provenance,
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
    saved_paths = save_excess_return_model_artifacts(
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
