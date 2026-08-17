"""Reusable model fitting, selection, and ranking-data preparation helpers."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from src.config import XG_PARAMS_REGRESSOR
from src.train.evaluation import build_model_only_top_n_basket_backtest_report
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
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
