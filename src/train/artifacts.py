"""Training artifact metadata, paths, and persistence."""

from pathlib import Path

import joblib

from src.config import MODEL_PATHS, PREDICTION_DAYS
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    TARGET_MODE_EXCESS_RETURN,
)

def build_model_metadata(
    linear_features,
    classifier_features,
    regressor_features,
    prediction_days,
    target_mode=TARGET_MODE_EXCESS_RETURN,
):
    """Build prediction-time metadata needed to align saved artifacts and features."""
    metadata = {
        "linear_features": linear_features,
        "classifier_features": classifier_features,
        "regressor_features": regressor_features,
        "classifier_feature_names": classifier_features,
        "regressor_feature_names": regressor_features,
        "prediction_days": prediction_days,
        "target_mode": target_mode,
        "target_type": "spy_relative_excess_forward_return",
        "benchmark_ticker": "SPY",
        "regressor_target": "targetReturns",
        "classifier_target": "beat_benchmark_target",
    }
    if target_mode == TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM:
        metadata.update(
            {
                "target_type": "cross_sectional_top_bottom_quintile",
                "classifier_target": "top_quintile_target",
                "regressor_target": None,
                "ranking_target": (
                    "top_20_vs_bottom_20_by_excess_forward_return"
                ),
                "ranking_group_key": "prediction_date",
                "ranking_training_sample": "ranking_train_sample",
                "primary_model_score": (
                    "probability_of_top_quintile_outperformance"
                ),
            }
        )
    if target_mode == TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG:
        metadata.update(
            {
                "target_type": "grouped_learning_to_rank_ndcg",
                "classifier_target": None,
                "regressor_target": None,
                "ranking_group_key": "prediction_date",
                "ranking_label_source": "excess_return_rank_pct_by_date",
                "ranking_label_type": "graded_0_to_4_from_rank_percentile",
                "primary_model_score": "xgboost_rank_ndcg_score",
                "model_artifact_type": "ranker",
            }
        )
    return metadata


def build_horizon_model_paths(prediction_days: int) -> dict:
    """Return artifact paths for one prediction horizon."""
    horizon_dir = Path("models") / f"horizon_{prediction_days}"
    return {
        "linear": str(horizon_dir / "linear_regression_model.pkl"),
        "linear_scaler": str(horizon_dir / "linear_regression_scaler.pkl"),
        "classifier": str(horizon_dir / "xgboost_classifier.json"),
        "regressor": str(horizon_dir / "xgboost_regressor.json"),
        "preparator": str(horizon_dir / "data_preparator.pkl"),
        "features": str(horizon_dir / "feature_names.pkl"),
        "model_metadata": str(horizon_dir / "model_metadata.pkl"),
    }


def build_rank_ndcg_model_paths(prediction_days: int) -> dict:
    """Return artifact paths for a grouped learning-to-rank horizon."""
    horizon_paths = build_horizon_model_paths(prediction_days)
    horizon_dir = Path("models") / f"horizon_{prediction_days}"
    return {
        "ranker": str(horizon_dir / "xgboost_ranker.json"),
        "preparator": horizon_paths["preparator"],
        "features": horizon_paths["features"],
        "model_metadata": horizon_paths["model_metadata"],
    }


def save_model_artifacts(
    artifact_paths,
    linear_model,
    scaler_lr,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
    regressor,
):
    """Save trained artifacts to one complete path set."""
    for path in artifact_paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(linear_model, artifact_paths["linear"])
    joblib.dump(scaler_lr, artifact_paths["linear_scaler"])
    joblib.dump(data_preparator, artifact_paths["preparator"])
    joblib.dump(all_features, artifact_paths["features"])
    joblib.dump(model_metadata, artifact_paths["model_metadata"])
    classifier.save_model(artifact_paths["classifier"])
    regressor.save_model(artifact_paths["regressor"])


def save_horizon_model_artifacts(
    prediction_days,
    linear_model,
    scaler_lr,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
    regressor,
):
    """Save horizon-specific artifacts and preserve legacy paths for default horizon."""
    horizon_paths = build_horizon_model_paths(prediction_days)
    save_model_artifacts(
        horizon_paths,
        linear_model,
        scaler_lr,
        data_preparator,
        all_features,
        model_metadata,
        classifier,
        regressor,
    )

    if prediction_days == PREDICTION_DAYS:
        save_model_artifacts(
            MODEL_PATHS,
            linear_model,
            scaler_lr,
            data_preparator,
            all_features,
            model_metadata,
            classifier,
            regressor,
        )

    return horizon_paths


def save_ranking_model_artifacts(
    prediction_days,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
):
    """Save ranking-mode artifacts without creating a regression artifact."""
    horizon_paths = build_horizon_model_paths(prediction_days)
    for path_key in ("preparator", "features", "model_metadata", "classifier"):
        Path(horizon_paths[path_key]).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(data_preparator, horizon_paths["preparator"])
    joblib.dump(all_features, horizon_paths["features"])
    joblib.dump(model_metadata, horizon_paths["model_metadata"])
    classifier.save_model(horizon_paths["classifier"])

    if prediction_days == PREDICTION_DAYS:
        for path_key in ("preparator", "features", "model_metadata", "classifier"):
            Path(MODEL_PATHS[path_key]).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data_preparator, MODEL_PATHS["preparator"])
        joblib.dump(all_features, MODEL_PATHS["features"])
        joblib.dump(model_metadata, MODEL_PATHS["model_metadata"])
        classifier.save_model(MODEL_PATHS["classifier"])

    return horizon_paths


def save_rank_ndcg_model_artifacts(
    prediction_days,
    data_preparator,
    all_features,
    model_metadata,
    ranker,
):
    """Save grouped ranker artifacts without classifier or regressor artifacts."""
    horizon_paths = build_rank_ndcg_model_paths(prediction_days)
    for path_key in ("preparator", "features", "model_metadata", "ranker"):
        Path(horizon_paths[path_key]).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(data_preparator, horizon_paths["preparator"])
    joblib.dump(all_features, horizon_paths["features"])
    joblib.dump(model_metadata, horizon_paths["model_metadata"])
    ranker.save_model(horizon_paths["ranker"])

    if prediction_days == PREDICTION_DAYS:
        legacy_ranker_path = "models/xgboost_ranker.json"
        for path in (
            MODEL_PATHS["preparator"],
            MODEL_PATHS["features"],
            MODEL_PATHS["model_metadata"],
            legacy_ranker_path,
        ):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data_preparator, MODEL_PATHS["preparator"])
        joblib.dump(all_features, MODEL_PATHS["features"])
        joblib.dump(model_metadata, MODEL_PATHS["model_metadata"])
        ranker.save_model(legacy_ranker_path)

    return horizon_paths


