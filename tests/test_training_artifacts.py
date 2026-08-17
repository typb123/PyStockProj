"""Focused tests for training artifact metadata, paths, and persistence."""

import src.train.artifacts as artifacts

def test_model_metadata_does_not_include_linear_regression_prediction():
    metadata = artifacts.build_model_metadata(
        linear_features=["Close"],
        classifier_features=["Close", "Volume"],
        regressor_features=["Close", "Volume"],
        prediction_days=5,
    )

    assert "LinearRegression_Prediction" not in metadata["classifier_features"]
    assert "LinearRegression_Prediction" not in metadata["regressor_features"]
    assert "LinearRegression_Prediction" not in metadata["classifier_feature_names"]
    assert "LinearRegression_Prediction" not in metadata["regressor_feature_names"]
    assert metadata["prediction_days"] == 5
    assert metadata["target_mode"] == "excess_return"
    assert metadata["target_type"] == "spy_relative_excess_forward_return"
    assert metadata["benchmark_ticker"] == "SPY"
    assert metadata["regressor_target"] == "targetReturns"
    assert metadata["classifier_target"] == "beat_benchmark_target"


def test_model_metadata_records_explicit_target_mode():
    metadata = artifacts.build_model_metadata(
        linear_features=["Close"],
        classifier_features=["Close", "Volume"],
        regressor_features=["Close", "Volume"],
        prediction_days=5,
        target_mode="excess_return",
    )

    assert metadata["target_mode"] == "excess_return"


def test_model_metadata_records_cross_sectional_top_bottom_fields():
    metadata = artifacts.build_model_metadata(
        linear_features=["Close"],
        classifier_features=["Close", "Volume"],
        regressor_features=["Close", "Volume"],
        prediction_days=5,
        target_mode="cross_sectional_top_bottom",
    )

    assert metadata["target_type"] == "cross_sectional_top_bottom_quintile"
    assert metadata["classifier_target"] == "top_quintile_target"
    assert metadata["regressor_target"] is None
    assert metadata["ranking_target"] == "top_20_vs_bottom_20_by_excess_forward_return"
    assert metadata["ranking_group_key"] == "prediction_date"
    assert metadata["ranking_training_sample"] == "ranking_train_sample"
    assert metadata["primary_model_score"] == "probability_of_top_quintile_outperformance"


def test_model_metadata_records_cross_sectional_rank_ndcg_fields():
    metadata = artifacts.build_model_metadata(
        linear_features=["Close"],
        classifier_features=["Close", "Volume"],
        regressor_features=["Close", "Volume"],
        prediction_days=5,
        target_mode="cross_sectional_rank_ndcg",
    )

    assert metadata["target_type"] == "grouped_learning_to_rank_ndcg"
    assert metadata["classifier_target"] is None
    assert metadata["regressor_target"] is None
    assert metadata["ranking_group_key"] == "prediction_date"
    assert metadata["ranking_label_source"] == "excess_return_rank_pct_by_date"
    assert metadata["ranking_label_type"] == "graded_0_to_4_from_rank_percentile"
    assert metadata["primary_model_score"] == "xgboost_rank_ndcg_score"
    assert metadata["model_artifact_type"] == "ranker"



def test_build_horizon_model_paths_uses_horizon_specific_directory():
    paths = artifacts.build_horizon_model_paths(20)

    assert paths["model_metadata"] == "models/horizon_20/model_metadata.pkl"
    assert paths["classifier"] == "models/horizon_20/xgboost_classifier.json"
    assert paths["regressor"] == "models/horizon_20/xgboost_regressor.json"


def test_build_rank_ndcg_model_paths_uses_separate_ranker_artifact():
    paths = artifacts.build_rank_ndcg_model_paths(20)

    assert paths == {
        "ranker": "models/horizon_20/xgboost_ranker.json",
        "preparator": "models/horizon_20/data_preparator.pkl",
        "features": "models/horizon_20/feature_names.pkl",
        "model_metadata": "models/horizon_20/model_metadata.pkl",
    }


def test_save_horizon_model_artifacts_writes_horizon_specific_paths(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    class FakeXgbModel:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as model_file:
                model_file.write("model")

    saved_paths = artifacts.save_horizon_model_artifacts(
        5,
        linear_model={"linear": True},
        scaler_lr={"scaler": True},
        data_preparator={"preparator": True},
        all_features=["Close"],
        model_metadata={"prediction_days": 5},
        classifier=FakeXgbModel(),
        regressor=FakeXgbModel(),
    )

    assert saved_paths["model_metadata"] == "models/horizon_5/model_metadata.pkl"
    assert (tmp_path / "models/horizon_5/model_metadata.pkl").exists()
    assert (tmp_path / "models/horizon_5/xgboost_classifier.json").exists()
    assert not (tmp_path / "models/model_metadata.pkl").exists()


def test_save_horizon_model_artifacts_preserves_legacy_paths_for_default_horizon(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    class FakeXgbModel:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as model_file:
                model_file.write("model")

    artifacts.save_horizon_model_artifacts(
        10,
        linear_model={"linear": True},
        scaler_lr={"scaler": True},
        data_preparator={"preparator": True},
        all_features=["Close"],
        model_metadata={"prediction_days": 10},
        classifier=FakeXgbModel(),
        regressor=FakeXgbModel(),
    )

    assert (tmp_path / "models/horizon_10/model_metadata.pkl").exists()
    assert (tmp_path / "models/model_metadata.pkl").exists()


def test_save_ranking_model_artifacts_uses_only_horizon_paths_for_non_default_horizon(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    class FakeClassifier:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as model_file:
                model_file.write("classifier")

    saved_paths = artifacts.save_ranking_model_artifacts(
        5,
        data_preparator={"preparator": True},
        all_features=["Close"],
        model_metadata={"prediction_days": 5},
        classifier=FakeClassifier(),
    )

    assert saved_paths == artifacts.build_horizon_model_paths(5)
    assert (tmp_path / "models/horizon_5/data_preparator.pkl").exists()
    assert (tmp_path / "models/horizon_5/feature_names.pkl").exists()
    assert (tmp_path / "models/horizon_5/model_metadata.pkl").exists()
    assert (tmp_path / "models/horizon_5/xgboost_classifier.json").exists()
    assert not (tmp_path / "models/horizon_5/xgboost_regressor.json").exists()
    assert not (tmp_path / "models/data_preparator.pkl").exists()
    assert not (tmp_path / "models/feature_names.pkl").exists()
    assert not (tmp_path / "models/model_metadata.pkl").exists()
    assert not (tmp_path / "models/xgboost_classifier.json").exists()


def test_save_rank_ndcg_model_artifacts_writes_only_ranker_model_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    class FakeRanker:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as model_file:
                model_file.write("ranker")

    saved_paths = artifacts.save_rank_ndcg_model_artifacts(
        5,
        data_preparator={"preparator": True},
        all_features=["Close"],
        model_metadata={"prediction_days": 5},
        ranker=FakeRanker(),
    )

    assert set(saved_paths) == {"ranker", "preparator", "features", "model_metadata"}
    assert saved_paths["ranker"] == "models/horizon_5/xgboost_ranker.json"
    assert (tmp_path / "models/horizon_5/xgboost_ranker.json").exists()
    assert not (tmp_path / "models/horizon_5/xgboost_classifier.json").exists()
    assert not (tmp_path / "models/horizon_5/xgboost_regressor.json").exists()

