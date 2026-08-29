"""Focused tests for immutable training artifact bundles."""

import json
from pathlib import Path

import pytest

import src.train.artifacts as artifacts


class FakeXgbModel:
    def __init__(self, content="model"):
        self.content = content

    def save_model(self, path):
        Path(path).write_text(self.content, encoding="utf-8")


class BrokenXgbModel:
    def save_model(self, path):
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("model save failed")


def build_metadata(target_mode, prediction_days, features=None):
    features = list(features or ["Close", "Volume"])
    training_population = {
        "universe_name": "test_universe",
        "candidate_tickers": ["AAA", "BBB"],
        "candidate_ticker_count": 2,
        "benchmark_ticker": "SPY",
    }
    return artifacts.build_model_metadata(
        linear_features=["Close"],
        classifier_features=features,
        regressor_features=(
            features if target_mode == "excess_return" else []
        ),
        prediction_days=prediction_days,
        target_mode=target_mode,
        training_population=training_population,
    )


def save_bundle(target_mode, prediction_days, *, model=None, metadata=None):
    features = ["Close", "Volume"]
    model = model or FakeXgbModel()
    metadata = metadata or build_metadata(target_mode, prediction_days, features)
    if target_mode == "excess_return":
        return artifacts.save_excess_return_model_artifacts(
            prediction_days,
            linear_model={"linear": True},
            scaler_lr={"scaler": True},
            data_preparator={"preparator": True},
            all_features=features,
            model_metadata=metadata,
            classifier=model,
            regressor=FakeXgbModel("regressor"),
        )
    if target_mode == "cross_sectional_top_bottom":
        return artifacts.save_cross_sectional_top_bottom_artifacts(
            prediction_days,
            data_preparator={"preparator": True},
            all_features=features,
            model_metadata=metadata,
            classifier=model,
        )
    return artifacts.save_cross_sectional_rank_ndcg_artifacts(
        prediction_days,
        data_preparator={"preparator": True},
        all_features=features,
        model_metadata=metadata,
        ranker=model,
    )


def test_model_metadata_does_not_include_linear_regression_prediction():
    metadata = build_metadata("excess_return", 5)

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


def test_model_metadata_records_cross_sectional_top_bottom_fields():
    metadata = build_metadata("cross_sectional_top_bottom", 5)

    assert metadata["target_type"] == "cross_sectional_top_bottom_quintile"
    assert metadata["classifier_target"] == "top_quintile_target"
    assert metadata["regressor_target"] is None
    assert metadata["ranking_target"] == (
        "top_20_vs_bottom_20_by_excess_forward_return"
    )
    assert metadata["ranking_group_key"] == "prediction_date"
    assert metadata["ranking_training_sample"] == "ranking_train_sample"
    assert metadata["primary_model_score"] == (
        "probability_of_top_quintile_outperformance"
    )


def test_model_metadata_records_cross_sectional_rank_ndcg_fields():
    metadata = build_metadata("cross_sectional_rank_ndcg", 5)

    assert metadata["target_type"] == "grouped_learning_to_rank_ndcg"
    assert metadata["classifier_target"] is None
    assert metadata["regressor_target"] is None
    assert metadata["ranking_group_key"] == "prediction_date"
    assert metadata["ranking_label_source"] == "excess_return_rank_pct_by_date"
    assert metadata["ranking_label_type"] == "graded_0_to_4_from_rank_percentile"
    assert metadata["primary_model_score"] == "xgboost_rank_ndcg_score"
    assert metadata["model_artifact_type"] == "ranker"


def test_saved_bundles_are_isolated_by_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected_artifact_types = {
        "excess_return": "linear_classifier_regressor",
        "cross_sectional_top_bottom": "classifier_only",
        "cross_sectional_rank_ndcg": "ranker",
    }

    saved = {
        mode: save_bundle(mode, 10)
        for mode in (
            "excess_return",
            "cross_sectional_top_bottom",
            "cross_sectional_rank_ndcg",
        )
    }

    for mode, paths in saved.items():
        bundle_dir = Path(paths["bundle_dir"])
        assert bundle_dir.parent == Path("models") / mode / "horizon_10"
        assert (bundle_dir / "data_preparator.pkl").exists()
        assert (bundle_dir / "feature_names.pkl").exists()
        assert (bundle_dir / "model_metadata.pkl").exists()
        assert Path(paths["current"]).exists()
        manifest = artifacts.load_current_bundle_manifest(mode, 10)
        assert manifest["target_mode"] == mode
        assert manifest["model_artifact_type"] == expected_artifact_types[mode]

    assert not (tmp_path / "models/horizon_10").exists()
    assert not (tmp_path / "models/model_metadata.pkl").exists()


def test_saved_bundles_are_isolated_by_horizon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    five_day = save_bundle("cross_sectional_rank_ndcg", 5)
    twenty_day = save_bundle("cross_sectional_rank_ndcg", 20)

    assert Path(five_day["bundle_dir"]).parent == Path(
        "models/cross_sectional_rank_ndcg/horizon_5"
    )
    assert Path(twenty_day["bundle_dir"]).parent == Path(
        "models/cross_sectional_rank_ndcg/horizon_20"
    )
    assert Path(five_day["current"]) != Path(twenty_day["current"])


def test_manifest_records_contract_semantics_artifacts_and_checksums(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    paths = save_bundle("cross_sectional_rank_ndcg", 10)

    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == artifacts.BUNDLE_SCHEMA_VERSION
    assert manifest["bundle_id"] == manifest_path.parent.name
    assert manifest["target_mode"] == "cross_sectional_rank_ndcg"
    assert manifest["model_artifact_type"] == "ranker"
    assert manifest["prediction_days"] == 10
    assert manifest["benchmark_ticker"] == "SPY"
    assert manifest["training_population"] == {
        "universe_name": "test_universe",
        "candidate_tickers": ["AAA", "BBB"],
        "candidate_ticker_count": 2,
        "benchmark_ticker": "SPY",
    }
    assert manifest["feature_contract"] == {
        "preprocessor": ["Close", "Volume"],
        "models": {"ranker": ["Close", "Volume"]},
    }
    assert manifest["score_target_semantics"]["primary_model_score"] == (
        "xgboost_rank_ndcg_score"
    )
    assert manifest["score_target_semantics"]["ranking_group_key"] == (
        "prediction_date"
    )
    assert set(manifest["artifacts"]) == {
        "preparator",
        "features",
        "model_metadata",
        "ranker",
    }
    for artifact_record in manifest["artifacts"].values():
        assert len(artifact_record["sha256"]) == 64
        assert (manifest_path.parent / artifact_record["filename"]).is_file()

    assert artifacts.load_and_validate_bundle_manifest(
        manifest_path.parent,
        expected_target_mode="cross_sectional_rank_ndcg",
        expected_prediction_days=10,
    ) == manifest


def test_manifest_validation_occurs_without_loading_joblib(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    paths = save_bundle("cross_sectional_top_bottom", 10)
    monkeypatch.setattr(
        artifacts.joblib,
        "load",
        lambda path: pytest.fail(f"joblib.load must not run before validation: {path}"),
    )

    manifest = artifacts.load_current_bundle_manifest(
        "cross_sectional_top_bottom",
        10,
    )

    assert manifest["target_mode"] == "cross_sectional_top_bottom"


def test_bundle_ids_are_immutable_and_current_moves_to_new_bundle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    first = save_bundle("excess_return", 10)
    first_manifest_path = Path(first["manifest"])
    first_manifest_bytes = first_manifest_path.read_bytes()

    second = save_bundle("excess_return", 10)

    assert first["bundle_dir"] != second["bundle_dir"]
    assert first_manifest_path.read_bytes() == first_manifest_bytes
    assert Path(first["bundle_dir"]).is_dir()
    assert Path(second["bundle_dir"]).is_dir()
    current = json.loads(Path(second["current"]).read_text(encoding="utf-8"))
    assert current["bundle_id"] == Path(second["bundle_dir"]).name

    monkeypatch.setattr(
        artifacts,
        "_new_bundle_id",
        lambda: Path(first["bundle_dir"]).name,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        save_bundle("excess_return", 10)
    assert first_manifest_path.read_bytes() == first_manifest_bytes


def test_current_pointer_identifies_and_resolves_published_bundle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    paths = save_bundle("cross_sectional_rank_ndcg", 20)
    current_path = Path(paths["current"])
    pointer = json.loads(current_path.read_text(encoding="utf-8"))

    assert pointer == {
        "schema_version": artifacts.BUNDLE_SCHEMA_VERSION,
        "bundle_id": Path(paths["bundle_dir"]).name,
        "target_mode": "cross_sectional_rank_ndcg",
        "prediction_days": 20,
        "manifest": f"{Path(paths['bundle_dir']).name}/manifest.json",
    }
    manifest = artifacts.load_current_bundle_manifest(
        "cross_sectional_rank_ndcg",
        20,
    )
    assert manifest["bundle_id"] == pointer["bundle_id"]


def test_checksum_validation_rejects_modified_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = save_bundle("cross_sectional_rank_ndcg", 10)
    Path(paths["ranker"]).write_text("tampered", encoding="utf-8")

    with pytest.raises(
        artifacts.ArtifactBundleValidationError,
        match="Checksum mismatch",
    ):
        artifacts.load_and_validate_bundle_manifest(paths["bundle_dir"])


def test_schema_one_bundle_without_provenance_is_not_reconstructed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    paths = save_bundle("cross_sectional_rank_ndcg", 10)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    del manifest["training_population"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        artifacts.ArtifactBundleValidationError,
        match="schema_version",
    ):
        artifacts.load_and_validate_bundle_manifest(paths["bundle_dir"])


def test_partial_save_cannot_replace_current_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = save_bundle("cross_sectional_rank_ndcg", 10)
    current_path = Path(first["current"])
    current_bytes = current_path.read_bytes()
    horizon_dir = current_path.parent

    with pytest.raises(RuntimeError, match="model save failed"):
        save_bundle(
            "cross_sectional_rank_ndcg",
            10,
            model=BrokenXgbModel(),
        )

    assert current_path.read_bytes() == current_bytes
    assert sorted(path.name for path in horizon_dir.iterdir()) == sorted(
        [Path(first["bundle_dir"]).name, "current.json"]
    )


def test_invalid_bundle_cannot_replace_current_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = save_bundle("cross_sectional_top_bottom", 10)
    current_path = Path(first["current"])
    current_bytes = current_path.read_bytes()
    bad_metadata = build_metadata("cross_sectional_top_bottom", 10)
    del bad_metadata["benchmark_ticker"]

    with pytest.raises(
        artifacts.ArtifactBundleValidationError,
        match="benchmark_ticker",
    ):
        save_bundle(
            "cross_sectional_top_bottom",
            10,
            metadata=bad_metadata,
        )

    assert current_path.read_bytes() == current_bytes
    assert artifacts.load_current_bundle_manifest(
        "cross_sectional_top_bottom",
        10,
    )["bundle_id"] == Path(first["bundle_dir"]).name
