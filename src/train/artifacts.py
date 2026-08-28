"""Immutable training artifact bundles, manifests, and publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping

import joblib

from src.config import MODEL_ARTIFACT_ROOT
from src.train.training_contract import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    TARGET_MODE_EXCESS_RETURN,
    validate_target_mode,
)


BUNDLE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CURRENT_POINTER_FILENAME = "current.json"

_BUNDLE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_COMMON_ARTIFACT_FILENAMES = {
    "preparator": "data_preparator.pkl",
    "features": "feature_names.pkl",
    "model_metadata": "model_metadata.pkl",
}

_MODE_BUNDLE_SPECS = {
    TARGET_MODE_EXCESS_RETURN: {
        "model_artifact_type": "linear_classifier_regressor",
        "model_feature_roles": ("linear", "classifier", "regressor"),
        "artifact_filenames": {
            **_COMMON_ARTIFACT_FILENAMES,
            "linear": "linear_regression_model.pkl",
            "linear_scaler": "linear_regression_scaler.pkl",
            "classifier": "xgboost_classifier.json",
            "regressor": "xgboost_regressor.json",
        },
    },
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM: {
        "model_artifact_type": "classifier_only",
        "model_feature_roles": ("classifier",),
        "artifact_filenames": {
            **_COMMON_ARTIFACT_FILENAMES,
            "classifier": "xgboost_classifier.json",
        },
    },
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG: {
        "model_artifact_type": "ranker",
        "model_feature_roles": ("ranker",),
        "artifact_filenames": {
            **_COMMON_ARTIFACT_FILENAMES,
            "ranker": "xgboost_ranker.json",
        },
    },
}

_SEMANTIC_METADATA_KEYS = (
    "target_type",
    "primary_model_score",
    "classifier_target",
    "regressor_target",
    "ranking_target",
    "ranking_group_key",
    "ranking_label_source",
    "ranking_label_type",
)


class ArtifactBundleValidationError(ValueError):
    """Raised when an artifact bundle or pointer is incomplete or incompatible."""


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


def _new_bundle_id() -> str:
    """Return a collision-resistant immutable bundle identifier."""
    return uuid.uuid4().hex


def _horizon_directory(
    target_mode: str,
    prediction_days: int,
    artifact_root: str | Path = MODEL_ARTIFACT_ROOT,
) -> Path:
    validate_target_mode(target_mode)
    if isinstance(prediction_days, bool) or not isinstance(prediction_days, int):
        raise ValueError("prediction_days must be a positive integer.")
    if prediction_days <= 0:
        raise ValueError("prediction_days must be a positive integer.")
    return Path(artifact_root) / target_mode / f"horizon_{prediction_days}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(value, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _atomic_write_json(path: Path, value: dict) -> None:
    """Atomically replace one JSON file in its destination directory."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as json_file:
            json.dump(value, json_file, indent=2, sort_keys=True)
            json_file.write("\n")
            json_file.flush()
            os.fsync(json_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _ordered_model_features(target_mode: str, model_metadata: dict) -> dict:
    if target_mode == TARGET_MODE_EXCESS_RETURN:
        source_keys = {
            "linear": "linear_features",
            "classifier": "classifier_features",
            "regressor": "regressor_features",
        }
    elif target_mode == TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM:
        source_keys = {"classifier": "classifier_features"}
    else:
        # Ranker inputs currently flow through classifier-named training variables.
        # The manifest records their actual artifact role explicitly.
        source_keys = {"ranker": "classifier_features"}

    return {
        role: list(model_metadata.get(metadata_key, []))
        for role, metadata_key in source_keys.items()
    }


def _build_manifest(
    *,
    bundle_id: str,
    target_mode: str,
    prediction_days: int,
    all_features: list[str],
    model_metadata: dict,
    artifact_paths: Mapping[str, Path],
) -> dict:
    spec = _MODE_BUNDLE_SPECS[target_mode]
    semantics = {
        key: model_metadata[key]
        for key in _SEMANTIC_METADATA_KEYS
        if model_metadata.get(key) is not None
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "target_mode": target_mode,
        "model_artifact_type": spec["model_artifact_type"],
        "prediction_days": prediction_days,
        "benchmark_ticker": model_metadata.get("benchmark_ticker"),
        "feature_contract": {
            "preprocessor": list(all_features),
            "models": _ordered_model_features(target_mode, model_metadata),
        },
        "score_target_semantics": semantics,
        "artifacts": {
            artifact_name: {
                "filename": artifact_path.name,
                "sha256": _sha256_file(artifact_path),
            }
            for artifact_name, artifact_path in artifact_paths.items()
        },
    }


def _require_nonempty_ordered_strings(value, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ArtifactBundleValidationError(
            f"Manifest {field_name} must be a non-empty list."
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise ArtifactBundleValidationError(
            f"Manifest {field_name} must contain non-empty strings."
        )
    if len(set(value)) != len(value):
        raise ArtifactBundleValidationError(
            f"Manifest {field_name} must not contain duplicates."
        )


def _validate_manifest_data(
    manifest: dict,
    bundle_dir: Path,
    *,
    expected_target_mode: str | None = None,
    expected_prediction_days: int | None = None,
    expected_bundle_id: str | None = None,
) -> None:
    if not isinstance(manifest, dict):
        raise ArtifactBundleValidationError("Bundle manifest must be a JSON object.")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ArtifactBundleValidationError(
            "Unsupported or missing artifact bundle schema_version."
        )

    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or not _BUNDLE_ID_PATTERN.fullmatch(bundle_id):
        raise ArtifactBundleValidationError("Manifest bundle_id is invalid.")
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ArtifactBundleValidationError(
            f"Manifest bundle_id {bundle_id!r} does not match {expected_bundle_id!r}."
        )

    target_mode = manifest.get("target_mode")
    try:
        validate_target_mode(target_mode)
    except ValueError as error:
        raise ArtifactBundleValidationError(str(error)) from error
    if expected_target_mode is not None and target_mode != expected_target_mode:
        raise ArtifactBundleValidationError(
            f"Manifest target_mode {target_mode!r} does not match "
            f"{expected_target_mode!r}."
        )

    prediction_days = manifest.get("prediction_days")
    if (
        isinstance(prediction_days, bool)
        or not isinstance(prediction_days, int)
        or prediction_days <= 0
    ):
        raise ArtifactBundleValidationError(
            "Manifest prediction_days must be a positive integer."
        )
    if (
        expected_prediction_days is not None
        and prediction_days != expected_prediction_days
    ):
        raise ArtifactBundleValidationError(
            f"Manifest prediction_days {prediction_days!r} does not match "
            f"{expected_prediction_days!r}."
        )

    benchmark_ticker = manifest.get("benchmark_ticker")
    if not isinstance(benchmark_ticker, str) or not benchmark_ticker:
        raise ArtifactBundleValidationError(
            "Manifest benchmark_ticker must be a non-empty string."
        )

    spec = _MODE_BUNDLE_SPECS[target_mode]
    if manifest.get("model_artifact_type") != spec["model_artifact_type"]:
        raise ArtifactBundleValidationError(
            "Manifest model_artifact_type is incompatible with target_mode."
        )

    feature_contract = manifest.get("feature_contract")
    if not isinstance(feature_contract, dict):
        raise ArtifactBundleValidationError("Manifest feature_contract is invalid.")
    preprocessor_features = feature_contract.get("preprocessor")
    _require_nonempty_ordered_strings(
        preprocessor_features,
        "feature_contract.preprocessor",
    )
    model_features = feature_contract.get("models")
    expected_roles = set(spec["model_feature_roles"])
    if not isinstance(model_features, dict) or set(model_features) != expected_roles:
        raise ArtifactBundleValidationError(
            "Manifest model feature roles are incompatible with target_mode."
        )
    for role, ordered_features in model_features.items():
        _require_nonempty_ordered_strings(
            ordered_features,
            f"feature_contract.models.{role}",
        )
        unknown_features = set(ordered_features).difference(preprocessor_features)
        if unknown_features:
            raise ArtifactBundleValidationError(
                f"Manifest model features for {role} are not in the preprocessor "
                f"contract: {sorted(unknown_features)}"
            )

    semantics = manifest.get("score_target_semantics")
    if not isinstance(semantics, dict) or not semantics:
        raise ArtifactBundleValidationError(
            "Manifest score_target_semantics must be a non-empty object."
        )

    artifacts = manifest.get("artifacts")
    expected_artifacts = spec["artifact_filenames"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise ArtifactBundleValidationError(
            "Manifest artifact set is incomplete or incompatible with target_mode."
        )
    for artifact_name, expected_filename in expected_artifacts.items():
        artifact_record = artifacts[artifact_name]
        if not isinstance(artifact_record, dict):
            raise ArtifactBundleValidationError(
                f"Manifest artifact record {artifact_name!r} is invalid."
            )
        if artifact_record.get("filename") != expected_filename:
            raise ArtifactBundleValidationError(
                f"Manifest artifact filename for {artifact_name!r} is invalid."
            )
        expected_checksum = artifact_record.get("sha256")
        if not isinstance(expected_checksum, str) or not _SHA256_PATTERN.fullmatch(
            expected_checksum
        ):
            raise ArtifactBundleValidationError(
                f"Manifest checksum for {artifact_name!r} is invalid."
            )
        artifact_path = bundle_dir / expected_filename
        if not artifact_path.is_file():
            raise ArtifactBundleValidationError(
                f"Required artifact file is missing: {expected_filename}"
            )
        actual_checksum = _sha256_file(artifact_path)
        if actual_checksum != expected_checksum:
            raise ArtifactBundleValidationError(
                f"Checksum mismatch for artifact: {expected_filename}"
            )


def load_and_validate_bundle_manifest(
    bundle_dir: str | Path,
    *,
    expected_target_mode: str | None = None,
    expected_prediction_days: int | None = None,
) -> dict:
    """Read plain JSON first, then validate bundle identity and file checksums."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_FILENAME
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactBundleValidationError(
            f"Unable to read bundle manifest: {manifest_path}"
        ) from error

    _validate_manifest_data(
        manifest,
        bundle_dir,
        expected_target_mode=expected_target_mode,
        expected_prediction_days=expected_prediction_days,
        expected_bundle_id=bundle_dir.name,
    )
    return manifest


def load_current_bundle_manifest(
    target_mode: str,
    prediction_days: int,
    *,
    artifact_root: str | Path = MODEL_ARTIFACT_ROOT,
) -> dict:
    """Resolve current.json and validate its immutable bundle without loading pickle."""
    horizon_dir = _horizon_directory(target_mode, prediction_days, artifact_root)
    current_path = horizon_dir / CURRENT_POINTER_FILENAME
    try:
        with current_path.open("r", encoding="utf-8") as current_file:
            pointer = json.load(current_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactBundleValidationError(
            f"Unable to read current bundle pointer: {current_path}"
        ) from error

    bundle_id = pointer.get("bundle_id") if isinstance(pointer, dict) else None
    expected_pointer = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "target_mode": target_mode,
        "prediction_days": prediction_days,
        "manifest": f"{bundle_id}/{MANIFEST_FILENAME}",
    }
    if (
        not isinstance(bundle_id, str)
        or not _BUNDLE_ID_PATTERN.fullmatch(bundle_id)
        or pointer != expected_pointer
    ):
        raise ArtifactBundleValidationError(
            f"Current bundle pointer is invalid: {current_path}"
        )

    return load_and_validate_bundle_manifest(
        horizon_dir / bundle_id,
        expected_target_mode=target_mode,
        expected_prediction_days=prediction_days,
    )


def _publish_artifact_bundle(
    *,
    target_mode: str,
    prediction_days: int,
    all_features: list[str],
    model_metadata: dict,
    artifact_writers: Mapping[str, Callable[[Path], None]],
) -> dict[str, str]:
    """Write, validate, atomically publish, and select one immutable bundle."""
    horizon_dir = _horizon_directory(target_mode, prediction_days)
    spec = _MODE_BUNDLE_SPECS[target_mode]
    expected_artifact_names = set(spec["artifact_filenames"])
    if set(artifact_writers) != expected_artifact_names:
        raise ValueError(
            "Artifact writers do not match the required bundle artifacts for "
            f"target_mode={target_mode!r}."
        )
    if model_metadata.get("target_mode") != target_mode:
        raise ValueError("model_metadata target_mode does not match the bundle mode.")
    if model_metadata.get("prediction_days") != prediction_days:
        raise ValueError(
            "model_metadata prediction_days does not match the bundle horizon."
        )

    horizon_dir.mkdir(parents=True, exist_ok=True)
    bundle_id = _new_bundle_id()
    final_dir = horizon_dir / bundle_id
    if final_dir.exists():
        raise FileExistsError(f"Artifact bundle already exists: {final_dir}")

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_id}.staging-",
            dir=horizon_dir,
        )
    )
    published = False
    try:
        staged_artifact_paths = {
            artifact_name: staging_dir / filename
            for artifact_name, filename in spec["artifact_filenames"].items()
        }
        for artifact_name, artifact_path in staged_artifact_paths.items():
            artifact_writers[artifact_name](artifact_path)

        manifest = _build_manifest(
            bundle_id=bundle_id,
            target_mode=target_mode,
            prediction_days=prediction_days,
            all_features=list(all_features),
            model_metadata=model_metadata,
            artifact_paths=staged_artifact_paths,
        )
        _write_json(staging_dir / MANIFEST_FILENAME, manifest)
        _validate_manifest_data(
            manifest,
            staging_dir,
            expected_target_mode=target_mode,
            expected_prediction_days=prediction_days,
            expected_bundle_id=bundle_id,
        )

        # Staging and final paths share a parent/filesystem, so this directory rename
        # publishes the complete validated bundle as one atomic operation.
        staging_dir.rename(final_dir)
        published = True
        load_and_validate_bundle_manifest(
            final_dir,
            expected_target_mode=target_mode,
            expected_prediction_days=prediction_days,
        )

        current_path = horizon_dir / CURRENT_POINTER_FILENAME
        _atomic_write_json(
            current_path,
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "bundle_id": bundle_id,
                "target_mode": target_mode,
                "prediction_days": prediction_days,
                "manifest": f"{bundle_id}/{MANIFEST_FILENAME}",
            },
        )
    finally:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)

    artifact_paths = {
        artifact_name: str(final_dir / filename)
        for artifact_name, filename in spec["artifact_filenames"].items()
    }
    artifact_paths.update(
        {
            "manifest": str(final_dir / MANIFEST_FILENAME),
            "bundle_dir": str(final_dir),
            "current": str(horizon_dir / CURRENT_POINTER_FILENAME),
        }
    )
    return artifact_paths


def _joblib_writer(value) -> Callable[[Path], None]:
    return lambda path: joblib.dump(value, path)


def save_excess_return_model_artifacts(
    prediction_days,
    linear_model,
    scaler_lr,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
    regressor,
):
    """Publish one excess-return regression/classification artifact bundle."""
    return _publish_artifact_bundle(
        target_mode=TARGET_MODE_EXCESS_RETURN,
        prediction_days=prediction_days,
        all_features=all_features,
        model_metadata=model_metadata,
        artifact_writers={
            "preparator": _joblib_writer(data_preparator),
            "features": _joblib_writer(all_features),
            "model_metadata": _joblib_writer(model_metadata),
            "linear": _joblib_writer(linear_model),
            "linear_scaler": _joblib_writer(scaler_lr),
            "classifier": classifier.save_model,
            "regressor": regressor.save_model,
        },
    )


def save_cross_sectional_top_bottom_artifacts(
    prediction_days,
    data_preparator,
    all_features,
    model_metadata,
    classifier,
):
    """Publish one cross-sectional top/bottom classifier artifact bundle."""
    return _publish_artifact_bundle(
        target_mode=TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
        prediction_days=prediction_days,
        all_features=all_features,
        model_metadata=model_metadata,
        artifact_writers={
            "preparator": _joblib_writer(data_preparator),
            "features": _joblib_writer(all_features),
            "model_metadata": _joblib_writer(model_metadata),
            "classifier": classifier.save_model,
        },
    )


def save_cross_sectional_rank_ndcg_artifacts(
    prediction_days,
    data_preparator,
    all_features,
    model_metadata,
    ranker,
):
    """Publish one grouped Rank-NDCG artifact bundle."""
    return _publish_artifact_bundle(
        target_mode=TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
        prediction_days=prediction_days,
        all_features=all_features,
        model_metadata=model_metadata,
        artifact_writers={
            "preparator": _joblib_writer(data_preparator),
            "features": _joblib_writer(all_features),
            "model_metadata": _joblib_writer(model_metadata),
            "ranker": ranker.save_model,
        },
    )
