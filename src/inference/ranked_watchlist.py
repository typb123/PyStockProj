"""Rank-NDCG bundle loading and synchronized ranked-watchlist inference."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.config import MODEL_ARTIFACT_ROOT, get_training_tickers
from src.data.data_prep import add_benchmark_relative_momentum
from src.data.inference_data import prepare_completed_ticker_features
from src.features.feature_contract import (
    MODEL_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)
from src.train.artifacts import (
    ArtifactBundleValidationError,
    resolve_current_bundle,
)
from src.train.training_contract import TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG


SERVABLE_PREDICTION_DAYS = (10, 20)
BENCHMARK_TICKER = "SPY"
INFERENCE_HISTORY_PERIOD = "5y"


class RankedWatchlistError(ValueError):
    """Base error for ranked-watchlist request and compatibility failures."""


class RankedWatchlistArtifactError(RankedWatchlistError):
    """Raised when the requested Rank-NDCG bundle cannot be served safely."""


class RankedWatchlistDataError(RankedWatchlistError):
    """Raised when synchronized market data cannot support a ranking."""


@dataclass(frozen=True)
class LoadedRankNdcgBundle:
    bundle_dir: Path
    manifest: dict
    data_preparator: object
    model_metadata: dict
    preprocessor_features: list[str]
    ranker_features: list[str]
    ranker: object


def _validate_serving_horizon(prediction_days: int) -> None:
    if prediction_days not in SERVABLE_PREDICTION_DAYS:
        supported = ", ".join(f"{days}d" for days in SERVABLE_PREDICTION_DAYS)
        raise RankedWatchlistError(
            f"Unsupported ranked-watchlist horizon {prediction_days!r}; "
            f"explicitly choose one of: {supported}."
        )


def _validate_rank_ndcg_manifest(manifest: dict, prediction_days: int) -> None:
    if manifest.get("target_mode") != TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG:
        raise RankedWatchlistArtifactError(
            "Current artifact bundle is not a Rank-NDCG bundle."
        )
    if manifest.get("model_artifact_type") != "ranker":
        raise RankedWatchlistArtifactError(
            "Current Rank-NDCG bundle does not contain a ranker artifact."
        )
    if manifest.get("prediction_days") != prediction_days:
        raise RankedWatchlistArtifactError(
            "Current Rank-NDCG bundle horizon does not match the requested horizon."
        )

    feature_contract = manifest.get("feature_contract", {})
    preprocessor_features = feature_contract.get("preprocessor")
    ranker_features = feature_contract.get("models", {}).get("ranker")
    if preprocessor_features != list(MODEL_FEATURE_COLUMNS):
        raise RankedWatchlistArtifactError(
            "Rank-NDCG bundle preprocessing feature contract is incompatible with "
            "the current feature implementation."
        )
    if not isinstance(ranker_features, list) or not ranker_features:
        raise RankedWatchlistArtifactError(
            "Rank-NDCG bundle is missing its ordered ranker feature contract."
        )


def _artifact_path(bundle_dir: Path, manifest: dict, artifact_name: str) -> Path:
    try:
        filename = manifest["artifacts"][artifact_name]["filename"]
    except (KeyError, TypeError) as error:
        raise RankedWatchlistArtifactError(
            f"Rank-NDCG bundle is missing artifact metadata for {artifact_name}."
        ) from error
    return bundle_dir / filename


def _load_rank_ndcg_bundle(
    prediction_days: int,
    *,
    artifact_root: str | Path = MODEL_ARTIFACT_ROOT,
) -> LoadedRankNdcgBundle:
    """Validate JSON/checksums first, then load Rank-NDCG serialized artifacts."""
    _validate_serving_horizon(prediction_days)
    try:
        bundle_dir, manifest = resolve_current_bundle(
            TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
            prediction_days,
            artifact_root=artifact_root,
        )
    except ArtifactBundleValidationError as error:
        raise RankedWatchlistArtifactError(
            f"No compatible current Rank-NDCG bundle for {prediction_days}d: {error}"
        ) from error

    _validate_rank_ndcg_manifest(manifest, prediction_days)

    # No serialized object is touched until current.json, manifest identity, schema,
    # required files, and all artifact checksums have passed validation above.
    saved_features = joblib.load(_artifact_path(bundle_dir, manifest, "features"))
    data_preparator = joblib.load(
        _artifact_path(bundle_dir, manifest, "preparator")
    )
    model_metadata = joblib.load(
        _artifact_path(bundle_dir, manifest, "model_metadata")
    )

    preprocessor_features = list(manifest["feature_contract"]["preprocessor"])
    ranker_features = list(manifest["feature_contract"]["models"]["ranker"])
    if list(saved_features) != preprocessor_features:
        raise RankedWatchlistArtifactError(
            "Serialized feature names do not match the validated manifest."
        )
    if list(getattr(data_preparator, "feature_columns", [])) != preprocessor_features:
        raise RankedWatchlistArtifactError(
            "Serialized data preparator does not match the validated feature contract."
        )
    if not hasattr(data_preparator, "scalar") or not hasattr(
        data_preparator.scalar,
        "transform",
    ):
        raise RankedWatchlistArtifactError(
            "Serialized data preparator is missing its fitted scaler."
        )
    if (
        model_metadata.get("target_mode")
        != TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG
        or model_metadata.get("prediction_days") != prediction_days
        or model_metadata.get("benchmark_ticker") != BENCHMARK_TICKER
        or list(model_metadata.get("classifier_features", [])) != ranker_features
        or model_metadata.get("training_population")
        != manifest.get("training_population")
        or model_metadata.get("data_provenance") != manifest.get("data_provenance")
    ):
        raise RankedWatchlistArtifactError(
            "Serialized model metadata does not match the validated manifest."
        )

    ranker = XGBRanker()
    ranker.load_model(_artifact_path(bundle_dir, manifest, "ranker"))
    return LoadedRankNdcgBundle(
        bundle_dir=bundle_dir,
        manifest=manifest,
        data_preparator=data_preparator,
        model_metadata=model_metadata,
        preprocessor_features=preprocessor_features,
        ranker_features=ranker_features,
        ranker=ranker,
    )


def _normalize_tickers(
    tickers: Iterable[str],
    *,
    source_label: str,
) -> list[str]:
    if isinstance(tickers, (str, bytes)):
        raise RankedWatchlistError("tickers must be an iterable of ticker strings.")
    normalized = []
    seen = set()
    try:
        raw_tickers = list(tickers)
    except TypeError as error:
        raise RankedWatchlistError(
            "tickers must be an iterable of ticker strings."
        ) from error
    for raw_ticker in raw_tickers:
        if not isinstance(raw_ticker, str) or not raw_ticker.strip():
            raise RankedWatchlistError(
                f"{source_label} tickers must contain only non-empty strings."
            )
        ticker = raw_ticker.strip().upper()
        if ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)
    if not normalized:
        raise RankedWatchlistError(f"{source_label} tickers must not be empty.")
    return normalized


def _resolve_candidate_source(*, universe, tickers):
    if (universe is None) == (tickers is None):
        raise RankedWatchlistError(
            "Specify exactly one candidate source: universe or tickers."
        )

    skipped = []
    if universe is not None:
        requested_tickers = _normalize_tickers(
            get_training_tickers(universe),
            source_label="Configured universe",
        )
        candidates = [
            ticker for ticker in requested_tickers if ticker != BENCHMARK_TICKER
        ]
        source = {
            "type": "configured_universe",
            "universe_name": universe,
            "validation_context": "configured_research_universe",
        }
    else:
        requested_tickers = _normalize_tickers(
            tickers,
            source_label="Custom",
        )
        candidates = []
        for ticker in requested_tickers:
            if ticker == BENCHMARK_TICKER:
                skipped.append(
                    {
                        "ticker": ticker,
                        "reason": "benchmark ticker is excluded from candidates",
                    }
                )
            else:
                candidates.append(ticker)
        source = {
            "type": "custom",
            "universe_name": None,
            "validation_context": "no_historical_validation_claim",
        }

    requested_count = len(candidates)

    if len(candidates) < 2:
        raise RankedWatchlistError(
            "At least 2 candidate tickers are required after SPY exclusion."
        )
    return candidates, source, requested_count, skipped


def _apply_training_population_context(
    source: dict,
    candidates: list[str],
    training_population: dict,
) -> dict:
    """Label configured requests as validated only when artifact provenance matches."""
    if source["type"] != "configured_universe":
        return source

    artifact_candidates = training_population.get("candidate_tickers", [])
    provenance_matches = (
        source["universe_name"] == training_population.get("universe_name")
        and set(candidates) == set(artifact_candidates)
    )
    result = dict(source)
    result["validation_context"] = (
        "configured_research_universe"
        if provenance_matches
        else "no_historical_validation_claim"
    )
    return result


def _prepare_histories(tickers: list[str], *, now=None) -> dict[str, object]:
    def prepare(ticker):
        try:
            return prepare_completed_ticker_features(
                ticker,
                period=INFERENCE_HISTORY_PERIOD,
                now=now,
            )
        except Exception as error:  # Per-ticker failures are reported, not fatal.
            return error

    with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
        prepared = list(executor.map(prepare, tickers))
    return dict(zip(tickers, prepared))


def _latest_spy_anchor_date(spy_features: pd.DataFrame) -> pd.Timestamp:
    if spy_features.empty or "prediction_date" not in spy_features:
        raise RankedWatchlistDataError("SPY has no completed feature dates.")
    dates = pd.to_datetime(spy_features["prediction_date"], errors="coerce").dropna()
    if dates.empty:
        raise RankedWatchlistDataError("SPY has no valid completed feature dates.")
    return dates.max().normalize()


def _build_synchronized_feature_frame(
    candidates: list[str],
    prepared_histories: dict[str, object],
    preprocessor_features: list[str],
) -> tuple[pd.Timestamp, pd.DataFrame, list[dict]]:
    spy_features = prepared_histories.get(BENCHMARK_TICKER)
    if not isinstance(spy_features, pd.DataFrame):
        raise RankedWatchlistDataError(
            f"SPY benchmark preparation failed: {spy_features}"
        )
    anchor_date = _latest_spy_anchor_date(spy_features)
    spy_anchor_rows = spy_features[
        pd.to_datetime(spy_features["prediction_date"]).dt.normalize() == anchor_date
    ]
    if spy_anchor_rows.empty:
        raise RankedWatchlistDataError(
            f"SPY has no feature row on anchor date {anchor_date.date()}."
        )

    skipped = []
    candidate_rows = []
    for ticker in candidates:
        features = prepared_histories.get(ticker)
        if not isinstance(features, pd.DataFrame):
            skipped.append(
                {"ticker": ticker, "reason": f"data preparation failed: {features}"}
            )
            continue
        exact_date_rows = features[
            pd.to_datetime(features["prediction_date"]).dt.normalize() == anchor_date
        ]
        if exact_date_rows.empty:
            skipped.append(
                {
                    "ticker": ticker,
                    "reason": (
                        "missing completed bar on SPY anchor date "
                        f"{anchor_date.date()}"
                    ),
                }
            )
            continue
        candidate_rows.append(exact_date_rows.iloc[[-1]].copy())

    if candidate_rows:
        synchronized = pd.concat(
            [spy_anchor_rows.iloc[[-1]].copy(), *candidate_rows],
            ignore_index=True,
        )
        synchronized = add_benchmark_relative_momentum(
            synchronized,
            BENCHMARK_TICKER,
        )
        synchronized = synchronized[synchronized["Ticker"] != BENCHMARK_TICKER]
    else:
        synchronized = pd.DataFrame()

    valid_rows = []
    for _, row in synchronized.iterrows():
        ticker = row["Ticker"]
        missing_columns = [
            feature for feature in preprocessor_features if feature not in row.index
        ]
        if missing_columns:
            skipped.append(
                {
                    "ticker": ticker,
                    "reason": f"missing required features: {missing_columns}",
                }
            )
            continue
        numeric = pd.to_numeric(row[preprocessor_features], errors="coerce")
        invalid_features = numeric.index[
            numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
        ].tolist()
        if invalid_features:
            skipped.append(
                {
                    "ticker": ticker,
                    "reason": f"invalid required features: {invalid_features}",
                }
            )
            continue
        valid_rows.append(
            {"Ticker": ticker, **numeric.astype(float).to_dict()}
        )

    if valid_rows:
        feature_frame = pd.DataFrame(valid_rows)
    else:
        feature_frame = pd.DataFrame(columns=["Ticker", *preprocessor_features])
    return anchor_date, feature_frame, skipped


def rank_candidate_universe(
    prediction_days: int,
    *,
    universe: str | None = None,
    tickers: Iterable[str] | None = None,
    now=None,
) -> dict:
    """Return one same-date Rank-NDCG watchlist for a configured or custom universe."""
    _validate_serving_horizon(prediction_days)
    candidates, source, requested_count, skipped = _resolve_candidate_source(
        universe=universe,
        tickers=tickers,
    )
    bundle = _load_rank_ndcg_bundle(prediction_days)
    source = _apply_training_population_context(
        source,
        candidates,
        bundle.manifest["training_population"],
    )

    prepared_histories = _prepare_histories(
        [BENCHMARK_TICKER, *candidates],
        now=now,
    )
    anchor_date, feature_frame, data_skips = _build_synchronized_feature_frame(
        candidates,
        prepared_histories,
        bundle.preprocessor_features,
    )
    skipped.extend(data_skips)
    if len(feature_frame) < 2:
        raise RankedWatchlistDataError(
            "Fewer than 2 valid candidates remain on the shared SPY anchor date "
            f"{anchor_date.date()}; skipped={skipped}."
        )

    ordered_preprocessor_input = feature_frame[bundle.preprocessor_features]
    scaled_values = bundle.data_preparator.scalar.transform(
        ordered_preprocessor_input.to_numpy(dtype=float)
    )
    scaled_features = pd.DataFrame(
        scaled_values,
        columns=bundle.preprocessor_features,
        index=feature_frame.index,
    )
    ranker_input = scaled_features[bundle.ranker_features]
    scores = np.asarray(bundle.ranker.predict(ranker_input), dtype=float)
    if scores.shape != (len(feature_frame),):
        raise RankedWatchlistDataError(
            "Ranker returned an unexpected number of scores."
        )

    scored_rows = []
    for ticker, score in zip(feature_frame["Ticker"], scores):
        if not np.isfinite(score):
            skipped.append(
                {"ticker": ticker, "reason": "ranker returned a non-finite score"}
            )
            continue
        scored_rows.append({"ticker": ticker, "ranker_score": float(score)})
    if len(scored_rows) < 2:
        raise RankedWatchlistDataError(
            "Fewer than 2 candidates received finite ranker scores."
        )

    scored_rows.sort(key=lambda row: (-row["ranker_score"], row["ticker"]))
    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(scored_rows, start=1)
    ]
    return {
        "bundle_id": bundle.manifest["bundle_id"],
        "prediction_days": prediction_days,
        "target_mode": TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
        "as_of_date": anchor_date.date().isoformat(),
        "candidate_source": source,
        "training_population": bundle.manifest["training_population"],
        "requested_count": requested_count,
        "ranked_count": len(ranked_rows),
        "skipped_count": len(skipped),
        "skipped_tickers": skipped,
        "score_semantics": "higher Rank-NDCG score ranks first; not a probability or return",
        "ranked_rows": ranked_rows,
    }
