"""Feature-subset helpers for Rank-NDCG ablation experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import math
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from src.features.feature_contract import MODEL_FEATURE_COLUMNS, MODEL_FEATURE_GROUPS
from src.train.walk_forward_runner import (
    run_rank_ndcg_walk_forward_from_context,
    run_walk_forward_models,
)


FEATURE_ABLATION_MODE_GROUPS = "groups"
FEATURE_ABLATION_MODE_LEAVE_ONE_OUT = "leave-one-out"
FEATURE_ABLATION_MODES = (
    FEATURE_ABLATION_MODE_GROUPS,
    FEATURE_ABLATION_MODE_LEAVE_ONE_OUT,
)
TOP_N_VALUES = (5, 10, 20)
PAIRED_FOLD_METRICS = (
    "model_excess",
    "model_minus_momentum",
    "model_minus_universe",
)
PAIRED_AGGREGATE_METRICS = (
    "average_model_excess",
    "average_model_minus_momentum",
    "fold_win_rate_vs_momentum",
    "average_model_minus_universe",
    "fold_win_rate_vs_universe",
)
DEFAULT_CPU_BUDGET = 24
DEFAULT_BENCHMARK_CPU_CONFIGURATIONS = (
    (1, 24),
    (2, 12),
    (3, 8),
    (4, 6),
    (6, 4),
)


@dataclass(frozen=True)
class AblationParallelismConfig:
    """Explicit CPU allocation for independent ablation variants."""

    outer_workers: int
    xgb_threads: int
    cpu_budget: int


@dataclass
class AblationRun:
    """One completed ablation run, retaining its deterministic input position."""

    index: int
    spec: "FeatureAblationSpec"
    report: dict
    elapsed_seconds: float
    native_threadpools: list[dict] | None = None


_ABLATION_WORKER_DATA: pd.DataFrame | None = None
_ABLATION_WORKER_CONTEXT: dict | None = None


@dataclass(frozen=True)
class FeatureAblationSpec:
    """Named model feature subset for one Rank-NDCG ablation run."""

    name: str
    feature_columns: list[str]
    removed_groups: str = ""
    included_groups: str = ""
    removed_features: tuple[str, ...] = ()


def validate_ablation_parallelism(
    outer_workers: int,
    xgb_threads: int,
    cpu_budget: int,
) -> AblationParallelismConfig:
    """Validate a non-oversubscribed process/thread allocation."""
    values = {
        "outer_workers": outer_workers,
        "xgb_threads": xgb_threads,
        "cpu_budget": cpu_budget,
    }
    normalized_values = {}
    for name, value in values.items():
        try:
            normalized_value = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a positive integer.") from error
        if normalized_value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        normalized_values[name] = normalized_value

    if (
        normalized_values["outer_workers"] * normalized_values["xgb_threads"]
        > normalized_values["cpu_budget"]
    ):
        raise ValueError(
            "outer_workers * xgb_threads must not exceed cpu_budget."
        )
    return AblationParallelismConfig(**normalized_values)


def build_benchmark_parallelism_configs(
    cpu_budget: int,
) -> list[AblationParallelismConfig]:
    """Build the standard CPU benchmark matrix after budget validation."""
    return [
        validate_ablation_parallelism(outer_workers, xgb_threads, cpu_budget)
        for outer_workers, xgb_threads in DEFAULT_BENCHMARK_CPU_CONFIGURATIONS
    ]


def _initialize_ablation_worker(data: pd.DataFrame | None, prepared_context=None) -> None:
    """Store one fetched frame or frozen Rank-NDCG context per worker."""
    global _ABLATION_WORKER_DATA, _ABLATION_WORKER_CONTEXT
    _ABLATION_WORKER_DATA = data
    _ABLATION_WORKER_CONTEXT = prepared_context


def _run_one_ablation(
    data: pd.DataFrame | None,
    index: int,
    spec: FeatureAblationSpec,
    walk_forward_kwargs: dict,
    runner: Callable[..., dict],
    blas_threads: int | None = None,
    collect_worker_diagnostics: bool = False,
    prepared_context: dict | None = None,
) -> AblationRun:
    """Run one independent variant and record its complete wall-clock duration."""
    started_at = time.perf_counter()
    blas_context = (
        threadpool_limits(limits=blas_threads, user_api="blas")
        if blas_threads is not None
        else nullcontext()
    )
    with blas_context:
        if prepared_context is not None:
            report = run_rank_ndcg_walk_forward_from_context(
                prepared_context,
                feature_columns_override=spec.feature_columns,
                **walk_forward_kwargs,
            )
        else:
            if data is None:
                raise RuntimeError("Ablation run requires data when no context is supplied.")
            report = runner(
                data.copy(),
                feature_columns_override=spec.feature_columns,
                **walk_forward_kwargs,
            )
        native_threadpools = (
            _native_threadpool_summary() if collect_worker_diagnostics else None
        )
    return AblationRun(
        index=index,
        spec=spec,
        report=report,
        elapsed_seconds=time.perf_counter() - started_at,
        native_threadpools=native_threadpools,
    )


def _native_threadpool_summary() -> list[dict]:
    """Return serializable native-pool details for an opt-in spawn smoke check."""
    return [
        {
            "user_api": pool.get("user_api"),
            "internal_api": pool.get("internal_api"),
            "prefix": pool.get("prefix"),
            "num_threads": pool.get("num_threads"),
        }
        for pool in threadpool_info()
    ]


def _run_one_ablation_worker(task) -> AblationRun:
    """Run a task using the process-local frame or frozen context."""
    if _ABLATION_WORKER_DATA is None and _ABLATION_WORKER_CONTEXT is None:
        raise RuntimeError("Ablation worker was not initialized with run state.")
    (
        index,
        spec,
        walk_forward_kwargs,
        blas_threads,
        collect_worker_diagnostics,
    ) = task
    return _run_one_ablation(
        _ABLATION_WORKER_DATA,
        index,
        spec,
        walk_forward_kwargs,
        run_walk_forward_models,
        blas_threads,
        collect_worker_diagnostics,
        _ABLATION_WORKER_CONTEXT,
    )


def run_rank_ndcg_ablation_specs(
    data: pd.DataFrame,
    specs: Iterable[FeatureAblationSpec],
    *,
    outer_workers: int,
    xgb_threads: int,
    cpu_budget: int,
    walk_forward_kwargs: dict,
    serial_runner: Callable[..., dict] | None = None,
    collect_worker_diagnostics: bool = False,
    prepared_context: dict | None = None,
) -> list[AblationRun]:
    """Run ordered ablation specs serially or in a bounded outer process pool.

    Outer workers receive either fetched data or one prepared Rank-NDCG context
    through their initializer, never fetch independently, and force Top-N random
    trial work to serial execution to avoid a nested process-pool beneath
    multi-threaded XGBoost fitting.
    """
    parallelism = validate_ablation_parallelism(
        outer_workers,
        xgb_threads,
        cpu_budget,
    )
    ordered_specs = list(specs)
    if not ordered_specs:
        return []

    worker_kwargs = dict(walk_forward_kwargs)
    worker_kwargs["xgb_threads"] = parallelism.xgb_threads
    blas_threads = 1 if parallelism.outer_workers > 1 else None
    if parallelism.outer_workers > 1:
        worker_kwargs["random_trial_workers"] = 1

    indexed_specs = list(enumerate(ordered_specs))
    if parallelism.outer_workers == 1:
        runner = serial_runner or run_walk_forward_models
        return [
            _run_one_ablation(
                data,
                index,
                spec,
                worker_kwargs,
                runner,
                blas_threads,
                collect_worker_diagnostics,
                prepared_context,
            )
            for index, spec in indexed_specs
        ]

    tasks = [
        (
            index,
            spec,
            worker_kwargs,
            blas_threads,
            collect_worker_diagnostics,
        )
        for index, spec in indexed_specs
    ]
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=parallelism.outer_workers,
        mp_context=mp_context,
        initializer=_initialize_ablation_worker,
        initargs=(None if prepared_context is not None else data, prepared_context),
    ) as executor:
        runs = list(executor.map(_run_one_ablation_worker, tasks))
    return sorted(runs, key=lambda run: run.index)


def _without(features: list[str], removed_features: set[str]) -> list[str]:
    return [feature for feature in features if feature not in removed_features]


def _only(features: list[str], included_features: set[str]) -> list[str]:
    return [feature for feature in features if feature in included_features]


def _all_features_baseline_spec() -> FeatureAblationSpec:
    return FeatureAblationSpec(
        name="all_features",
        feature_columns=list(MODEL_FEATURE_COLUMNS),
        included_groups="all",
    )


def _resolve_requested_drop_features(
    drop_features: Iterable[str] | None,
) -> list[str]:
    """Validate requested LOFO removals and order them by the feature contract."""
    all_features = list(MODEL_FEATURE_COLUMNS)
    if drop_features is None:
        return all_features

    requested_features = list(drop_features)
    unknown_features = sorted(set(requested_features).difference(all_features))
    if unknown_features:
        raise ValueError(
            "drop_features may only include MODEL_FEATURE_COLUMNS. "
            f"Invalid entries: {unknown_features}"
        )
    duplicate_features = sorted(
        {
            feature
            for feature in requested_features
            if requested_features.count(feature) > 1
        }
    )
    if duplicate_features:
        raise ValueError(
            f"drop_features contains duplicates: {duplicate_features}"
        )
    requested_set = set(requested_features)
    return [feature for feature in all_features if feature in requested_set]


def build_leave_one_feature_out_ablation_specs(
    drop_features: Iterable[str] | None = None,
) -> list[FeatureAblationSpec]:
    """Build a full baseline plus one ordered, single-feature removal per spec.

    Passing ``drop_features`` limits the variants without changing the baseline.
    An omitted selection generates all removals in feature-contract order.
    """
    all_features = list(MODEL_FEATURE_COLUMNS)
    selected_features = _resolve_requested_drop_features(drop_features)
    specs = [_all_features_baseline_spec()]
    specs.extend(
        FeatureAblationSpec(
            name=f"drop__{feature}",
            feature_columns=_without(all_features, {feature}),
            removed_features=(feature,),
        )
        for feature in selected_features
    )
    return specs


def build_rank_ndcg_feature_ablation_specs(
    mode: str = FEATURE_ABLATION_MODE_GROUPS,
    drop_features: Iterable[str] | None = None,
) -> list[FeatureAblationSpec]:
    """Build the initial controlled feature-group ablation grid."""
    if mode not in FEATURE_ABLATION_MODES:
        raise ValueError(
            f"Unsupported ablation mode {mode!r}. "
            f"Expected one of {', '.join(FEATURE_ABLATION_MODES)}."
        )
    if mode == FEATURE_ABLATION_MODE_LEAVE_ONE_OUT:
        return build_leave_one_feature_out_ablation_specs(drop_features)
    if drop_features is not None:
        raise ValueError("drop_features is only supported in leave-one-out mode.")

    all_features = list(MODEL_FEATURE_COLUMNS)
    groups = MODEL_FEATURE_GROUPS
    raw_ohlc = {"open_to_close", "high_to_close", "low_to_close"}
    price_level_trend = set(
        [
            *raw_ohlc,
            *groups["trend"],
            "bb_middle_to_close",
            "bb_upper_to_close",
            "bb_lower_to_close",
            "tenkan_sen_to_close",
            "kijun_sen_to_close",
            "senkou_span_a_to_close",
            "senkou_span_b_to_close",
            "chikou_lag_close_26_to_close",
        ]
    )
    volume_scale = {"Volume", "vma_10", "vma_20", "rolling_signed_volume_20d"}
    minimal_momentum_risk_oscillator = set(
        [
            *groups["absolute_momentum"],
            *groups["spy_relative_momentum"],
            "dailyReturn",
            "volatility",
            "atr_to_close",
            "bb_std_to_close",
            "rsi",
            "stoch_k",
            "stoch_d",
            "macd_histogram_to_close",
        ]
    )
    momentum_plus_volatility = set(
        [
            *groups["absolute_momentum"],
            *groups["spy_relative_momentum"],
            "dailyReturn",
            "volatility",
            "atr_to_close",
            "bb_std_to_close",
        ]
    )

    return [
        _all_features_baseline_spec(),
        FeatureAblationSpec(
            name="drop_raw_ohlc",
            feature_columns=_without(all_features, raw_ohlc),
            removed_groups="intraday_price_relative",
        ),
        FeatureAblationSpec(
            name="drop_price_level_trend",
            feature_columns=_without(all_features, price_level_trend),
            removed_groups="price_relative,trend,bollinger_distances,ichimoku_distances",
        ),
        FeatureAblationSpec(
            name="drop_volume_scale",
            feature_columns=_without(all_features, volume_scale),
            removed_groups="raw_volume,vma,rolling_signed_volume",
        ),
        FeatureAblationSpec(
            name="drop_ichimoku",
            feature_columns=_without(all_features, set(groups["ichimoku"])),
            removed_groups="ichimoku",
        ),
        FeatureAblationSpec(
            name="drop_relative_momentum",
            feature_columns=_without(
                all_features,
                set(groups["spy_relative_momentum"]),
            ),
            removed_groups="spy_relative_momentum",
        ),
        FeatureAblationSpec(
            name="drop_macd_redundant",
            feature_columns=_without(
                all_features,
                {"macd_to_close", "signal_line_to_close"},
            ),
            removed_groups="macd_to_close,signal_line_to_close",
        ),
        FeatureAblationSpec(
            name="minimal_momentum_risk_oscillator",
            feature_columns=_only(all_features, minimal_momentum_risk_oscillator),
            included_groups=(
                "momentum,relative_momentum,risk,rsi,stochastic,"
                "macd_histogram_to_close"
            ),
        ),
        FeatureAblationSpec(
            name="momentum_only",
            feature_columns=_only(all_features, set(groups["absolute_momentum"])),
            included_groups="absolute_momentum",
        ),
        FeatureAblationSpec(
            name="momentum_plus_volatility",
            feature_columns=_only(all_features, momentum_plus_volatility),
            included_groups=(
                "momentum,relative_momentum,volatility,atr_to_close,"
                "bb_std_to_close"
            ),
        ),
    ]


def _numeric(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def assert_matching_ablation_population(
    baseline_report: dict,
    ablation_report: dict,
) -> None:
    """Fail when two reports did not use the same eligible walk-forward rows."""
    baseline_identity = baseline_report.get("population_identity")
    ablation_identity = ablation_report.get("population_identity")
    if baseline_identity is None or ablation_identity is None:
        raise ValueError(
            "Baseline and ablation reports must include population_identity."
        )
    if baseline_identity != ablation_identity:
        raise ValueError(
            "Baseline and ablation population identities differ. "
            "Feature-ablation comparisons require identical eligible rows, "
            "labels, folds, and embargoed split membership."
        )


def _paired_metric_delta(ablated_value, baseline_value):
    """Return ablated minus baseline, preserving unavailable values as NaN."""
    ablated_value = _numeric(ablated_value)
    baseline_value = _numeric(baseline_value)
    if math.isnan(ablated_value) or math.isnan(baseline_value):
        return np.nan
    return ablated_value - baseline_value


def _paired_fold_win_loss_tie(deltas: list[float]) -> dict:
    """Count ablation wins/losses/ties using the ablated-minus-baseline delta."""
    finite_deltas = [delta for delta in deltas if not math.isnan(_numeric(delta))]
    return {
        "ablation_win_count": int(sum(delta > 0.0 for delta in finite_deltas)),
        "baseline_win_count": int(sum(delta < 0.0 for delta in finite_deltas)),
        "tie_count": int(sum(delta == 0.0 for delta in finite_deltas)),
        "evaluated_fold_count": int(len(finite_deltas)),
    }


def build_paired_ablation_report(
    baseline_report: dict,
    ablation_report: dict,
    *,
    baseline_name: str = "all_features",
    ablation_name: str,
) -> dict:
    """Pair an ablation against the full baseline without collapsing fold data.

    Every delta follows ``ablated - baseline``: positive values mean removing
    the feature improved the measured metric, and negative values mean it hurt.
    """
    assert_matching_ablation_population(baseline_report, ablation_report)

    baseline_folds = {
        int(fold["fold_index"]): fold for fold in baseline_report.get("folds", [])
    }
    ablation_folds = {
        int(fold["fold_index"]): fold for fold in ablation_report.get("folds", [])
    }
    if baseline_folds.keys() != ablation_folds.keys():
        raise ValueError("Baseline and ablation reports have different fold indexes.")

    paired_top_n = {}
    for top_n in TOP_N_VALUES:
        bucket_name = f"top_{top_n}"
        paired_folds = []
        metric_deltas = {metric_name: [] for metric_name in PAIRED_FOLD_METRICS}
        for fold_index in sorted(baseline_folds):
            baseline_metrics = baseline_folds[fold_index].get("top_n", {}).get(
                bucket_name,
                {},
            )
            ablation_metrics = ablation_folds[fold_index].get("top_n", {}).get(
                bucket_name,
                {},
            )
            deltas = {
                metric_name: _paired_metric_delta(
                    ablation_metrics.get(metric_name),
                    baseline_metrics.get(metric_name),
                )
                for metric_name in PAIRED_FOLD_METRICS
            }
            for metric_name, delta in deltas.items():
                metric_deltas[metric_name].append(delta)
            paired_folds.append(
                {
                    "fold_index": fold_index,
                    "baseline": {
                        metric_name: _numeric(baseline_metrics.get(metric_name))
                        for metric_name in PAIRED_FOLD_METRICS
                    },
                    "ablated": {
                        metric_name: _numeric(ablation_metrics.get(metric_name))
                        for metric_name in PAIRED_FOLD_METRICS
                    },
                    "delta": deltas,
                }
            )

        baseline_aggregate = baseline_report.get("aggregate", {}).get(
            "top_n",
            {},
        ).get(bucket_name, {})
        ablation_aggregate = ablation_report.get("aggregate", {}).get(
            "top_n",
            {},
        ).get(bucket_name, {})
        paired_top_n[bucket_name] = {
            "baseline": {
                metric_name: _numeric(baseline_aggregate.get(metric_name))
                for metric_name in PAIRED_AGGREGATE_METRICS
            },
            "ablated": {
                metric_name: _numeric(ablation_aggregate.get(metric_name))
                for metric_name in PAIRED_AGGREGATE_METRICS
            },
            "delta": {
                metric_name: _paired_metric_delta(
                    ablation_aggregate.get(metric_name),
                    baseline_aggregate.get(metric_name),
                )
                for metric_name in PAIRED_AGGREGATE_METRICS
            },
            "fold_win_loss_tie": {
                metric_name: _paired_fold_win_loss_tie(deltas)
                for metric_name, deltas in metric_deltas.items()
            },
            "folds": paired_folds,
        }

    return {
        "baseline_name": baseline_name,
        "ablation_name": ablation_name,
        "delta_convention": "ablated_minus_baseline",
        "population_identity": baseline_report["population_identity"],
        "top_n": paired_top_n,
    }


def flatten_paired_ablation_fold_rows(paired_reports: Iterable[dict]) -> list[dict]:
    """Flatten paired per-fold metrics for an inspectable companion CSV."""
    rows = []
    for paired_report in paired_reports:
        for top_n in TOP_N_VALUES:
            bucket_name = f"top_{top_n}"
            for paired_fold in paired_report.get("top_n", {}).get(
                bucket_name,
                {},
            ).get("folds", []):
                row = {
                    "baseline_name": paired_report["baseline_name"],
                    "ablation_name": paired_report["ablation_name"],
                    "delta_convention": paired_report["delta_convention"],
                    "top_n": top_n,
                    "fold_index": paired_fold["fold_index"],
                }
                for metric_name in PAIRED_FOLD_METRICS:
                    row[f"baseline_{metric_name}"] = paired_fold["baseline"][
                        metric_name
                    ]
                    row[f"ablated_{metric_name}"] = paired_fold["ablated"][
                        metric_name
                    ]
                    row[f"delta_{metric_name}"] = paired_fold["delta"][metric_name]
                rows.append(row)
    return rows


def extract_ablation_result_row(
    report: dict,
    spec: FeatureAblationSpec,
    prediction_days: int,
    period: str,
    universe: str,
    paired_report: dict | None = None,
) -> dict:
    """Extract a flat CSV-ready row from one walk-forward aggregate report."""
    aggregate = report.get("aggregate", {})
    top_n_summary = aggregate.get("top_n", {})
    fold_count = aggregate.get("fold_count", 0)
    try:
        evaluated_fold_count = int(fold_count)
    except (TypeError, ValueError):
        evaluated_fold_count = 0

    row = {
        "ablation_name": spec.name,
        "prediction_days": int(prediction_days),
        "period": period,
        "universe": universe,
        "model_seed": report.get("model_seed"),
        "feature_count": len(spec.feature_columns),
        "removed_groups": spec.removed_groups,
        "included_groups": spec.included_groups,
        "removed_features": ",".join(spec.removed_features),
        "evaluated_fold_count": evaluated_fold_count,
    }

    for top_n in (5, 10, 20):
        bucket = top_n_summary.get(f"top_{top_n}", {})
        for metric_name in (
            "average_model_excess",
            "average_model_minus_momentum",
            "fold_win_rate_vs_momentum",
            "average_model_minus_universe",
            "fold_win_rate_vs_universe",
        ):
            row[f"top_{top_n}_{metric_name}"] = _numeric(
                bucket.get(metric_name)
            )

    if paired_report is not None:
        row["comparison_baseline_name"] = paired_report["baseline_name"]
        row["delta_convention"] = paired_report["delta_convention"]
        for top_n in TOP_N_VALUES:
            bucket_name = f"top_{top_n}"
            paired_bucket = paired_report.get("top_n", {}).get(bucket_name, {})
            for metric_name in PAIRED_AGGREGATE_METRICS:
                row[f"baseline_top_{top_n}_{metric_name}"] = _numeric(
                    paired_bucket.get("baseline", {}).get(metric_name)
                )
                row[f"delta_top_{top_n}_{metric_name}"] = _numeric(
                    paired_bucket.get("delta", {}).get(metric_name)
                )
            for metric_name, counts in paired_bucket.get(
                "fold_win_loss_tie",
                {},
            ).items():
                for count_name, count in counts.items():
                    row[
                        f"top_{top_n}_{metric_name}_{count_name}"
                    ] = count

    return row


def _format_percent(value):
    value = _numeric(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.2%}"


def format_ablation_table(results: list[dict]) -> str:
    """Format a compact comparison table for stdout."""
    if not results:
        return "No ablation results."

    display_columns = {
        "ablation_name": "ablation",
        "feature_count": "features",
        "top_5_average_model_excess": "top5_avg_excess",
        "top_10_average_model_excess": "top10_avg_excess",
        "top_5_average_model_minus_momentum": "top5_avg_minus_mom",
        "top_10_average_model_minus_momentum": "top10_avg_minus_mom",
        "top_5_fold_win_rate_vs_momentum": "top5_win_vs_mom",
        "top_10_fold_win_rate_vs_momentum": "top10_win_vs_mom",
        "delta_top_5_average_model_excess": "top5_delta_excess",
        "delta_top_10_average_model_excess": "top10_delta_excess",
    }
    table = pd.DataFrame(results).reindex(columns=display_columns).copy()
    for column in list(display_columns)[2:]:
        table[column] = table[column].map(_format_percent)

    return table.rename(columns=display_columns).to_string(index=False)


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _selection_filename_part(drop_features: Iterable[str] | None) -> str:
    """Return a concise, contract-ordered identifier for focused LOFO runs."""
    if drop_features is None:
        return ""

    requested_set = set(drop_features)
    selected_features = [
        feature for feature in MODEL_FEATURE_COLUMNS if feature in requested_set
    ]
    if len(selected_features) == 1:
        return f"_drop_{_safe_filename_part(selected_features[0])}"
    if not selected_features:
        return ""

    selection_digest = hashlib.sha256(
        "|".join(selected_features).encode("utf-8")
    ).hexdigest()[:8]
    return f"_drops_{len(selected_features)}_{selection_digest}"


def build_output_path(
    prediction_days: int,
    period: str,
    universe: str,
    *,
    mode: str = FEATURE_ABLATION_MODE_GROUPS,
    model_seed: int | None = None,
    drop_features: Iterable[str] | None = None,
) -> Path:
    """Return a collision-resistant default CSV path for one ablation run."""
    seed_part = "" if model_seed is None else f"_seed_{int(model_seed)}"
    selection_part = (
        _selection_filename_part(drop_features)
        if mode == FEATURE_ABLATION_MODE_LEAVE_ONE_OUT
        else ""
    )
    filename = (
        "rank_ndcg_walk_forward_feature_ablations_"
        f"{prediction_days}d_{_safe_filename_part(period)}_"
        f"{_safe_filename_part(universe)}_{_safe_filename_part(mode)}"
        f"{selection_part}{seed_part}.csv"
    )
    return Path("reports") / filename


def build_paired_fold_output_path(output_path: Path) -> Path:
    """Return the companion CSV path that keeps paired fold metrics intact."""
    return output_path.with_name(f"{output_path.stem}_paired_folds.csv")
