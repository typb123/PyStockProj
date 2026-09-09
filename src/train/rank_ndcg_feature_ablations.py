"""Feature-subset helpers for Rank-NDCG ablation experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_contract import MODEL_FEATURE_COLUMNS, MODEL_FEATURE_GROUPS


@dataclass(frozen=True)
class FeatureAblationSpec:
    """Named model feature subset for one Rank-NDCG ablation run."""

    name: str
    feature_columns: list[str]
    removed_groups: str = ""
    included_groups: str = ""


def _without(features: list[str], removed_features: set[str]) -> list[str]:
    return [feature for feature in features if feature not in removed_features]


def _only(features: list[str], included_features: set[str]) -> list[str]:
    return [feature for feature in features if feature in included_features]


def build_rank_ndcg_feature_ablation_specs() -> list[FeatureAblationSpec]:
    """Build the initial controlled feature-group ablation grid."""
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
        FeatureAblationSpec(
            name="all_features",
            feature_columns=all_features,
            included_groups="all",
        ),
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


def extract_ablation_result_row(
    report: dict,
    spec: FeatureAblationSpec,
    prediction_days: int,
    period: str,
    universe: str,
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
        "feature_count": len(spec.feature_columns),
        "removed_groups": spec.removed_groups,
        "included_groups": spec.included_groups,
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
    }
    table = pd.DataFrame(results).reindex(columns=display_columns).copy()
    for column in list(display_columns)[2:]:
        table[column] = table[column].map(_format_percent)

    return table.rename(columns=display_columns).to_string(index=False)


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_output_path(prediction_days: int, period: str, universe: str) -> Path:
    """Return the default CSV path for one ablation run."""
    filename = (
        "rank_ndcg_walk_forward_feature_ablations_"
        f"{prediction_days}d_{_safe_filename_part(period)}_"
        f"{_safe_filename_part(universe)}.csv"
    )
    return Path("reports") / filename
