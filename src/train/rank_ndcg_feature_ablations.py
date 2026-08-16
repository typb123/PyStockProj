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
    raw_ohlc = {"Open", "High", "Low", "Close"}
    price_level_trend = set(
        [
            "Open",
            "High",
            "Low",
            "Close",
            *groups["trend"],
            "BB_Middle",
            "BB_Upper",
            "BB_Lower",
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_lag_close_26",
        ]
    )
    volume_scale = {"Volume", "vma_10", "vma_20", "obv"}
    minimal_momentum_risk_oscillator = set(
        [
            *groups["absolute_momentum"],
            *groups["spy_relative_momentum"],
            "dailyReturn",
            "volatility",
            "ATR",
            "BB_Std",
            "rsi",
            "stoch_k",
            "stoch_d",
            "macdHistogram",
        ]
    )
    momentum_plus_volatility = set(
        [
            *groups["absolute_momentum"],
            *groups["spy_relative_momentum"],
            "dailyReturn",
            "volatility",
            "ATR",
            "BB_Std",
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
            removed_groups="raw_ohlc",
        ),
        FeatureAblationSpec(
            name="drop_price_level_trend",
            feature_columns=_without(all_features, price_level_trend),
            removed_groups="raw_ohlc,trend,bollinger_levels,ichimoku_levels",
        ),
        FeatureAblationSpec(
            name="drop_volume_scale",
            feature_columns=_without(all_features, volume_scale),
            removed_groups="raw_volume,vma,obv",
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
            feature_columns=_without(all_features, {"macd", "signalLine"}),
            removed_groups="macd,signalLine",
        ),
        FeatureAblationSpec(
            name="minimal_momentum_risk_oscillator",
            feature_columns=_only(all_features, minimal_momentum_risk_oscillator),
            included_groups="momentum,relative_momentum,risk,rsi,stochastic,macdHistogram",
        ),
        FeatureAblationSpec(
            name="momentum_only",
            feature_columns=_only(all_features, set(groups["absolute_momentum"])),
            included_groups="absolute_momentum",
        ),
        FeatureAblationSpec(
            name="momentum_plus_volatility",
            feature_columns=_only(all_features, momentum_plus_volatility),
            included_groups="momentum,relative_momentum,volatility,ATR,BB_Std",
        ),
    ]


def _nested(report: dict, path: tuple[str, ...], default=np.nan):
    current = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _numeric(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _difference(left, right):
    left = _numeric(left)
    right = _numeric(right)
    if math.isnan(left) or math.isnan(right):
        return np.nan
    return left - right


def extract_ablation_result_row(
    report: dict,
    spec: FeatureAblationSpec,
    prediction_days: int,
    period: str,
    universe: str,
) -> dict:
    """Extract a flat CSV-ready metrics row from one train_models report."""
    basket_backtest = report.get("basket_backtest", {})
    diagnostics = report.get("same_date_ranking_diagnostics", {})
    model_diagnostics = diagnostics.get("model", diagnostics)
    momentum_diagnostics = diagnostics.get("momentum", {})

    row = {
        "ablation_name": spec.name,
        "prediction_days": int(prediction_days),
        "period": period,
        "universe": universe,
        "feature_count": len(spec.feature_columns),
        "removed_groups": spec.removed_groups,
        "included_groups": spec.included_groups,
        "model_mean_rank_ic": _numeric(
            _nested(model_diagnostics, ("rank_ic", "mean_rank_ic"))
        ),
        "momentum_mean_rank_ic": _numeric(
            _nested(momentum_diagnostics, ("rank_ic", "mean_rank_ic"))
        ),
        "top_5_realized_top20_rate": _numeric(
            _nested(
                model_diagnostics,
                (
                    "selected_realized_rank_distribution",
                    "top_5",
                    "top_20_realized_rate",
                ),
            )
        ),
        "top_5_realized_bottom20_rate": _numeric(
            _nested(
                model_diagnostics,
                (
                    "selected_realized_rank_distribution",
                    "top_5",
                    "bottom_20_realized_rate",
                ),
            )
        ),
    }

    for top_n in (5, 10, 20):
        bucket = basket_backtest.get(f"top_{top_n}", {})
        model_excess = _numeric(
            _nested(bucket, ("model", "average_basket_excess_return"))
        )
        momentum_excess = _numeric(
            _nested(bucket, ("momentum_baseline", "average_basket_excess_return"))
        )
        row[f"top_{top_n}_model_excess"] = model_excess
        row[f"top_{top_n}_model_minus_momentum"] = _difference(
            model_excess,
            momentum_excess,
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

    columns = [
        "ablation_name",
        "feature_count",
        "top_5_model_excess",
        "top_10_model_excess",
        "top_20_model_excess",
        "top_5_model_minus_momentum",
        "top_10_model_minus_momentum",
        "top_20_model_minus_momentum",
        "model_mean_rank_ic",
        "momentum_mean_rank_ic",
        "top_5_realized_top20_rate",
        "top_5_realized_bottom20_rate",
    ]
    table = pd.DataFrame(results)[columns].copy()
    for column in [
        "top_5_model_excess",
        "top_10_model_excess",
        "top_20_model_excess",
        "top_5_model_minus_momentum",
        "top_10_model_minus_momentum",
        "top_20_model_minus_momentum",
        "top_5_realized_top20_rate",
        "top_5_realized_bottom20_rate",
    ]:
        table[column] = table[column].map(_format_percent)
    for column in ["model_mean_rank_ic", "momentum_mean_rank_ic"]:
        table[column] = table[column].map(
            lambda value: "n/a" if math.isnan(_numeric(value)) else f"{float(value):.4f}"
        )

    return table.to_string(index=False)


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_output_path(prediction_days: int, period: str, universe: str) -> Path:
    """Return the default CSV path for one ablation run."""
    filename = (
        "rank_ndcg_feature_ablations_"
        f"{prediction_days}d_{_safe_filename_part(period)}_"
        f"{_safe_filename_part(universe)}.csv"
    )
    return Path("reports") / filename
