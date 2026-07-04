"""Same-date ranking diagnostics for model scores."""

import numpy as np
import pandas as pd


REQUIRED_RANKING_DIAGNOSTIC_COLUMNS = [
    "prediction_date",
    "excess_forward_return",
    "excess_return_rank_pct_by_date",
]


def build_same_date_ranking_diagnostics(
    split_metadata,
    model_scores,
    score_column="model_score",
    top_n_values=(5, 10, 20),
):
    """Measure whether model scores rank same-date realized excess returns."""
    metadata = split_metadata.copy()
    missing_columns = [
        column
        for column in REQUIRED_RANKING_DIAGNOSTIC_COLUMNS
        if column not in metadata.columns
    ]
    if missing_columns:
        reason = f"Missing required columns: {missing_columns}"
        return _unavailable_ranking_diagnostics(score_column, top_n_values, reason)

    scores = np.asarray(model_scores, dtype=float)
    if len(scores) != len(metadata):
        reason = (
            f"model_scores length {len(scores)} does not match metadata length "
            f"{len(metadata)}"
        )
        return _unavailable_ranking_diagnostics(score_column, top_n_values, reason)

    metadata[score_column] = scores
    if "Ticker" in metadata.columns:
        metadata = metadata[metadata["Ticker"] != "SPY"].copy()
    rank_ic_summary = _build_rank_ic_summary(metadata, score_column)
    realized_rank_distribution = _build_selected_realized_rank_distribution(
        metadata,
        score_column,
        top_n_values,
    )

    return {
        "available": bool(
            rank_ic_summary.get("available")
            or realized_rank_distribution.get("available")
        ),
        "score_column": score_column,
        "rank_ic": rank_ic_summary,
        "selected_realized_rank_distribution": realized_rank_distribution,
    }


def build_same_date_ranking_diagnostics_with_baselines(
    split_metadata,
    model_scores,
    model_score_column="model_score",
    prediction_days=None,
    top_n_values=(5, 10, 20),
):
    """Build model and strict horizon-matched momentum ranking diagnostics."""
    model_diagnostics = build_same_date_ranking_diagnostics(
        split_metadata,
        model_scores,
        score_column=model_score_column,
        top_n_values=top_n_values,
    )
    combined_diagnostics = dict(model_diagnostics)
    if prediction_days is None:
        reason = "prediction_days is required for horizon-matched baseline diagnostics."
        combined_diagnostics["model"] = model_diagnostics
        combined_diagnostics["momentum"] = _unavailable_ranking_diagnostics(
            "momentum_<prediction_days>d",
            top_n_values,
            reason,
        )
        combined_diagnostics["relative_momentum"] = _unavailable_ranking_diagnostics(
            "relative_momentum_<prediction_days>d",
            top_n_values,
            reason,
        )
        return combined_diagnostics

    momentum_column = f"momentum_{int(prediction_days)}d"
    relative_momentum_column = f"relative_momentum_{int(prediction_days)}d"

    combined_diagnostics["model"] = model_diagnostics
    combined_diagnostics["momentum"] = _build_baseline_ranking_diagnostics(
        split_metadata,
        momentum_column,
        top_n_values,
        baseline_label="Momentum",
    )
    combined_diagnostics["relative_momentum"] = _build_baseline_ranking_diagnostics(
        split_metadata,
        relative_momentum_column,
        top_n_values,
        baseline_label="Relative momentum",
    )
    return combined_diagnostics


def _build_baseline_ranking_diagnostics(
    split_metadata,
    score_column,
    top_n_values,
    baseline_label,
):
    if score_column not in split_metadata.columns:
        reason = (
            f"{baseline_label} score column '{score_column}' is not present in "
            "split_metadata."
        )
        return _unavailable_ranking_diagnostics(score_column, top_n_values, reason)

    return build_same_date_ranking_diagnostics(
        split_metadata,
        split_metadata[score_column].to_numpy(),
        score_column=score_column,
        top_n_values=top_n_values,
    )


def _unavailable_ranking_diagnostics(score_column, top_n_values, reason):
    return {
        "available": False,
        "score_column": score_column,
        "reason": reason,
        "rank_ic": {
            "available": False,
            "date_count": 0,
            "mean_rank_ic": np.nan,
            "median_rank_ic": np.nan,
            "rank_ic_positive_rate": np.nan,
            "reason": reason,
        },
        "selected_realized_rank_distribution": {
            "available": False,
            "reason": reason,
            **{
                f"top_{top_n}": _empty_realized_rank_bucket()
                for top_n in top_n_values
            },
        },
    }


def _build_rank_ic_summary(metadata, score_column):
    rank_ic_values = []
    for _, date_group in metadata.groupby("prediction_date", sort=True):
        usable_group = date_group[
            [score_column, "excess_forward_return"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if len(usable_group) < 2:
            continue
        if (
            usable_group[score_column].nunique(dropna=True) < 2
            or usable_group["excess_forward_return"].nunique(dropna=True) < 2
        ):
            continue

        rank_ic = usable_group[score_column].corr(
            usable_group["excess_forward_return"],
            method="spearman",
        )
        if pd.notna(rank_ic) and np.isfinite(rank_ic):
            rank_ic_values.append(float(rank_ic))

    if not rank_ic_values:
        return {
            "available": False,
            "date_count": 0,
            "mean_rank_ic": np.nan,
            "median_rank_ic": np.nan,
            "rank_ic_positive_rate": np.nan,
            "reason": "No prediction dates had at least two non-constant usable rows.",
        }

    rank_ic_values = np.asarray(rank_ic_values, dtype=float)
    return {
        "available": True,
        "date_count": int(len(rank_ic_values)),
        "mean_rank_ic": float(np.mean(rank_ic_values)),
        "median_rank_ic": float(np.median(rank_ic_values)),
        "rank_ic_positive_rate": float(np.mean(rank_ic_values > 0)),
    }


def _build_selected_realized_rank_distribution(
    metadata,
    score_column,
    top_n_values,
):
    usable_metadata = metadata.dropna(
        subset=[score_column, "excess_return_rank_pct_by_date"]
    ).copy()
    grouped_metadata = list(usable_metadata.groupby("prediction_date", sort=True))
    if not grouped_metadata:
        reason = "No selected rows had usable realized rank percentiles."
        return {
            "available": False,
            "reason": reason,
            **{
                f"top_{top_n}": _empty_realized_rank_bucket()
                for top_n in top_n_values
            },
        }

    bucket_reports = {}
    any_selected_rows = False
    for top_n in top_n_values:
        selected_groups = []
        for _, date_group in grouped_metadata:
            selected_count = min(int(top_n), len(date_group))
            if selected_count <= 0:
                continue
            selected_groups.append(
                date_group.sort_values(score_column, ascending=False).head(
                    selected_count
                )
            )

        selected_rank_pct = (
            pd.concat(selected_groups, ignore_index=True)[
                "excess_return_rank_pct_by_date"
            ]
            if selected_groups
            else pd.Series(dtype=float)
        )
        selected_rank_pct = selected_rank_pct.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        bucket_reports[f"top_{top_n}"] = _realized_rank_bucket(selected_rank_pct)
        any_selected_rows = any_selected_rows or len(selected_rank_pct) > 0

    if not any_selected_rows:
        return {
            "available": False,
            "reason": "No selected rows had usable realized rank percentiles.",
            **bucket_reports,
        }

    return {
        "available": True,
        **bucket_reports,
    }


def _realized_rank_bucket(rank_pct):
    if len(rank_pct) == 0:
        return _empty_realized_rank_bucket()

    rank_pct = pd.Series(rank_pct, dtype=float)
    selected_count = int(len(rank_pct))
    top_mask = rank_pct >= 0.80
    bottom_mask = rank_pct <= 0.20
    middle_mask = (rank_pct > 0.20) & (rank_pct < 0.80)
    return {
        "selected_count": selected_count,
        "top_20_realized_rate": float(top_mask.mean()),
        "middle_60_realized_rate": float(middle_mask.mean()),
        "bottom_20_realized_rate": float(bottom_mask.mean()),
        "average_realized_rank_pct": float(rank_pct.mean()),
    }


def _empty_realized_rank_bucket():
    return {
        "selected_count": 0,
        "top_20_realized_rate": np.nan,
        "middle_60_realized_rate": np.nan,
        "bottom_20_realized_rate": np.nan,
        "average_realized_rank_pct": np.nan,
    }
