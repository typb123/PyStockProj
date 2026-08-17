"""Lightweight target-mode and feature-subset training policy."""

from src.features.feature_contract import MODEL_FEATURE_COLUMNS


TARGET_MODE_EXCESS_RETURN = "excess_return"
TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM = "cross_sectional_top_bottom"
TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG = "cross_sectional_rank_ndcg"
TARGET_MODES = (
    TARGET_MODE_EXCESS_RETURN,
    TARGET_MODE_CROSS_SECTIONAL_TOP_BOTTOM,
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
)
WALK_FORWARD_TARGET_MODES = (
    TARGET_MODE_EXCESS_RETURN,
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
)


def validate_target_mode(target_mode: str) -> None:
    """Reject unknown target modes before training begins."""
    if target_mode in TARGET_MODES:
        return
    raise ValueError(
        f"Unsupported target_mode={target_mode!r}. "
        f"Expected one of {', '.join(TARGET_MODES)}."
    )


def validate_walk_forward_target_mode(target_mode: str) -> None:
    """Reject target modes that do not have walk-forward evaluation support."""
    if target_mode in WALK_FORWARD_TARGET_MODES:
        return
    raise ValueError(
        f"Unsupported walk-forward target_mode={target_mode!r}. "
        f"Expected one of {', '.join(WALK_FORWARD_TARGET_MODES)}."
    )


def resolve_model_feature_columns(feature_columns_override=None) -> list[str]:
    """Return validated model feature columns, preserving defaults when omitted."""
    if feature_columns_override is None:
        return list(MODEL_FEATURE_COLUMNS)

    feature_columns = list(feature_columns_override)
    if not feature_columns:
        raise ValueError("feature_columns_override must contain at least one feature.")

    allowed_features = set(MODEL_FEATURE_COLUMNS)
    unknown_features = [
        feature for feature in feature_columns if feature not in allowed_features
    ]
    if unknown_features:
        raise ValueError(
            "feature_columns_override may only include MODEL_FEATURE_COLUMNS. "
            f"Invalid entries: {unknown_features}"
        )

    duplicate_features = sorted(
        {
            feature
            for feature in feature_columns
            if feature_columns.count(feature) > 1
        }
    )
    if duplicate_features:
        raise ValueError(
            f"feature_columns_override contains duplicate features: {duplicate_features}"
        )

    return feature_columns
