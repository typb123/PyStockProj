"""Run group or leave-one-feature-out Rank-NDCG ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_TRAINING_UNIVERSE,
    PREDICTION_DAYS,
    TRAINING_UNIVERSES,
    get_training_tickers,
)
from src.data.training_data import prepare_data_parallel
from src.train.training_contract import TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG
from src.train.rank_ndcg_feature_ablations import (
    FEATURE_ABLATION_MODES,
    build_paired_ablation_report,
    build_paired_fold_output_path,
    build_output_path,
    build_rank_ndcg_feature_ablation_specs,
    extract_ablation_result_row,
    flatten_paired_ablation_fold_rows,
    format_ablation_table,
)
from src.train.walk_forward_runner import (
    run_walk_forward_models,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Rank-NDCG group or leave-one-feature-out ablations."
    )
    parser.add_argument(
        "--prediction-days",
        type=int,
        default=PREDICTION_DAYS,
        help=f"Prediction horizon in trading days. Defaults to {PREDICTION_DAYS}.",
    )
    parser.add_argument(
        "--period",
        default="10y",
        help='YFinance history period for raw OHLCV fetches. Defaults to "10y".',
    )
    parser.add_argument(
        "--universe",
        choices=sorted(TRAINING_UNIVERSES),
        default=DEFAULT_TRAINING_UNIVERSE,
        help=f"Training universe to fetch. Defaults to {DEFAULT_TRAINING_UNIVERSE}.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=100,
        help="Random baseline trials for Top-N reports.",
    )
    parser.add_argument(
        "--random-trial-workers",
        type=int,
        default=8,
        help="Worker count for Top-N random baseline trials.",
    )
    parser.add_argument(
        "--mode",
        choices=FEATURE_ABLATION_MODES,
        default="groups",
        help="Ablation grid to run. Defaults to the existing group grid.",
    )
    parser.add_argument(
        "--drop-feature",
        action="append",
        default=None,
        help=(
            "Feature to remove in leave-one-out mode. Repeat to select multiple "
            "removals; omit for a full leave-one-out grid."
        ),
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Optional explicit XGBoost seed shared by baseline and ablations.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the raw YFinance OHLCV cache.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Defaults under reports/.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    use_cache = not args.no_cache
    tickers = get_training_tickers(args.universe)
    specs = build_rank_ndcg_feature_ablation_specs(
        mode=args.mode,
        drop_features=args.drop_feature,
    )

    print(
        "Fetching data for "
        f"universe={args.universe}, tickers={len(tickers)}, period={args.period}."
    )
    data = prepare_data_parallel(tickers, period=args.period, use_cache=use_cache)
    if data.empty:
        raise ValueError("No data fetched for ablation run.")

    results = []
    paired_reports = []
    baseline_report = None
    for spec in specs:
        print(
            f"Running {spec.name} "
            f"({len(spec.feature_columns)} features)..."
        )
        report = run_walk_forward_models(
            data.copy(),
            prediction_days=args.prediction_days,
            random_trials=args.random_trials,
            random_trial_workers=args.random_trial_workers,
            target_mode=TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
            feature_columns_override=spec.feature_columns,
            model_seed=args.model_seed,
        )
        paired_report = None
        if spec.name == "all_features":
            if baseline_report is not None:
                raise ValueError("Ablation specs may include all_features only once.")
            baseline_report = report
        else:
            if baseline_report is None:
                raise ValueError("Ablation specs must start with all_features.")
            paired_report = build_paired_ablation_report(
                baseline_report,
                report,
                ablation_name=spec.name,
            )
            paired_reports.append(paired_report)
        results.append(
            extract_ablation_result_row(
                report,
                spec,
                prediction_days=args.prediction_days,
                period=args.period,
                universe=args.universe,
                paired_report=paired_report,
            )
        )

    print(format_ablation_table(results))
    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(
            args.prediction_days,
            args.period,
            args.universe,
            mode=args.mode,
            model_seed=args.model_seed,
            drop_features=args.drop_feature,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved ablation results to {output_path}")
    paired_fold_rows = flatten_paired_ablation_fold_rows(paired_reports)
    if paired_fold_rows:
        paired_fold_output_path = build_paired_fold_output_path(output_path)
        pd.DataFrame(paired_fold_rows).to_csv(paired_fold_output_path, index=False)
        print(f"Saved paired fold results to {paired_fold_output_path}")
    return results


if __name__ == "__main__":
    main()
