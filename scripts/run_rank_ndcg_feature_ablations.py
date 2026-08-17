"""Run feature-group ablations for the cross-sectional Rank-NDCG model."""

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
from src.train.rank_ndcg_feature_ablations import (
    build_output_path,
    build_rank_ndcg_feature_ablation_specs,
    extract_ablation_result_row,
    format_ablation_table,
)
from src.train.trainer import (
    TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
    prepare_data_parallel,
    run_walk_forward_models,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Rank-NDCG feature-group ablations."
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
    specs = build_rank_ndcg_feature_ablation_specs()

    print(
        "Fetching data for "
        f"universe={args.universe}, tickers={len(tickers)}, period={args.period}."
    )
    data = prepare_data_parallel(tickers, period=args.period, use_cache=use_cache)
    if data.empty:
        raise ValueError("No data fetched for ablation run.")

    results = []
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
        )
        results.append(
            extract_ablation_result_row(
                report,
                spec,
                prediction_days=args.prediction_days,
                period=args.period,
                universe=args.universe,
            )
        )

    print(format_ablation_table(results))
    output_path = (
        Path(args.output)
        if args.output
        else build_output_path(args.prediction_days, args.period, args.universe)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved ablation results to {output_path}")
    return results


if __name__ == "__main__":
    main()
