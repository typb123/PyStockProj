"""Run group or leave-one-feature-out Rank-NDCG ablations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

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
    DEFAULT_CPU_BUDGET,
    FEATURE_ABLATION_MODES,
    build_benchmark_parallelism_configs,
    build_paired_ablation_report,
    build_paired_fold_output_path,
    build_output_path,
    build_rank_ndcg_feature_ablation_specs,
    extract_ablation_result_row,
    flatten_paired_ablation_fold_rows,
    format_ablation_table,
    run_rank_ndcg_ablation_specs,
    validate_ablation_parallelism,
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
        "--outer-workers",
        type=_positive_int,
        default=1,
        help="Independent ablation processes. Defaults to 1 (serial variants).",
    )
    parser.add_argument(
        "--xgb-threads",
        type=_positive_int,
        default=None,
        help=(
            "XGBoost CPU threads per ablation process. Defaults to an even "
            "budget split."
        ),
    )
    parser.add_argument(
        "--cpu-budget",
        type=_positive_int,
        default=os.cpu_count() or DEFAULT_CPU_BUDGET,
        help="Maximum logical CPUs allocated across ablation processes.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run the small Rank-NDCG CPU configuration benchmark instead of "
            "an experiment."
        ),
    )
    parser.add_argument(
        "--benchmark-folds",
        type=_positive_int,
        default=1,
        help="Most-recent expanding folds per benchmark variant. Defaults to 1.",
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


def _positive_int(value):
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def _resolve_parallelism_args(args):
    xgb_threads = args.xgb_threads or max(1, args.cpu_budget // args.outer_workers)
    return validate_ablation_parallelism(
        args.outer_workers,
        xgb_threads,
        args.cpu_budget,
    )


def _benchmark_specs():
    """Build the fixed focused workload used by every CPU configuration."""
    focused_specs = build_rank_ndcg_feature_ablation_specs(
        mode="leave-one-out",
        drop_features=["rolling_signed_volume_20d"],
    )
    # Six fixed slots keep the 6x4 configuration occupied while ensuring all
    # benchmark rows compare the exact same ordered scientific workload.
    return focused_specs * 3


def _run_benchmark(data, args):
    """Benchmark bounded process/thread configurations on recent focused folds."""
    benchmark_results = []
    benchmark_specs = _benchmark_specs()
    for parallelism in build_benchmark_parallelism_configs(args.cpu_budget):
        print(
            "Benchmarking "
            f"outer_workers={parallelism.outer_workers}, "
            f"xgb_threads={parallelism.xgb_threads}, "
            f"cpu_budget={parallelism.cpu_budget}, "
            f"variant_runs={len(benchmark_specs)}, "
            f"recent_folds={args.benchmark_folds}."
        )
        started_at = time.perf_counter()
        runs = run_rank_ndcg_ablation_specs(
            data,
            benchmark_specs,
            outer_workers=parallelism.outer_workers,
            xgb_threads=parallelism.xgb_threads,
            cpu_budget=parallelism.cpu_budget,
            walk_forward_kwargs={
                "prediction_days": args.prediction_days,
                "random_trials": args.random_trials,
                # Keep benchmark CPU measurements free from random-baseline pools.
                "random_trial_workers": 1,
                "target_mode": TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
                "model_seed": args.model_seed,
                "max_folds": args.benchmark_folds,
            },
            serial_runner=run_walk_forward_models,
        )
        wall_clock_seconds = time.perf_counter() - started_at
        model_fit_count = sum(len(run.report.get("folds", [])) for run in runs)
        benchmark_results.append(
            {
                "outer_workers": parallelism.outer_workers,
                "xgb_threads": parallelism.xgb_threads,
                "cpu_budget": parallelism.cpu_budget,
                "benchmark_variant_runs": len(benchmark_specs),
                "recent_fold_count": args.benchmark_folds,
                "model_fit_count": model_fit_count,
                "wall_clock_seconds": wall_clock_seconds,
                "fits_per_minute": (
                    model_fit_count * 60.0 / wall_clock_seconds
                    if wall_clock_seconds > 0.0
                    else float("nan")
                ),
            }
        )

    benchmark_table = pd.DataFrame(benchmark_results)
    print(benchmark_table.to_string(index=False))
    return benchmark_results


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    parallelism = _resolve_parallelism_args(args)
    use_cache = not args.no_cache
    tickers = get_training_tickers(args.universe)

    print(
        "Fetching data for "
        f"universe={args.universe}, tickers={len(tickers)}, period={args.period}."
    )
    data = prepare_data_parallel(tickers, period=args.period, use_cache=use_cache)
    if data.empty:
        raise ValueError("No data fetched for ablation run.")
    if args.benchmark:
        return _run_benchmark(data, args)

    specs = build_rank_ndcg_feature_ablation_specs(
        mode=args.mode,
        drop_features=args.drop_feature,
    )

    results = []
    paired_reports = []
    baseline_report = None
    for spec in specs:
        print(
            f"Queueing {spec.name} "
            f"({len(spec.feature_columns)} features)..."
        )
    if parallelism.outer_workers > 1 and args.random_trial_workers > 1:
        print(
            "Outer ablation workers enabled; random baseline trials run "
            "serially per worker."
        )

    runs = run_rank_ndcg_ablation_specs(
        data,
        specs,
        outer_workers=parallelism.outer_workers,
        xgb_threads=parallelism.xgb_threads,
        cpu_budget=parallelism.cpu_budget,
        walk_forward_kwargs={
            "prediction_days": args.prediction_days,
            "random_trials": args.random_trials,
            "random_trial_workers": args.random_trial_workers,
            "target_mode": TARGET_MODE_CROSS_SECTIONAL_RANK_NDCG,
            "model_seed": args.model_seed,
        },
        serial_runner=run_walk_forward_models,
    )

    for run in runs:
        spec = run.spec
        report = run.report
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
