"""Report assembly, formatting, and logging for model training."""

import logging

import pandas as pd

from src.config import PREDICTION_DAYS
from src.train.evaluation import (
    build_actual_return_baseline_report,
    build_classification_report,
    build_combined_signal_report,
    build_probability_summary,
    build_probability_tail_report,
    build_probability_threshold_report,
    build_probability_ranked_top_n_selection_reports,
    build_predicted_return_quantile_report,
    build_regression_report,
    build_return_correlation_report,
    build_same_date_ranking_diagnostics_with_baselines,
    build_top_n_selection_reports,
    build_trading_relevance_report,
    build_validation_selected_threshold_report,
)
from src.train.evaluation.walk_forward import build_walk_forward_fold_report

def log_feature_importances(model_name, feature_importances, top_n=5):
    """Log a compact INFO summary and full DEBUG feature-importance details."""
    top_importances = feature_importances.head(top_n)
    top_importance_summary = {
        row["feature"]: round(float(row["importance"]), 4)
        for _, row in top_importances.iterrows()
    }
    logging.info(
        f"Top {top_n} Feature Importances for {model_name}: {top_importance_summary}"
    )
    logging.debug(f"Feature Importances for {model_name}:")
    for _, row in feature_importances.iterrows():
        logging.debug(f"{row['feature']}: {row['importance']:.4f}")


def format_top_n_ranked_selection_summary(
    top_n_report,
    title="XGBoost Top-N Ranked Selection Summary:",
):
    """Format nested Top-N ranked-selection diagnostics for readable INFO logs."""
    lines = [title]

    for bucket_name, bucket_report in top_n_report.items():
        model = bucket_report.get("model", {})
        random_baseline = bucket_report.get("random_baseline", {})
        momentum_baseline = bucket_report.get("momentum_baseline", {})
        relative_momentum_baseline = bucket_report.get("relative_momentum_baseline", {})
        universe = bucket_report.get("universe", {})
        momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
        relative_momentum_label = relative_momentum_baseline.get(
            "relative_momentum_score_column",
            "relative_momentum",
        )

        model_excess = model.get("average_selected_excess_return_vs_benchmark")
        random_excess = random_baseline.get(
            "average_selected_excess_return_vs_benchmark"
        )
        momentum_excess = momentum_baseline.get(
            "average_selected_excess_return_vs_benchmark"
        )
        relative_momentum_excess = relative_momentum_baseline.get(
            "average_selected_excess_return_vs_benchmark"
        )
        universe_excess = universe.get("average_excess_return_vs_benchmark")
        beat_rate = model.get("beat_benchmark_rate")

        lines.append(f"{bucket_name}:")
        if momentum_baseline.get("available", True):
            momentum_text = _format_percent(momentum_excess)
            minus_momentum_text = _format_percent_delta(
                model_excess,
                momentum_excess,
            )
        else:
            momentum_text = "unavailable"
            minus_momentum_text = "unavailable"
        if relative_momentum_baseline.get("available", True):
            relative_momentum_text = _format_percent(relative_momentum_excess)
            minus_relative_momentum_text = _format_percent_delta(
                model_excess,
                relative_momentum_excess,
            )
        else:
            relative_momentum_text = "unavailable"
            minus_relative_momentum_text = "unavailable"

        lines.append(
            "  "
            f"model excess={_format_percent(model_excess)}, "
            f"random excess={_format_percent(random_excess)}, "
            f"{momentum_label} excess={momentum_text}, "
            f"{relative_momentum_label} excess={relative_momentum_text}, "
            f"universe excess={_format_percent(universe_excess)}"
        )
        lines.append(
            "  "
            f"model minus random={_format_percent_delta(model_excess, random_excess)}, "
            f"model minus momentum={minus_momentum_text}, "
            f"model minus relative momentum={minus_relative_momentum_text}, "
            f"model beat rate={_format_percent(beat_rate)}"
        )

    return "\n".join(lines)


def format_top_n_basket_backtest_summary(
    top_n_report,
    title="XGBoost Top-N Basket Backtest Summary:",
):
    """Format nested Top-N basket backtest diagnostics for readable INFO logs."""
    lines = [title]

    for bucket_name, bucket_report in top_n_report.items():
        model = bucket_report.get("model", {})
        random_baseline = bucket_report.get("random_baseline", {})
        momentum_baseline = bucket_report.get("momentum_baseline", {})
        relative_momentum_baseline = bucket_report.get("relative_momentum_baseline", {})
        universe = bucket_report.get("universe", {})
        benchmark = bucket_report.get("benchmark", {})
        momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
        relative_momentum_label = relative_momentum_baseline.get(
            "relative_momentum_score_column",
            "relative_momentum",
        )

        model_raw = model.get("average_basket_raw_return")
        model_excess = model.get("average_basket_excess_return")
        random_excess = random_baseline.get("average_basket_excess_return")
        momentum_excess = momentum_baseline.get("average_basket_excess_return")
        relative_momentum_excess = relative_momentum_baseline.get(
            "average_basket_excess_return"
        )
        universe_excess = universe.get("average_basket_excess_return")
        benchmark_raw = benchmark.get("average_basket_raw_return")
        beat_rate = model.get("beat_benchmark_rate")

        lines.append(f"{bucket_name}:")
        if momentum_baseline.get("available", True):
            momentum_text = _format_percent(momentum_excess)
            minus_momentum_text = _format_percent_delta(
                model_excess,
                momentum_excess,
            )
        else:
            momentum_text = "unavailable"
            minus_momentum_text = "unavailable"
        if relative_momentum_baseline.get("available", True):
            relative_momentum_text = _format_percent(relative_momentum_excess)
            minus_relative_momentum_text = _format_percent_delta(
                model_excess,
                relative_momentum_excess,
            )
        else:
            relative_momentum_text = "unavailable"
            minus_relative_momentum_text = "unavailable"

        lines.append(
            "  "
            f"model raw={_format_percent(model_raw)}, "
            f"model excess={_format_percent(model_excess)}, "
            f"random excess={_format_percent(random_excess)}, "
            f"{momentum_label} excess={momentum_text}, "
            f"{relative_momentum_label} excess={relative_momentum_text}, "
            f"universe excess={_format_percent(universe_excess)}, "
            f"benchmark raw={_format_percent(benchmark_raw)}"
        )
        lines.append(
            "  "
            f"model minus random={_format_percent_delta(model_excess, random_excess)}, "
            f"model minus momentum={minus_momentum_text}, "
            f"model minus relative momentum={minus_relative_momentum_text}, "
            f"beat benchmark rate={_format_percent(beat_rate)}"
        )
        ci_text = _format_basket_bootstrap_ci_summary(
            bucket_report.get("bootstrap_confidence_intervals", {})
        )
        if ci_text:
            lines.append(f"  {ci_text}")

    return "\n".join(lines)


def format_top_n_basket_backtest_by_year_summary(
    top_n_by_year_report,
    title="XGBoost Top-N Basket Backtest By-Year Summary:",
):
    """Format by-year Top-N basket backtest diagnostics for readable INFO logs."""
    lines = [title]

    for bucket_name, bucket_report in top_n_by_year_report.items():
        for year, year_report in bucket_report.items():
            model = year_report.get("model", {})
            random_baseline = year_report.get("random_baseline", {})
            momentum_baseline = year_report.get("momentum_baseline", {})
            relative_momentum_baseline = year_report.get(
                "relative_momentum_baseline", {}
            )
            universe = year_report.get("universe", {})
            momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
            relative_momentum_label = relative_momentum_baseline.get(
                "relative_momentum_score_column",
                "relative_momentum",
            )

            model_excess = model.get("average_basket_excess_return")
            random_excess = random_baseline.get("average_basket_excess_return")
            momentum_excess = momentum_baseline.get("average_basket_excess_return")
            relative_momentum_excess = relative_momentum_baseline.get(
                "average_basket_excess_return"
            )
            universe_excess = universe.get("average_basket_excess_return")
            beat_rate = model.get("beat_benchmark_rate")
            evaluated_dates = model.get("evaluated_dates")

            if momentum_baseline.get("available", True):
                momentum_text = _format_percent(momentum_excess)
                minus_momentum_text = _format_percent_delta(
                    model_excess,
                    momentum_excess,
                )
            else:
                momentum_text = "unavailable"
                minus_momentum_text = "unavailable"
            if relative_momentum_baseline.get("available", True):
                relative_momentum_text = _format_percent(relative_momentum_excess)
                minus_relative_momentum_text = _format_percent_delta(
                    model_excess,
                    relative_momentum_excess,
                )
            else:
                relative_momentum_text = "unavailable"
                minus_relative_momentum_text = "unavailable"

            lines.append(
                f"{year} {bucket_name}: "
                f"model excess={_format_percent(model_excess)}, "
                f"random excess={_format_percent(random_excess)}, "
                f"{momentum_label} excess={momentum_text}, "
                f"{relative_momentum_label} excess={relative_momentum_text}, "
                f"universe excess={_format_percent(universe_excess)}, "
                f"model minus random={_format_percent_delta(model_excess, random_excess)}, "
                f"model minus momentum={minus_momentum_text}, "
                f"model minus relative momentum={minus_relative_momentum_text}, "
                f"beat benchmark rate={_format_percent(beat_rate)}, "
                f"evaluated dates={_format_count(evaluated_dates)}"
            )

    return "\n".join(lines)


def format_horizon_comparison_summary(horizon_reports):
    """Format basket backtest metrics across prediction horizons."""
    lines = ["Horizon Comparison Summary:"]

    for prediction_days in sorted(horizon_reports):
        lines.append(f"{prediction_days}d:")
        basket_report = horizon_reports[prediction_days].get("basket_backtest", {})

        for bucket_name, bucket_report in basket_report.items():
            model = bucket_report.get("model", {})
            random_baseline = bucket_report.get("random_baseline", {})
            momentum_baseline = bucket_report.get("momentum_baseline", {})
            relative_momentum_baseline = bucket_report.get(
                "relative_momentum_baseline", {}
            )
            universe = bucket_report.get("universe", {})
            benchmark = bucket_report.get("benchmark", {})
            momentum_label = momentum_baseline.get("momentum_score_column", "momentum")
            relative_momentum_label = relative_momentum_baseline.get(
                "relative_momentum_score_column",
                "relative_momentum",
            )

            if momentum_baseline.get("available", True):
                momentum_text = _format_percent(
                    momentum_baseline.get("average_basket_excess_return")
                )
            else:
                momentum_text = "unavailable"
            if relative_momentum_baseline.get("available", True):
                relative_momentum_text = _format_percent(
                    relative_momentum_baseline.get("average_basket_excess_return")
                )
            else:
                relative_momentum_text = "unavailable"

            lines.append(
                "  "
                f"{bucket_name} "
                f"model excess={_format_percent(model.get('average_basket_excess_return'))}, "
                f"random={_format_percent(random_baseline.get('average_basket_excess_return'))}, "
                f"{momentum_label}={momentum_text}, "
                f"{relative_momentum_label}={relative_momentum_text}, "
                f"universe={_format_percent(universe.get('average_basket_excess_return'))}, "
                f"benchmark raw={_format_percent(benchmark.get('average_basket_raw_return'))}"
            )

    return "\n".join(lines)


def format_same_date_ranking_diagnostics_summary(
    ranking_diagnostics,
    title="Same-Date Ranking Diagnostics:",
):
    """Format same-date ranking diagnostics for readable INFO logs."""
    if not ranking_diagnostics.get("available", False):
        reason = ranking_diagnostics.get("reason", "unavailable")
        return f"{title}\n  unavailable: {reason}"

    lines = [title]
    rank_ic = ranking_diagnostics.get("rank_ic", {})
    if rank_ic.get("available", False):
        lines.append(
            "  "
            f"rank IC dates={_format_count(rank_ic.get('date_count'))}, "
            f"mean={_format_decimal(rank_ic.get('mean_rank_ic'))}, "
            f"median={_format_decimal(rank_ic.get('median_rank_ic'))}, "
            f"positive rate={_format_percent(rank_ic.get('rank_ic_positive_rate'))}"
        )
    else:
        lines.append(f"  rank IC unavailable: {rank_ic.get('reason', 'unavailable')}")

    selected_distribution = ranking_diagnostics.get(
        "selected_realized_rank_distribution",
        {},
    )
    if selected_distribution.get("available", False):
        for bucket_name in sorted(
            [
                key
                for key in selected_distribution
                if key.startswith("top_") and isinstance(selected_distribution[key], dict)
            ],
            key=lambda key: int(key.split("_", maxsplit=1)[1]),
        ):
            bucket_report = selected_distribution[bucket_name]
            lines.append(
                "  "
                f"{bucket_name}: selected={_format_count(bucket_report.get('selected_count'))}, "
                f"realized top20={_format_percent(bucket_report.get('top_20_realized_rate'))}, "
                f"middle60={_format_percent(bucket_report.get('middle_60_realized_rate'))}, "
                f"bottom20={_format_percent(bucket_report.get('bottom_20_realized_rate'))}, "
                f"avg rank pct={_format_decimal(bucket_report.get('average_realized_rank_pct'))}"
            )
    else:
        lines.append(
            "  selected realized-rank unavailable: "
            f"{selected_distribution.get('reason', 'unavailable')}"
        )

    return "\n".join(lines)


def _format_percent(value):
    """Format optional numeric report values as percentages for logs."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_decimal(value):
    """Format optional numeric diagnostics values as compact decimals."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def _format_percent_delta(left, right):
    """Format the percentage-point difference between two report values."""
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return "n/a"
    return f"{float(left) - float(right):.2%}"


def _format_count(value):
    """Format optional count-like report values for logs."""
    if value is None or pd.isna(value):
        return "n/a"
    return str(int(value))


def _format_basket_bootstrap_ci_summary(ci_report):
    """Format compact bootstrap CI diagnostics for one basket bucket."""
    if not ci_report:
        return ""

    ci_specs = [
        (
            "model excess",
            ci_report.get("model_average_basket_excess_return"),
        ),
        (
            "model minus momentum",
            ci_report.get("model_minus_momentum_baseline_average_basket_excess_return"),
        ),
        (
            "model minus relative momentum",
            ci_report.get(
                "model_minus_relative_momentum_baseline_average_basket_excess_return"
            ),
        ),
        (
            "model minus universe",
            ci_report.get("model_minus_universe_average_basket_excess_return"),
        ),
    ]
    formatted_intervals = [
        _format_bootstrap_interval(label, interval)
        for label, interval in ci_specs
        if interval and interval.get("available") is True
    ]
    return ", ".join(formatted_intervals)


def _format_bootstrap_interval(label, interval):
    """Format one percentile bootstrap interval for logs."""
    confidence_level = interval.get("confidence_level")
    confidence_text = (
        "CI"
        if confidence_level is None or pd.isna(confidence_level)
        else f"{float(confidence_level):.0%} CI"
    )
    return (
        f"{label} {confidence_text}="
        f"[{_format_percent(interval.get('ci_lower'))}, "
        f"{_format_percent(interval.get('ci_upper'))}]"
    )



def format_xgboost_regressor_validation_selection_report(selection_report):
    """Format validation-based XGBoost regressor selection for INFO logs."""
    lines = ["XGBoost Regressor Validation Top-N Selection Report:"]
    candidates = selection_report.get("candidates", [])

    for candidate_report in candidates:
        selected_suffix = " selected" if candidate_report.get("selected") else ""
        bucket_excess_returns = candidate_report.get(
            "validation_top_n_basket_excess_returns",
            {},
        )
        bucket_names = sorted(
            bucket_excess_returns,
            key=lambda bucket_name: int(bucket_name.split("_", maxsplit=1)[1]),
        )
        bucket_text = ", ".join(
            f"{bucket_name}={_format_percent(bucket_excess_returns.get(bucket_name))}"
            for bucket_name in bucket_names
        )
        lines.append(
            f"{candidate_report.get('candidate_name')}{selected_suffix}: "
            f"score={_format_percent(candidate_report.get('validation_top_n_mean_excess_return'))}, "
            f"{bucket_text}, "
            f"params={candidate_report.get('params')}"
        )

    lines.append(
        "selected="
        f"{selection_report.get('selected_candidate_name')} "
        f"score={_format_percent(selection_report.get('selected_validation_top_n_mean_excess_return'))}"
    )
    return "\n".join(lines)



def evaluate_model(model, x_test, y_test, model_type="regression"):
    """Log a compact regression or classification report for a held-out split."""
    y_pred = model.predict(x_test)

    if model_type == "regression":
        report = build_regression_report(y_test, y_pred)
        logging.info(
            f"MSE: {report['mse']:.4f}, MAE: {report['mae']:.4f}, R²: {report['r2']:.4f}"
        )

    elif model_type == "classification":
        report = build_classification_report(y_test, y_pred)
        logging.info(f"Accuracy: {report['accuracy']:.4%}")
        logging.info(f"Classification Report: {report}")


def log_xgboost_test_report(
    y_train,
    y_val,
    y_test,
    direction_y_test,
    test_split_metadata,
    classifier_predictions,
    classifier_validation_probability_up,
    classifier_probability_up,
    regressor_predictions,
    prediction_days=PREDICTION_DAYS,
    random_trials=100,
    random_trial_workers=4,
    regressor_validation_selection_report=None,
):
    """
    Log final SPY-relative XGBoost diagnostics without tuning on test data.

    Validation probabilities may select a beat-benchmark threshold; test
    probabilities only evaluate the already-selected rule. y values are excess
    returns relative to SPY.
    """
    classification_report_data = build_classification_report(
        direction_y_test, classifier_predictions
    )
    regression_report_data = build_regression_report(
        y_test, regressor_predictions, y_train=y_train
    )
    actual_return_baseline_report = build_actual_return_baseline_report(y_test)
    trading_report_data = build_trading_relevance_report(
        y_test, regressor_predictions, classifier_predictions
    )
    probability_summary = build_probability_summary(classifier_probability_up)
    probability_tail_report = build_probability_tail_report(
        classifier_probability_up, y_test
    )
    probability_threshold_report = build_probability_threshold_report(
        classifier_probability_up, y_test
    )
    validation_selected_threshold_report = build_validation_selected_threshold_report(
        classifier_validation_probability_up,
        y_val,
        classifier_probability_up,
        y_test,
    )
    predicted_return_quantile_report = build_predicted_return_quantile_report(
        y_test, regressor_predictions
    )
    return_correlation_report = build_return_correlation_report(
        y_test, regressor_predictions
    )
    combined_signal_report = build_combined_signal_report(
        classifier_probability_up,
        y_test,
        regressor_predictions,
    )
    top_n_selection_reports = build_top_n_selection_reports(
        test_split_metadata,
        regressor_predictions,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    ranking_diagnostics = build_same_date_ranking_diagnostics_with_baselines(
        test_split_metadata,
        regressor_predictions,
        model_score_column="predicted_excess_return",
        prediction_days=prediction_days,
    )
    top_n_ranked_selection_report = top_n_selection_reports["ranked_selection"]
    top_n_basket_backtest_report = top_n_selection_reports["basket_backtest"]
    top_n_basket_backtest_by_year_report = top_n_selection_reports.get(
        "basket_backtest_by_year",
        {},
    )
    classifier_top_n_selection_reports = (
        build_probability_ranked_top_n_selection_reports(
            test_split_metadata,
            classifier_probability_up,
            prediction_days=prediction_days,
            random_trials=random_trials,
            random_trial_workers=random_trial_workers,
        )
    )
    classifier_top_n_ranked_selection_report = classifier_top_n_selection_reports[
        "ranked_selection"
    ]
    classifier_top_n_basket_backtest_report = classifier_top_n_selection_reports[
        "basket_backtest"
    ]
    classifier_top_n_basket_backtest_by_year_report = (
        classifier_top_n_selection_reports.get("basket_backtest_by_year", {})
    )

    logging.info(
        f"XGBoost Beat-Benchmark Classification Report: {classification_report_data}"
    )
    logging.info(f"XGBoost Excess Return Regression Report: {regression_report_data}")
    logging.info(
        f"XGBoost Excess Return Baseline Report: {actual_return_baseline_report}"
    )
    logging.info(f"XGBoost Beat-Benchmark Relevance Report: {trading_report_data}")
    logging.info(f"XGBoost Beat-Benchmark Probability Summary: {probability_summary}")
    logging.info(
        f"XGBoost Beat-Benchmark Probability Tail Report: {probability_tail_report}"
    )
    logging.info(
        f"XGBoost Beat-Benchmark Probability Threshold Report: {probability_threshold_report}"
    )
    logging.info(
        f"XGBoost Beat-Benchmark Validation-Selected Threshold Report: {validation_selected_threshold_report}"
    )
    logging.info(
        f"XGBoost Regressor Predicted Excess Return Quantile Report: {predicted_return_quantile_report}"
    )
    logging.info(
        f"XGBoost Regressor Excess Return Correlation Report: {return_correlation_report}"
    )
    logging.info(f"XGBoost Combined Signal Report: {combined_signal_report}")
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics,
            title="XGBoost Regressor Model Score Same-Date Ranking Diagnostics:",
        )
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics["momentum"],
            title="XGBoost Regressor Momentum Same-Date Ranking Diagnostics:",
        )
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics["relative_momentum"],
            title=(
                "XGBoost Regressor Relative Momentum Same-Date Ranking Diagnostics:"
            ),
        )
    )
    if regressor_validation_selection_report:
        logging.info(
            format_xgboost_regressor_validation_selection_report(
                regressor_validation_selection_report
            )
        )
    logging.info(format_top_n_ranked_selection_summary(top_n_ranked_selection_report))
    logging.info(format_top_n_basket_backtest_summary(top_n_basket_backtest_report))
    if top_n_basket_backtest_by_year_report:
        logging.info(
            format_top_n_basket_backtest_by_year_summary(
                top_n_basket_backtest_by_year_report
            )
        )
    logging.info(
        format_top_n_ranked_selection_summary(
            classifier_top_n_ranked_selection_report,
            title="XGBoost Classifier-Probability Top-N Ranked Selection Summary:",
        )
    )
    logging.info(
        format_top_n_basket_backtest_summary(
            classifier_top_n_basket_backtest_report,
            title="XGBoost Classifier-Probability Top-N Basket Backtest Summary:",
        )
    )
    logging.debug(
        f"XGBoost Top-N Ranked Selection Report: {top_n_ranked_selection_report}"
    )
    logging.debug(
        f"XGBoost Top-N Basket Backtest Report: {top_n_basket_backtest_report}"
    )
    logging.debug(
        "XGBoost Top-N Basket Backtest By-Year Report: "
        f"{top_n_basket_backtest_by_year_report}"
    )
    logging.debug(
        "XGBoost Classifier-Probability Top-N Ranked Selection Report: "
        f"{classifier_top_n_ranked_selection_report}"
    )
    logging.debug(
        "XGBoost Classifier-Probability Top-N Basket Backtest Report: "
        f"{classifier_top_n_basket_backtest_report}"
    )
    logging.debug(
        "XGBoost Classifier-Probability Top-N Basket Backtest By-Year Report: "
        f"{classifier_top_n_basket_backtest_by_year_report}"
    )
    return {
        "regressor_validation_selection": regressor_validation_selection_report or {},
        "ranked_selection": top_n_ranked_selection_report,
        "basket_backtest": top_n_basket_backtest_report,
        "basket_backtest_by_year": top_n_basket_backtest_by_year_report,
        "same_date_ranking_diagnostics": ranking_diagnostics,
        "classifier_probability_ranked_selection": (
            classifier_top_n_ranked_selection_report
        ),
        "classifier_probability_basket_backtest": classifier_top_n_basket_backtest_report,
        "classifier_probability_basket_backtest_by_year": (
            classifier_top_n_basket_backtest_by_year_report
        ),
    }



def log_rank_ndcg_test_report(
    test_split_metadata,
    ranking_scores,
    prediction_days=PREDICTION_DAYS,
    random_trials=100,
    random_trial_workers=4,
    phase_timing_collector: dict | None = None,
):
    """Log Top-N reports that rank candidates by grouped Rank-NDCG score."""
    top_n_kwargs = {
        "prediction_days": prediction_days,
        "random_trials": random_trials,
        "random_trial_workers": random_trial_workers,
    }
    if phase_timing_collector is not None:
        top_n_kwargs["phase_timing_collector"] = phase_timing_collector
    top_n_selection_reports = build_top_n_selection_reports(
        test_split_metadata,
        ranking_scores,
        **top_n_kwargs,
    )
    ranking_diagnostics = build_same_date_ranking_diagnostics_with_baselines(
        test_split_metadata,
        ranking_scores,
        model_score_column="xgboost_rank_ndcg_score",
        prediction_days=prediction_days,
    )
    top_n_ranked_selection_report = top_n_selection_reports["ranked_selection"]
    top_n_basket_backtest_report = top_n_selection_reports["basket_backtest"]
    top_n_basket_backtest_by_year_report = top_n_selection_reports.get(
        "basket_backtest_by_year",
        {},
    )

    logging.info("XGBoost Rank-NDCG Score: xgboost_rank_ndcg_score")
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics,
            title="XGBoost Rank-NDCG Same-Date Ranking Diagnostics:",
        )
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics["momentum"],
            title="XGBoost Rank-NDCG Momentum Same-Date Ranking Diagnostics:",
        )
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics["relative_momentum"],
            title=(
                "XGBoost Rank-NDCG Relative Momentum Same-Date Ranking Diagnostics:"
            ),
        )
    )
    logging.info(
        format_top_n_ranked_selection_summary(
            top_n_ranked_selection_report,
            title="XGBoost Rank-NDCG Top-N Ranked Selection Summary:",
        )
    )
    logging.info(
        format_top_n_basket_backtest_summary(
            top_n_basket_backtest_report,
            title="XGBoost Rank-NDCG Top-N Basket Backtest Summary:",
        )
    )
    if top_n_basket_backtest_by_year_report:
        logging.info(
            format_top_n_basket_backtest_by_year_summary(
                top_n_basket_backtest_by_year_report,
                title="XGBoost Rank-NDCG Top-N Basket Backtest By-Year Summary:",
            )
        )

    return {
        "ranked_selection": top_n_ranked_selection_report,
        "basket_backtest": top_n_basket_backtest_report,
        "basket_backtest_by_year": top_n_basket_backtest_by_year_report,
        "same_date_ranking_diagnostics": ranking_diagnostics,
    }


def log_ranking_classifier_test_report(
    test_split_metadata,
    ranking_scores,
    prediction_days=PREDICTION_DAYS,
    random_trials=100,
    random_trial_workers=4,
):
    """Log Top-N reports that rank candidates by ranking-classifier score."""
    top_n_selection_reports = build_top_n_selection_reports(
        test_split_metadata,
        ranking_scores,
        prediction_days=prediction_days,
        random_trials=random_trials,
        random_trial_workers=random_trial_workers,
    )
    ranking_diagnostics = build_same_date_ranking_diagnostics_with_baselines(
        test_split_metadata,
        ranking_scores,
        model_score_column="probability_of_top_quintile_outperformance",
        prediction_days=prediction_days,
    )
    top_n_ranked_selection_report = top_n_selection_reports["ranked_selection"]
    top_n_basket_backtest_report = top_n_selection_reports["basket_backtest"]
    top_n_basket_backtest_by_year_report = top_n_selection_reports.get(
        "basket_backtest_by_year",
        {},
    )

    logging.info(
        "XGBoost Cross-Sectional Ranking Score: "
        "probability_of_top_quintile_outperformance"
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics,
            title=(
                "XGBoost Ranking-Classifier Model Score Same-Date Ranking "
                "Diagnostics:"
            ),
        )
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics["momentum"],
            title="XGBoost Ranking-Classifier Momentum Same-Date Ranking Diagnostics:",
        )
    )
    logging.info(
        format_same_date_ranking_diagnostics_summary(
            ranking_diagnostics["relative_momentum"],
            title=(
                "XGBoost Ranking-Classifier Relative Momentum Same-Date Ranking "
                "Diagnostics:"
            ),
        )
    )
    logging.info(
        format_top_n_ranked_selection_summary(
            top_n_ranked_selection_report,
            title="XGBoost Ranking-Classifier Top-N Ranked Selection Summary:",
        )
    )
    logging.info(
        format_top_n_basket_backtest_summary(
            top_n_basket_backtest_report,
            title="XGBoost Ranking-Classifier Top-N Basket Backtest Summary:",
        )
    )
    if top_n_basket_backtest_by_year_report:
        logging.info(
            format_top_n_basket_backtest_by_year_summary(
                top_n_basket_backtest_by_year_report,
                title="XGBoost Ranking-Classifier Top-N Basket Backtest By-Year Summary:",
            )
        )

    return {
        "ranked_selection": top_n_ranked_selection_report,
        "basket_backtest": top_n_basket_backtest_report,
        "basket_backtest_by_year": top_n_basket_backtest_by_year_report,
        "same_date_ranking_diagnostics": ranking_diagnostics,
    }



def build_walk_forward_report_for_split(
    fold,
    split,
    selection_report,
    basket_backtest_report,
):
    """Apply split date ranges to a fold before building its compact report."""
    report_fold = dict(fold)
    split_date_ranges = split.get("split_date_ranges", {})
    report_fold["train_date_range"] = split_date_ranges.get(
        "train",
        fold["train_date_range"],
    )
    report_fold["validation_date_range"] = split_date_ranges.get(
        "validation",
        fold["validation_date_range"],
    )
    report_fold["test_date_range"] = split_date_ranges.get(
        "test",
        fold["test_date_range"],
    )
    return build_walk_forward_fold_report(
        report_fold,
        selection_report,
        basket_backtest_report,
    )


def format_walk_forward_fold_summary(fold_report):
    """Format one walk-forward fold for concise logs."""
    lines = [
        "Walk-Forward Fold "
        f"{fold_report['fold_index']}: "
        f"train={_format_date_range(fold_report['train_date_range'])}, "
        f"validation={_format_date_range(fold_report['validation_date_range'])}, "
        f"test={_format_date_range(fold_report['test_date_range'])}, "
        f"selected={fold_report.get('selected_candidate_name')} "
        f"score={_format_percent(fold_report.get('validation_selection_score'))}"
    ]
    for bucket_name, bucket_metrics in fold_report.get("top_n", {}).items():
        lines.append(
            "  "
            f"{bucket_name}: "
            f"model excess={_format_percent(bucket_metrics.get('model_excess'))}, "
            "model minus momentum="
            f"{_format_percent(bucket_metrics.get('model_minus_momentum'))}, "
            "model minus universe="
            f"{_format_percent(bucket_metrics.get('model_minus_universe'))}"
        )
    return "\n".join(lines)


def format_walk_forward_summary(walk_forward_report):
    """Format aggregate walk-forward diagnostics for console and logs."""
    aggregate = walk_forward_report.get("aggregate", {})
    lines = [
        "Walk-Forward Top-N Summary:",
        f"folds={aggregate.get('fold_count', 0)}",
    ]
    selected_counts = aggregate.get("selected_candidate_counts", {})
    if selected_counts:
        lines.append(f"selected candidate counts={selected_counts}")

    for bucket_name, bucket_summary in aggregate.get("top_n", {}).items():
        lines.append(
            f"{bucket_name}: "
            "avg model excess="
            f"{_format_percent(bucket_summary.get('average_model_excess'))}, "
            "avg model minus momentum="
            f"{_format_percent(bucket_summary.get('average_model_minus_momentum'))}, "
            "avg model minus universe="
            f"{_format_percent(bucket_summary.get('average_model_minus_universe'))}, "
            "win rate vs momentum="
            f"{_format_percent(bucket_summary.get('fold_win_rate_vs_momentum'))}, "
            "win rate vs universe="
            f"{_format_percent(bucket_summary.get('fold_win_rate_vs_universe'))}"
        )

    for fold_report in walk_forward_report.get("folds", []):
        lines.append(format_walk_forward_fold_summary(fold_report))

    return "\n".join(lines)


def _format_date_range(date_range):
    """Format date range metadata for logs."""
    if not date_range or date_range.get("start") is None:
        return "n/a"
    return f"{date_range['start']}..{date_range['end']}"
