import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from src.config import PREDICTION_DAYS, REQUIRED_COLUMNS


class DataPreparator:
    """
    Prepare features, SPY-relative targets, global date splits, and scaled arrays.

    The benchmark ticker supplies date-aligned forward returns for target creation
    and is removed from candidate training rows after labels are built.
    """

    def __init__(self, benchmark_ticker: str = "SPY"):
        self.scalar = StandardScaler()
        self.feature_columns: List[str] = []
        self.benchmark_ticker = benchmark_ticker

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove provider columns that are not model features."""
        df = df.copy()
        columns_to_drop = ["Dividends", "Stock Splits"]
        return df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )

    def create_target(
        self,
        df: pd.DataFrame,
        prediction_days: int = PREDICTION_DAYS,
    ) -> pd.DataFrame:
        """Create ticker-aware raw forward returns without benchmark alignment."""
        df = self._normalize_prediction_date(df)
        df = self._create_raw_forward_returns(df, prediction_days)
        df = df.dropna(subset=["raw_forward_return"]).copy()
        df["targetReturns"] = df["raw_forward_return"]
        return df

    def _normalize_prediction_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "prediction_date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["prediction_date"] = df.index
            else:
                raise ValueError("Input data must include prediction_date.")

        if "Ticker" not in df.columns:
            raise ValueError("Input data must include Ticker.")

        df["_source_index"] = df.index
        df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.normalize()
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

    def _create_raw_forward_returns(
        self,
        df: pd.DataFrame,
        prediction_days: int,
    ) -> pd.DataFrame:
        df = df.sort_values(["Ticker", "prediction_date"]).copy()
        future_close = df.groupby("Ticker", sort=False)["Close"].shift(-prediction_days)
        df["raw_forward_return"] = (future_close - df["Close"]) / df["Close"]
        return df

    def _build_benchmark_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        benchmark_rows = df[df["Ticker"] == self.benchmark_ticker]
        if benchmark_rows.empty:
            raise ValueError(f"Benchmark ticker {self.benchmark_ticker} is missing.")

        benchmark_returns = benchmark_rows[
            ["prediction_date", "raw_forward_return"]
        ].dropna(subset=["raw_forward_return"])
        benchmark_returns = benchmark_returns.drop_duplicates(
            subset=["prediction_date"],
            keep="first",
        )
        return benchmark_returns.rename(
            columns={"raw_forward_return": "benchmark_forward_return"}
        )

    def _create_benchmark_relative_targets(
        self,
        df: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        candidates = df[df["Ticker"] != self.benchmark_ticker].copy()
        candidates = candidates.merge(
            benchmark_returns,
            on="prediction_date",
            how="left",
        )
        candidates["excess_forward_return"] = (
            candidates["raw_forward_return"] - candidates["benchmark_forward_return"]
        )
        candidates["targetReturns"] = candidates["excess_forward_return"]
        candidates["beat_benchmark_target"] = (
            candidates["raw_forward_return"] > candidates["benchmark_forward_return"]
        ).astype(int)
        return candidates

    def _drop_unusable_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        required_target_columns = [
            "raw_forward_return",
            "benchmark_forward_return",
            "excess_forward_return",
            "targetReturns",
            "beat_benchmark_target",
        ]
        return df.dropna(
            subset=self.feature_columns + required_target_columns
        ).copy()

    def _split_by_prediction_date(
        self,
        df: pd.DataFrame,
        prediction_days: int,
        val_size: float,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        unique_dates = np.array(sorted(df["prediction_date"].drop_duplicates()))
        num_dates = len(unique_dates)
        test_count = int(np.ceil(num_dates * test_size))
        val_count = int(np.ceil(num_dates * val_size))
        gap_count = prediction_days

        train_end = num_dates - test_count - gap_count - val_count - gap_count
        if train_end <= 0 or val_count <= 0 or test_count <= 0:
            raise ValueError(
                "Not enough dates to create train, validation, and test splits with the requested gap."
            )

        val_start = train_end + gap_count
        val_end = val_start + val_count
        test_start = val_end + gap_count

        train_dates = set(unique_dates[:train_end])
        val_dates = set(unique_dates[val_start:val_end])
        test_dates = set(unique_dates[test_start:])

        train_df = df[df["prediction_date"].isin(train_dates)].copy()
        val_df = df[df["prediction_date"].isin(val_dates)].copy()
        test_df = df[df["prediction_date"].isin(test_dates)].copy()
        return train_df, val_df, test_df

    def _build_split_metadata(self, split_df: pd.DataFrame) -> pd.DataFrame:
        metadata_columns = [
            "_source_index",
            "Ticker",
            "prediction_date",
            "dailyReturn",
            "raw_forward_return",
            "benchmark_forward_return",
            "excess_forward_return",
            "beat_benchmark_target",
        ]
        optional_momentum_columns = [
            "momentum_5d",
            "momentum_10d",
            "momentum_20d",
            "momentum_50d",
        ]
        available_columns = metadata_columns + [
            column
            for column in optional_momentum_columns
            if column in split_df.columns
        ]
        return split_df[available_columns].copy()

    def prepare_for_train(
        self,
        df: pd.DataFrame,
        prediction_days: int = PREDICTION_DAYS,
        val_size: float = 0.15,
        test_size: float = 0.15,
        dropna: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Build SPY-relative targets, global date splits, and scaled model arrays.

        The dropna parameter remains for compatibility; missing required features
        and labels are always dropped.
        """
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be greater than 0 and less than 1.")
        if val_size < 0 or val_size >= 1:
            raise ValueError(
                "val_size must be greater than or equal to 0 and less than 1."
            )
        if val_size + test_size >= 1:
            raise ValueError("val_size + test_size must be less than 1.")

        df = self.prepare_features(df)
        df = self._normalize_prediction_date(df)
        self._validate_required_columns(df)
        self.feature_columns = list(REQUIRED_COLUMNS)

        df = self._create_raw_forward_returns(df, prediction_days)
        benchmark_returns = self._build_benchmark_forward_returns(df)
        df = self._create_benchmark_relative_targets(df, benchmark_returns)
        df = self._drop_unusable_rows(df)

        train_df, val_df, test_df = self._split_by_prediction_date(
            df,
            prediction_days,
            val_size,
            test_size,
        )

        x_train = train_df[self.feature_columns].values
        x_val = val_df[self.feature_columns].values
        x_test = test_df[self.feature_columns].values
        y_train = train_df["targetReturns"].values
        y_val = val_df["targetReturns"].values
        y_test = test_df["targetReturns"].values

        x_train = self.scalar.fit_transform(x_train)
        x_val = self.scalar.transform(x_val)
        x_test = self.scalar.transform(x_test)

        direction_y_train = train_df["beat_benchmark_target"].values.astype(int)
        direction_y_val = val_df["beat_benchmark_target"].values.astype(int)
        direction_y_test = test_df["beat_benchmark_target"].values.astype(int)

        return {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "direction_y_train": direction_y_train,
            "direction_y_val": direction_y_val,
            "direction_y_test": direction_y_test,
            "target_y_train": y_train,
            "target_y_val": y_val,
            "target_y_test": y_test,
            "feature_names": self.feature_columns,
            "scalar": self.scalar,
            "split_metadata": {
                "train": self._build_split_metadata(train_df),
                "val": self._build_split_metadata(val_df),
                "test": self._build_split_metadata(test_df),
            },
        }
