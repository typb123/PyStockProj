import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from src.config import REQUIRED_COLUMNS


class DataPreparator:
    """
    Prepare stock features, targets, chronological splits, and scaled arrays.

    Rows with missing required features or targetReturns are dropped before splitting.
    The dropna parameter is retained for API compatibility but is currently unused.
    """

    def __init__(self):
        self.scalar = StandardScaler()
        self.feature_columns: List[str] = []  # Feature order used by training/inference.

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature data by removing non-model market event columns.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df = df.copy()

        columns_to_drop = ["Dividends", "Stock Splits"]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )

        return df

    def create_target(self, df: pd.DataFrame, prediction_days: int = 5) -> pd.DataFrame:
        """
        Create future-return targets without crossing ticker boundaries.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing historical stock data.
            prediction_days (int): The number of days ahead to calculate the target.

        Returns:
            pd.DataFrame: DataFrame with 'targetReturns' and no intermediate target column.
        """
        df = df.copy()

        # Grouped shifts prevent one ticker from using the next ticker's future close.
        if "Ticker" in df.columns:
            df["target"] = df.groupby("Ticker")["Close"].shift(-prediction_days)
        else:
            df["target"] = df["Close"].shift(-prediction_days)
        df["targetReturns"] = (df["target"] - df["Close"]) / df["Close"]
        df.dropna(subset=["targetReturns"], inplace=True)
        df = df.drop(columns=["target"])
        return df

    def _split_group(
        self,
        group: pd.DataFrame,
        prediction_days: int,
        val_size: float,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split one ticker or single-series group into chronological train/val/test blocks.

        A prediction_days-row embargo separates adjacent splits so future-return labels
        near a boundary cannot overlap the next evaluation window.
        """
        group = group.sort_index()
        num_rows = len(group)
        test_count = int(np.ceil(num_rows * test_size))
        val_count = int(np.ceil(num_rows * val_size))
        gap_count = prediction_days

        train_end = num_rows - test_count - gap_count - val_count - gap_count
        if train_end <= 0 or val_count <= 0 or test_count <= 0:
            raise ValueError(
                "Not enough rows to create train, validation, and test splits with the requested gap."
            )

        val_start = train_end + gap_count
        val_end = val_start + val_count
        test_start = val_end + gap_count

        # Oldest rows train; middle rows validate; newest rows remain test.
        train = group.iloc[:train_end]
        val = group.iloc[val_start:val_end]
        test = group.iloc[test_start:]
        return train, val, test

    def prepare_for_train(
        self,
        df: pd.DataFrame,
        prediction_days: int = 5,
        val_size: float = 0.15,
        test_size: float = 0.15,
        dropna: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare arrays for model training with leakage-aware time-series splits.

        Target creation is ticker-aware when Ticker exists. Splits are chronological
        within each ticker, then the scaler is fit on train only and reused for
        validation and test.

        Parameters:
            df (pd.DataFrame): Input DataFrame with raw stock data.
            prediction_days (int): Number of days ahead to predict.
            val_size (float): Proportion of data reserved for validation.
            test_size (float): Proportion of data reserved for testing.
            dropna (bool): Retained for compatibility; rows with missing required
                features or targetReturns are always dropped.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing train, validation, and test sets,
                feature names, split metadata, and the fitted scaler.
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

        df = self.create_target(df, prediction_days)

        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        self.feature_columns = list(REQUIRED_COLUMNS)

        # Drop rows the models cannot use instead of imputing future-dependent values.
        df = df.dropna(subset=self.feature_columns + ["targetReturns"])
        df = df.copy()
        df["_source_index"] = df.index

        # Split each ticker independently so coverage remains chronological per ticker.
        if "Ticker" in df.columns:
            grouped_data = [group for _, group in df.groupby("Ticker", sort=False)]
        else:
            grouped_data = [df]

        split_parts = [
            self._split_group(group, prediction_days, val_size, test_size)
            for group in grouped_data
        ]

        train_df = pd.concat([parts[0] for parts in split_parts], ignore_index=True)
        val_df = pd.concat([parts[1] for parts in split_parts], ignore_index=True)
        test_df = pd.concat([parts[2] for parts in split_parts], ignore_index=True)

        # Separate features (X) and target returns (Y) after split boundaries are fixed.
        x_train = train_df[self.feature_columns].values
        x_val = val_df[self.feature_columns].values
        x_test = test_df[self.feature_columns].values
        y_train = train_df["targetReturns"].values
        y_val = val_df["targetReturns"].values
        y_test = test_df["targetReturns"].values

        # Fit the scaler on train only; validation and test stay out-of-sample.
        x_train = self.scalar.fit_transform(x_train)
        x_val = self.scalar.transform(x_val)
        x_test = self.scalar.transform(x_test)

        return {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "direction_y_train": (y_train > 0).astype(int),
            "direction_y_val": (y_val > 0).astype(int),
            "direction_y_test": (y_test > 0).astype(int),
            "target_y_train": y_train,
            "target_y_val": y_val,
            "target_y_test": y_test,
            "feature_names": self.feature_columns,
            "scalar": self.scalar,
            # Exposes source rows for audits without making Ticker a model feature.
            "split_metadata": {
                "train": train_df[["_source_index", "Ticker"]]
                if "Ticker" in train_df.columns
                else train_df[["_source_index"]],
                "val": val_df[["_source_index", "Ticker"]]
                if "Ticker" in val_df.columns
                else val_df[["_source_index"]],
                "test": test_df[["_source_index", "Ticker"]]
                if "Ticker" in test_df.columns
                else test_df[["_source_index"]],
            },
        }
