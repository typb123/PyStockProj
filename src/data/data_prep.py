import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from src.config import REQUIRED_COLUMNS


class DataPreparator:
    """
    Prepare stock data for both scikit-learn (Linear Regression) and XGBoost.
    Allows for optional NaN-dropping, which is required by linear models
    but not strictly needed for XGBoost.
    """

    def __init__(self):
        self.scalar = StandardScaler()
        self.feature_columns: List[str] = []  # Store the names of the features

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training by dropping unnecessary columns only.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df = df.copy()  # Create copy to leave original data intact

        # Drop unnecessary columns
        columns_to_drop = ["Dividends", "Stock Splits"]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )

        return df

    def create_target(self, df: pd.DataFrame, prediction_days: int = 5) -> pd.DataFrame:
        """
        Creates a target variable based on future returns.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing historical stock data.
            prediction_days (int): The number of days ahead to calculate the target.

        Returns:
            pd.DataFrame: DataFrame with an added 'targetReturns' column and intermediate 'target' column removed.
        """
        df = df.copy()

        if "Ticker" in df.columns:
            df["target"] = df.groupby("Ticker")["Close"].shift(-prediction_days)
        else:
            df["target"] = df["Close"].shift(-prediction_days)
        df["targetReturns"] = (df["target"] - df["Close"]) / df["Close"]
        df.dropna(
            subset=["targetReturns"], inplace=True
        )  # Drop rows with NaN values caused by shifting
        df = df.drop(columns=["target"])  # Drop the intermediate column
        return df

    def _split_group(
        self,
        group: pd.DataFrame,
        prediction_days: int,
        val_size: float,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        Prepares data for training by cleaning, creating targets, scaling, and splitting into
        training, validation, and test sets (sequentially, for time-series data).

        Parameters:
            df (pd.DataFrame): Input DataFrame with raw stock data.
            prediction_days (int): Number of days ahead to predict.
            val_size (float): Proportion of data reserved for validation.
            test_size (float): Proportion of data reserved for testing.
            dropna (bool): Whether to drop rows with NaN values (True for linear regression,
                        False for XGBoost which can handle NaNs).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing train, validation, and test sets,
                                feature names, and the fitted scaler.
        """
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be greater than 0 and less than 1.")
        if val_size < 0 or val_size >= 1:
            raise ValueError(
                "val_size must be greater than or equal to 0 and less than 1."
            )
        if val_size + test_size >= 1:
            raise ValueError("val_size + test_size must be less than 1.")

        # Clean data and prepare features
        df = self.prepare_features(df)

        # Create target variables
        df = self.create_target(df, prediction_days)

        # Identify feature columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        self.feature_columns = list(REQUIRED_COLUMNS)

        # Drop and rows with Nan Vals
        df = df.dropna(subset=self.feature_columns + ["targetReturns"])
        df = df.copy()
        df["_source_index"] = df.index

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

        # Separate features (X) and targe (Y)
        x_train = train_df[self.feature_columns].values
        x_val = val_df[self.feature_columns].values
        x_test = test_df[self.feature_columns].values
        y_train = train_df["targetReturns"].values
        y_val = val_df["targetReturns"].values
        y_test = test_df["targetReturns"].values

        # Scale the features
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
