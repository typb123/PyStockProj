import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import REQUIRED_COLUMNS

class DataPreparator:
    """
    Prepare stock data for both scikit-learn (Linear Regression) and XGBoost.
    Allows for optional NaN-dropping, which is required by linear models
    but not strictly needed for XGBoost.
    """
    def __init__(self):
        self.scalar = StandardScaler()
        self.featureColumns: List[str] = [] #Store the names of the features

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training by dropping unnecessary columns only.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
           """
        df = df.copy() #Create copy to leave original data intact

        #Drop unnecessary columns
        columnsToDrop = ['Dividends', 'Stock Splits']
        df = df.drop(columns=[col for col in columnsToDrop if col in df.columns], errors='ignore')

        return df

    def create_target(self, df: pd.DataFrame, predictionDays: int = 5) -> pd.DataFrame:
        """
        Creates a target variable based on future returns.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing historical stock data.
            predictionDays (int): The number of days ahead to calculate the target.

        Returns:
            pd.DataFrame: DataFrame with an added 'targetReturns' column and intermediate 'target' column removed.
    """
        df = df.copy()

        df['target'] = df['Close'].shift(-predictionDays)
        df['targetReturns'] = (df['target'] - df['Close']) / df['Close']
        df.dropna(subset=['targetReturns'], inplace=True)  # Drop rows with NaN values caused by shifting
        df = df.drop(columns=['target']) #Drop the intermediate column
        return df

    def prepare_for_train(
        self,
        df: pd.DataFrame,
        predictionDays: int = 5,
        val_size: float = 0.15,
        test_size:  float = 0.15,
        dropna: bool = True
    )-> Dict[str, np.ndarray]:

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

        #Clean data and prepare features
        df = self.prepare_features(df)

        #Create target variables
        df = self.create_target(df, predictionDays)

        #Identify feature columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        self.featureColumns = list(REQUIRED_COLUMNS)

        #Drop and rows with Nan Vals
        df = df.dropna()

        #Separate features (X) and targe (Y)
        X = df[self.featureColumns].values
        Y = df['targetReturns'].values

        # Scale the features
        X = self.scalar.fit_transform(X)

        #Split the data into training and test sets
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=test_size, shuffle=False)

        return {
            'XTrain': XTrain,
            'XTest': XTest,
            'YTrain': YTrain,
            'YTest': YTest,
            'featureNames': self.featureColumns,
            'scalar': self.scalar
        }
