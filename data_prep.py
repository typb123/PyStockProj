import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreparator: 
    """Prepare stock data for scikit learn models"""
    def __init__(self):
        self.scalar = StandardScaler()
        self.featureColumns: List[str] = [] #Store the names of the features

    def prepareFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Input: pd.DataFrame
           Output: Cleaned pd.DataFrame"""
        df = df.copy() #Create copy to leave original data intact

        #Drop unnecessary columns
        columnsToDrop = ['Dividends', 'Stock Splits']
        df = df.drop(columns=[col for col in columnsToDrop if col in df.columns], errors='ignore')

        #Drop any rows with NaN values
        df.dropna(inplace=True)

        return df
    
    def createTarget(self, df: pd.DataFrame, predictionDays: int = 5) -> pd.DataFrame:
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

    def getFeatureColumns(self, df: pd.DataFrame) -> List[str]:
        '''
        Identifies feature columns and excludes columns like 'Close' and target variables 
        '''
        excludeColumns = ['High', 'Low', 'Close', 'Capital Gains', 'targetReturns', 'Ticker']
        return [col for col in df.columns if col not in excludeColumns] 

    def prepareForTrain(self, df: pd.DataFrame, 
                        predictionDays: int = 5, 
                        testSize: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepares data for training. Features like scaling and train-test splitting for scikit learn.

        Parameters:
            pd.DataFrame
            predictionDays (int): number of days ahead to predict 
            testSize (float): Proportion of data reserved for testing

        Returns:
            Dict[str, np.ndarray]: Dictionary containing training and test sets and feature names 
        """

        #Clean data and prepare features
        df = self.prepareFeatures(df)
        
        #Create target variables
        df = self.createTarget(df, predictionDays)

        #Identify feature columns
        self.featureColumns = self.getFeatureColumns(df)

        #Drop and rows with Nan Vals
        df = df.dropna()

        #Separate features (X) and targe (Y)
        X = df[self.featureColumns].values
        Y = df['targetReturns'].values

        # Scale the features
        X = self.scalar.fit_transform(X)

        #Split the data into training and test sets
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=testSize, shuffle=False)

        return {
            'XTrain': XTrain,
            'XTest': XTest,
            'YTrain': YTrain,
            'YTest': YTest,
            'featureNames': self.featureColumns,
            'scalar': self.scalar
        } 
    

