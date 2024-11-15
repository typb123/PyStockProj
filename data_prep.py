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
        df.dropna()

        return df
    
    def createTarget(self, df: pd.DataFrame, predictionDays: int = 1) -> pd.DataFrame:
        """
        Create a target variable based on future returns.
        """
        return





    def getFeatureColumns(self, df: pd.DataFrame) -> List[str]:
        return 




    def prepareForTrain(self, df: pd.DataFrame, 
                        predictionDays: int = 1, 
                        testSize: float = 0.2) -> Dict[str, np.ndarray]:
        



        return
    
    def inverseTransformPredictions(self, predictions: np.ndarray) -> np.ndarray:

        return



