import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers


# Add Categorical Features
CAT_FEATURES = [
    "league", "patch", "year", "split", "playoffs",
    "side", "position",
    "teamname", "champion", "firstPick",
    "ban1","ban2","ban3","ban4","ban5",
    "pick1","pick2","pick3","pick4","pick5",
]

# Add Early Game Numeric Features
NUM_FEATURES = [
    "goldat10","xpat10","csat10",
    "opp_goldat10","opp_xpat10","opp_csat10",
    "golddiffat10","xpdiffat10","csdiffat10",
    "killsat10","assistsat10","deathsat10",
    "opp_killsat10","opp_assistsat10","opp_deathsat10",
]


LABEL = "result"


class Model:
    def __init__(self, df : pd.DataFrame, seed: int = 42):
        self.seed = seed
        
        
        # only keep if they exist (train based off the features we want)
        self.cat_features = [f for f in CAT_FEATURES if f in df.columns]
        self.num_features = [f for f in NUM_FEATURES if f in df.columns]
        # concat
        self.features = self.cat_features + self.num_features
        
    def split(self, train=0.8, val=0.1):
        """_summary_

        Args:
            train (float, optional): % of data that will go to Training the Model | Defaults to 0.8.
            val (float, optional): % of data that will go to Evaluating the Model |. Defaults to 0.1.
        """
        pass
    
    def build_lookups(self):
        """_summary_
        Build a lookup table for all non-numeric features so model can read them
        """
        pass
    
    def _df_to_ds(self, dataframe: pd.DataFrame, shuffle: bool, batch_size: int):
        """
        pd.DataFrame --> tf.data.Dataset

        Args:
            dataframe (pd.DataFrame): _description_
            shuffle (bool): _description_
            batch_size (int): _description_
        """
        pass
    
    def build_model(self, cat_dim:int=8, hidden=(128,64), dropout=0.25, l2=1e-4):
        """
        Args:
            cat_dim (int, optional): _description_. Defaults to 8.
            hidden (tuple, optional): _description_. Defaults to (128,64).
            dropout (float, optional): _description_. Defaults to 0.25.
            l2 (_type_, optional): _description_. Defaults to 1e-4.
        """
        pass
    
    def fit(self, epochs=60, batch_size=256):
        """
        Args:
            epochs (int, optional): _description_. Defaults to 60.
            batch_size (int, optional): _description_. Defaults to 256.
        """
        pass
    
    def evaluate(self, batch_size=256):
        """
        Args:
            batch_size (int, optional): _description_. Defaults to 256.
        """
        pass
    
    def predict_proba(self, row:dict) -> float:
        """
        Args:
            row (dict): _description_

        Returns:
            float: _description_
        """
        return 0.1
