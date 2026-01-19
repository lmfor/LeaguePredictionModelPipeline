import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from src.data import DatasetConfig, RetrieveData


# Prepare df

FEATURE_COLS = [
    "league",
    "blueteam", "redteam",
    "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport",
    "redtop", "redjungle", "redmid", "redadc", "redsupport",
]

LABEL = "result"


class Model:
    def __init__(self, df : pd.DataFrame, seed: int = 42):
        self.seed = seed
        
        # keep what i need
        self.df = df.dropna(subset=FEATURE_COLS + [LABEL]).copy()
        self.df[LABEL] = self.df[LABEL].astype("int32")
        
        # setup later
        self.team_lookup = None
        self.champ_lookup = None
        self.model = None
        
        self.df_train = None
        self.df_val = None
        self.df_test = None
        
        
    
    # Split into training & eval datasets
    def split(self, train=0.8, val=0.1):
        pass
    
    # build vocab + lookups
    def build_looksups(self):
        if self.df_train is None:
            raise RuntimeError("Call split() before build_lookups()")
    # ...
    
    def _df_to_ds(
            self, 
            dataframe : pd.DataFrame, 
            shuffle : bool, 
            batch_size: int
        ):
        pass
    
    
    # Construct model
    def build_model(
            self, 
            team_dim : int = 8, 
            champ_dim: int = 16, 
            hidden=(128,68), 
            dropout=0.25
        ):
        pass
    
    # train & eval & save
    def fit(self, epochs=60, batch_size=128):
        if self.model is None:
            raise RuntimeError("Call build_model() before fit()")
        
        pass
    
    def evaluate(self, batch_size=128):
        pass
    
    def save(self, path : str = "models/draft_winner_tf.keras"):
        if self.model is None:
            raise RuntimeError("[SAVE ERROR] ! No model to save.")
        
        pass
    
    
    # Prediction
    def predict_proba_one(self, row:dict) -> float:
        if self.model is None:
            raise RuntimeError("[PREDICTION ERROR] ! Train/build model first.")
        return 0.1
    
        
