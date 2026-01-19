import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers


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
        
        # INIT as NONE for error control
        self.team_lookup = None
        self.champ_lookup = None
        self.model = None
        
        self.df_train = None
        self.df_val = None
        self.df_test = None
        
        
    
    # Split into training & eval datasets
    def split(self, train=0.8, val=0.1):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(self.df))
        rng.shuffle(idx)
        
        n = len(idx)
        n_train = int(train * n)
        n_val = int(val * n)
        
        # Allocating training / val / test data
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train + n_val:]
        
        # initialize vals
        self.df_train = self.df.iloc[train_idx].reset_index(drop=True)
        self.df_val = self.df.iloc[val_idx].reset_index(drop=True)
        self.df_test = self.df.iloc[test_idx].reset_index(drop=True)
        
        return self

        
        
    
    # build vocab + lookups
    def build_looksups(self):
        if self.df_train is None:
            raise RuntimeError("Call split() before build_lookups()")

        # ==
        teams = pd.concat([self.df_train["blueteam"], self.df_train["redteam"]]).astype(str).unique()
       
        champs = pd.concat([
            self.df_train["bluetop"], self.df_train["bluejungle"], self.df_train["bluemid"],
            self.df_train["blueadc"], self.df_train["bluesupport"],
            self.df_train["redtop"], self.df_train["redjungle"], self.df_train["redmid"],
            self.df_train["redadc"], self.df_train["redsupport"],
        ], axis=0).astype(str).unique()
        
        self.team_lookup = layers.StringLookup(vocabulary=teams, mask_token=None, num_oov_indices=1)
        self.champs_lookup = layers.StringLookup(vocabulary=champs, mask_token=None, num_oov_indices=1)
        return self
    
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
    
        
